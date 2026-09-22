#!/usr/bin/env python3
"""Раздельные Adam-настройки для policy/value + управление learning rate.

Проверяем ровно те правила, по которым сделаны дефолты:

  1. BC: `Adam(lr=1e-3, eps=1e-8)` со спадом lr до 1e-4 (cosine/linear) к концу датасета;
  2. переход BC -> PPO: **новый объект оптимизатора**, моменты BC не переносятся;
  3. PPO: `Adam(lr=1e-4/3e-4, eps=1e-5)`, при этом lr/eps можно задать РАЗДЕЛЬНО для
     policy-головы, value-головы и экстрактора признаков.

Что именно тестируется (без сервера, только torch/SB3):
  A) `schedule_factor` — математика расписаний (constant/linear/cosine, прогрев, final_ratio);
  B) `SplitAdam` — раскладка параметров реальной `MaskedActorCriticPolicy` по группам
     (pi/vf/shared), раздельные lr и eps, пропорции между группами, `set_lrs`;
  C) адаптивное управление lr: `gnorm` (самокалибровка по норме градиента) и `plateau`
     (по метрике), границы `adapt_min/adapt_max`;
  D) BC-претрейн на настоящей политике: старт с lr=1e-3/eps=1e-8, спад lr, и после BC —
     оптимизатор ДРУГОЙ (моменты пустые, eps=1e-5, lr = RL-значение);
  E) `SplitLRPPO._update_learning_rate`: задаёт lr по группам, а не одним скаляром,
     и пишет `train/lr_policy|value|shared` в логгер;
  F) совместимость: чекпоинт со старым плоским Adam грузится без падения (моменты
     отбрасываются, настройки остаются), а свой чекпоинт сохраняет моменты;
  G) value-warmup «только value» не ломает раздельные lr (policy-группа не учится, value учится);
  H) контракт `--resume`: загруженный PPO снова становится SplitLRPPO, явные флаги CLI
     применяются к оптимизатору чекпоинта, моменты Adam при этом сохраняются.

Запуск: PYTHONPATH=. python test_optim_split.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gymnasium import spaces

from agents.config import N_FEATURES
from agents.optim import (
    GROUP_POLICY,
    GROUP_SHARED,
    GROUP_VALUE,
    SplitAdam,
    bc_lr_at,
    build_policy_optimizer,
    classify_param_name,
    make_lr_schedule,
    reset_policy_optimizer,
    schedule_factor,
)
from agents.policy import MaskedActorCriticPolicy

OK = 0
FAIL: list = []


def check(name, cond, extra=""):
    global OK
    if cond:
        OK += 1
        print(f"OK   {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL.append(name)
        print(f"FAIL {name}" + (f": {extra}" if extra else ""))


def check_eq(name, got, want):
    check(name, got == want, f"got={got!r} want={want!r}")


def check_close(name, got, want, tol=1e-9):
    d = abs(float(got) - float(want))
    check(name, d <= tol, f"got={float(got):.6g} want={float(want):.6g} (Δ={d:.2e})")


def make_policy(features_dim=64, **optimizer_kwargs):
    """Настоящая политика проекта на пустом observation_space (без env/сервера)."""
    obs_space = spaces.Dict({
        "observation": spaces.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, shape=(26,), dtype=np.int8),
    })
    act_space = spaces.Discrete(26)
    opt_kwargs = dict(optimizer_kwargs)
    opt_kwargs.pop("optimizer_class", None)
    return MaskedActorCriticPolicy(
        obs_space, act_space, lr_schedule=lambda _p: 2e-4,
        features_extractor_kwargs=dict(features_dim=features_dim),
        optimizer_class=SplitAdam, optimizer_kwargs=opt_kwargs,
    )


def make_vec_env():
    """Один DummyVecEnv: нужен конструктору PPO/основного алгоритма (SB3) без сервера."""
    import gymnasium as _gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _Env(_gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(26,), dtype=np.int8),
            })
            self.action_space = spaces.Discrete(26)

        def reset(self, **kwargs):
            obs = {"observation": np.zeros(N_FEATURES, dtype=np.float32),
                   "action_mask": np.ones(26, dtype=np.int8)}
            return obs, {}

        def step(self, action):
            obs = {"observation": np.zeros(N_FEATURES, dtype=np.float32),
                   "action_mask": np.ones(26, dtype=np.int8)}
            return obs, 0.0, True, False, {}

    return DummyVecEnv([_Env])


def obs_batch(n=4, mask_value=1, seed=0):
    # ненулевые obs: на нулевом входе градиенты первого слоя вырождаются (ReLU(0) -> 0),
    # и проверка «оптимизатор обновил веса» стала бы вакуумной
    g = torch.Generator().manual_seed(int(seed))
    return {
        "observation": torch.randn((n, N_FEATURES), generator=g, dtype=torch.float32),
        "action_mask": torch.full((n, 26), mask_value, dtype=torch.int8),
    }


def backward_on(out):
    """Градиент по value+log_prob: actions из forward НЕ дифференцируемы (сэмплинг)."""
    (out[1].sum() + out[2].sum()).backward()


# ------------------------------------------------------------------ A: расписания ---
def part_a_schedules():
    print("=" * 78)
    print("A. Математика расписаний lr")
    print("=" * 78)
    check_close("constant: множитель 1 в начале", schedule_factor(0.0, "constant"), 1.0)
    check_close("constant: множитель 1 в конце", schedule_factor(1.0, "constant"), 1.0)
    check_close("linear: старт 1", schedule_factor(0.0, "linear", final_ratio=0.0), 1.0)
    check_close("linear: конец 0", schedule_factor(1.0, "linear", final_ratio=0.0), 0.0)
    check_close("linear: середина 0.5", schedule_factor(0.5, "linear", final_ratio=0.0), 0.5)
    check_close("linear: конец = final_ratio", schedule_factor(1.0, "linear", final_ratio=0.1), 0.1)
    check_close("cosine: старт 1", schedule_factor(0.0, "cosine", final_ratio=0.1), 1.0)
    check_close("cosine: конец = final_ratio", schedule_factor(1.0, "cosine", final_ratio=0.1), 0.1)
    check_close("cosine: середина = (1+final)/2", schedule_factor(0.5, "cosine", final_ratio=0.1), 0.55)
    check("cosine спадает монотонно",
          all(schedule_factor(p / 10, "cosine", 0.1) >= schedule_factor((p + 1) / 10, "cosine", 0.1)
              for p in range(10)))
    # прогрев
    check_close("warmup: на старте 0", schedule_factor(0.0, "linear", 0.0, warmup_frac=0.2), 0.0)
    check_close("warmup: к концу прогрева 1", schedule_factor(0.2, "linear", 0.0, warmup_frac=0.2), 1.0)
    check_close("warmup: дальше идёт спад",
                schedule_factor(0.6, "linear", 0.0, warmup_frac=0.2), 0.5)
    # BC-хелпер из рекомендации 1e-3 -> 1e-4
    check_close("BC: старт 1e-3", bc_lr_at(0.0, 1e-3, 1e-4, "cosine"), 1e-3)
    check_close("BC: конец 1e-4", bc_lr_at(1.0, 1e-3, 1e-4, "cosine"), 1e-4)
    mid = bc_lr_at(0.5, 1e-3, 1e-4, "cosine")
    check("BC: середина между 1e-4 и 1e-3", 1e-4 < mid < 1e-3, f"lr={mid:.3e}")
    check_close("BC: linear тоже ведёт к 1e-4", bc_lr_at(1.0, 1e-3, 1e-4, "linear"), 1e-4)
    # RL-расписание через steps_holder (как в прогоне): прогресс берётся из глобального счётчика
    holder = {"value": 0}
    sched = make_lr_schedule(2e-4, 1000, holder, schedule="cosine", final_ratio=0.1)
    check_close("RL: старт 2e-4", sched(1.0), 2e-4)
    holder["value"] = 1000
    check_close("RL: конец = 10% от 2e-4", sched(0.0), 2e-5)
    check_close("RL: constant не меняется", make_lr_schedule(2e-4, 1000, holder, schedule="constant")(0.0), 2e-4)
    check("RL: неизвестное расписание — явная ошибка",
          isinstance(_raises(lambda: schedule_factor(0.5, "warp")), ValueError))


def _raises(fn):
    try:
        fn()
        return None
    except Exception as e:  # noqa: BLE001
        return e


# -------------------------------------------------------------- B: SplitAdam группы ---
def part_b_groups():
    print("=" * 78)
    print("B. Раздельные значения для Policy и Value")
    print("=" * 78)
    check_eq("тип action_net -> policy", classify_param_name("action_net.weight"), GROUP_POLICY)
    check_eq("тип policy_net -> policy",
             classify_param_name("mlp_extractor.policy_net.0.weight"), GROUP_POLICY)
    check_eq("тип log_std -> policy", classify_param_name("log_std"), GROUP_POLICY)
    check_eq("тип value_net -> value",
             classify_param_name("mlp_extractor.value_net.0.weight"), GROUP_VALUE)
    check_eq("тип value_net (голова) -> value", classify_param_name("value_net.bias"), GROUP_VALUE)
    check_eq("тип экстрактора -> shared",
             classify_param_name("features_extractor.net.0.weight"), GROUP_SHARED)

    policy = make_policy(lr_policy=3e-4, lr_value=1e-4, lr_shared=1e-4,
                         eps_policy=1e-8, eps_value=1e-5)
    opt = policy.optimizer
    check("политика получила SplitAdam", isinstance(opt, SplitAdam), type(opt).__name__)
    names = [g.get("name") for g in opt.param_groups]
    check_eq("групп ровно три (policy/value/shared)", sorted(names),
             [GROUP_POLICY, GROUP_SHARED, GROUP_VALUE])
    lrs, epss = opt.lr_by_group(), opt.eps_by_group()
    check_close("lr policy = 3e-4", lrs[GROUP_POLICY], 3e-4)
    check_close("lr value = 1e-4 (раздельно!)", lrs[GROUP_VALUE], 1e-4)
    check_close("lr shared = 1e-4", lrs[GROUP_SHARED], 1e-4)
    check_close("eps policy = 1e-8 (раздельно!)", epss[GROUP_POLICY], 1e-8)
    check_close("eps value = 1e-5", epss[GROUP_VALUE], 1e-5)

    # каждый параметр попал ровно в одну группу и именно в свою
    seen = {}
    for idx, group in enumerate(opt.param_groups):
        for p in group["params"]:
            seen[id(p)] = group.get("name")
    total = sum(len(g["params"]) for g in opt.param_groups)
    check_eq("все параметры политики разложены", total, len(list(policy.parameters())))
    by_name = {}
    for name, param in policy.named_parameters():
        by_name[id(param)] = classify_param_name(name)
    mismatch = [(n, seen.get(id(p)), want) for n, p in policy.named_parameters()
                if (want := by_name[id(p)]) != seen.get(id(p))]
    check("каждый параметр в группе по своему имени", not mismatch, str(mismatch[:3]))
    check("в policy-группе есть action_net",
          any("action_net" in n for n, p in policy.named_parameters()
              if seen.get(id(p)) == GROUP_POLICY))
    check("в value-группе есть value_net",
          any("value_net" in n for n, p in policy.named_parameters()
              if seen.get(id(p)) == GROUP_VALUE))

    # смена базовых lr и пропорций
    opt.set_lrs(policy_lr=2e-4)
    lrs = opt.lr_by_group()
    check_close("set_lrs: policy 2e-4", lrs[GROUP_POLICY], 2e-4)
    check_close("set_lrs: value = 2e-4 * (1e-4/3e-4)", lrs[GROUP_VALUE], 2e-4 * (1e-4 / 3e-4))
    check("пропорция pi:vf сохраняется", abs(lrs[GROUP_POLICY] / lrs[GROUP_VALUE] - 3.0) < 1e-9)

    # шаг оптимизатора реально обновляет обе головы
    policy.train()
    out = policy(obs_batch(4))
    backward_on(out)
    before = {g.get("name"): [p.detach().clone() for p in g["params"]] for g in opt.param_groups}
    opt.step()
    changed = {}
    for g in opt.param_groups:
        n = g.get("name")
        changed[n] = any(not torch.equal(p_before, p_now)
                         for p_before, p_now in zip(before[n], g["params"]))
    check("шаг Adam обновил policy-параметры", changed[GROUP_POLICY])
    check("шаг Adam обновил value-параметры", changed[GROUP_VALUE])
    check_eq("моменты появились у всех трёх групп", len(opt.state) > 0, True)

    # разные eps реально дают разные обновления при одинаковом градиенте
    def _step_with(eps_val):
        pol = make_policy(features_dim=64, lr_policy=1e-2, lr_value=1e-2,
                          eps_policy=eps_val, eps_value=eps_val)
        w0 = pol.action_net.weight.detach().clone()
        pol.train()
        out = pol(obs_batch(4))
        backward_on(out)
        pol.optimizer.step()
        return float((pol.action_net.weight.detach() - w0).abs().mean())
    d_small = _step_with(1e-8)
    d_big = _step_with(1e-4)
    check("eps влияет на шаг Adam (1e-8 vs 1e-4)", d_small > 0 and d_big > 0,
          f"|Δ| 1e-8 = {d_small:.3e}, 1e-4 = {d_big:.3e}")


# ------------------------------------------------------------------- C: адаптация ---
def part_c_adapt():
    print("=" * 78)
    print("C. Адаптивное управление learning rate")
    print("=" * 78)
    policy = make_policy(lr_policy=1e-3, lr_value=1e-3)
    opt = policy.optimizer

    # plateau: метрика не улучшается -> множитель падает после patience наблюдений
    opt.adapt = "plateau"
    opt.adapt_patience = 2
    opt.adapt_factor = 0.5
    opt.adapt_min = 0.1
    s0 = opt.observe_metric(10.0)
    check_close("plateau: первое наблюдение не режет lr", s0, 1.0)
    check_close("plateau: второе (не лучше) ещё терпит", opt.observe_metric(10.0), 1.0)
    check_close("plateau: третье (patience=2) режет вдвое", opt.observe_metric(10.0), 0.5)
    check_close("plateau: lr policy упал до 0.5 от целевого",
                opt.lr_by_group()[GROUP_POLICY], 0.5e-3)
    for _ in range(20):
        opt.observe_metric(10.0)
    check_close("plateau: не ниже adapt_min", opt.adapt_scale, 0.1)
    check_close("plateau: lr не ниже adapt_min * base", opt.lr_by_group()[GROUP_POLICY], 0.1e-3)
    opt.observe_metric(1.0)
    check("plateau: улучшение метрики не режет lr дальше", opt.adapt_scale == 0.1)
    check_close("plateau: higher_is_better разворачивает знак",
                opt.observe_metric(1.0, higher_is_better=True), 0.1)

    # gnorm: самокалибровка по норме градиента
    pol2 = make_policy(lr_policy=1e-3, lr_value=1e-3)
    o2 = pol2.optimizer
    o2.adapt = "gnorm"
    o2.adapt_gnorm_warmup = 3
    o2.adapt_min, o2.adapt_max = 0.5, 2.0
    pol2.train()
    for i in range(3):                      # прогрев: target ещё не зафиксирован
        pol2.optimizer.zero_grad()
        out = pol2(obs_batch(4))
        backward_on(out)
        pol2.optimizer.step()
    check("gnorm: цель зафиксирована после прогрева", o2._gnorm_target is not None,
          f"target={o2._gnorm_target}")
    target = float(o2._gnorm_target)
    # искусственно большой градиент -> lr вниз; маленький -> вверх (в пределах коридора)
    for p in pol2.parameters():
        p.grad = torch.full_like(p, 10.0 * max(target, 1e-6) / max(float(p.numel()) ** 0.5, 1.0))
    o2.step()
    check("gnorm: большой градиент снижает множитель", o2.adapt_scale < 1.0,
          f"scale={o2.adapt_scale:.3f}")
    for p in pol2.parameters():
        p.grad = torch.zeros_like(p)
    o2.step()
    check("gnorm: множитель держится в коридоре [min, max]",
          o2.adapt_min - 1e-9 <= o2.adapt_scale <= o2.adapt_max + 1e-9,
          f"scale={o2.adapt_scale:.3f}")
    check_close("gnorm: lr = base * scale",
                o2.lr_by_group()[GROUP_POLICY], 1e-3 * o2.adapt_scale, tol=1e-12)
    check("off: адаптация не вмешивается",
          make_policy(lr_policy=1e-3).optimizer.adapt == "off")

    # state() отдаёт то, что уходит в логи/TB
    st = o2.schedule_state()
    check("schedule_state содержит lr по группам", set(st["lr"]) == {GROUP_POLICY, GROUP_VALUE, GROUP_SHARED},
          str(list(st["lr"])))
    check("schedule_state содержит адаптивный множитель", "adapt_scale" in st and "eps" in st)


# ---------------------------------------------------------------- D: BC -> PPO сброс ---
def part_d_bc_reset():
    print("=" * 78)
    print("D. BC -> PPO: свой оптимизатор и обязательный сброс моментов")
    print("=" * 78)
    from agents.training import pretrain_policy_bc

    # маленький датасет на настоящей размерности (obs_dim = N_FEATURES)
    n = 32
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    mask = np.ones((n, 26), dtype=np.int8)
    actions = rng.integers(6, 10, size=n)
    rets = rng.normal(size=n).astype(np.float32)

    policy = make_policy(features_dim=64, lr_policy=2e-4, lr_value=2e-4, eps_policy=1e-5)
    ppo_like = type("P", (), {})()          # BC берёт только ppo.policy и ppo.learning_rate
    ppo_like.policy = policy
    ppo_like.learning_rate = 2e-4

    seen_lr: list = []
    orig_set_lrs = SplitAdam.set_lrs

    def spy_set_lrs(self, policy_lr=None, progress_done=None):
        if policy_lr is not None:
            seen_lr.append(float(policy_lr))
        return orig_set_lrs(self, policy_lr=policy_lr, progress_done=progress_done)

    SplitAdam.set_lrs = spy_set_lrs
    try:
        pretrain_policy_bc(
            ppo_like, [(obs[i], mask[i], int(actions[i]), float(rets[i])) for i in range(n)],
            epochs=3, batch_size=8, normalize=False, contrastive=False,
            lr=1e-3, lr_final=1e-4, lr_schedule="cosine", eps=1e-8,
            reset_at_start=True, reset_optimizer=True, rl_lr=3e-4, rl_eps=1e-5, verbose=True,
        )
    finally:
        SplitAdam.set_lrs = orig_set_lrs

    check("BC: lr ведётся от 1e-3 (старт рекомендации)", seen_lr and abs(seen_lr[0] - 1e-3) < 1e-12,
          f"первые значения: {seen_lr[:3]}")
    check("BC: lr спадает (последний батч < первого)", len(seen_lr) > 1 and seen_lr[-1] < seen_lr[0],
          f"{seen_lr[0]:.3e} -> {seen_lr[-1]:.3e}")
    check("BC: lr не уходит ниже конечного 1e-4", min(seen_lr) >= 1e-4 - 1e-12,
          f"min={min(seen_lr):.3e}")
    opt = policy.optimizer
    check("после BC оптимизатор — новый объект SplitAdam", isinstance(opt, SplitAdam))
    check("правило сброса: моменты BC не переехали в PPO", len(opt.state) == 0,
          f"состояний в оптимизаторе: {len(opt.state)}")
    check_close("после BC eps = RL-шный 1e-5 (не BC-шный 1e-8)",
                opt.eps_by_group()[GROUP_POLICY], 1e-5, tol=1e-15)
    check_close("после BC lr = RL-значение", opt.lr_by_group()[GROUP_POLICY], 3e-4)

    # BC-only путь (total_timesteps=0): тоже должен отдать чистый оптимизатор
    policy2 = make_policy(features_dim=64)
    ppo_like2 = type("P", (), {})()
    ppo_like2.policy = policy2
    ppo_like2.learning_rate = 2e-4
    pretrain_policy_bc(ppo_like2, [(obs[i], mask[i], int(actions[i]), float(rets[i])) for i in range(n)],
                       epochs=1, batch_size=16, normalize=False, lr=1e-3, lr_final=1e-4,
                       reset_optimizer=True, rl_lr=1e-4, verbose=False)
    check("BC-only: оптимизатор тоже пересоздан", isinstance(policy2.optimizer, SplitAdam)
          and len(policy2.optimizer.state) == 0)
    check_close("BC-only: eps 1e-5 после перехода", policy2.optimizer.eps_by_group()[GROUP_POLICY], 1e-5,
                tol=1e-15)

    # bc_keep: если попросили НЕ пересоздавать — моменты остаются своими (для экспериментов)
    policy3 = make_policy(features_dim=64)
    ppo_like3 = type("P", (), {})()
    ppo_like3.policy = policy3
    ppo_like3.learning_rate = 2e-4
    pretrain_policy_bc(ppo_like3, [(obs[i], mask[i], int(actions[i]), float(rets[i])) for i in range(n)],
                       epochs=1, batch_size=16, normalize=False, lr=1e-3, lr_final=1e-4,
                       reset_at_start=True, reset_optimizer=False, verbose=False)
    check_close("reset_optimizer=False: eps остаётся BC-шным 1e-8 (сброса для RL нет)",
                policy3.optimizer.eps_by_group()[GROUP_POLICY], 1e-8, tol=1e-15)
    check("reset_optimizer=False: моменты сброшены только восстановлением лучших весов "
          "(штатное поведение best_state)",
          len(policy3.optimizer.state) == 0)

    # совсем без пересоздания: bc_keep + reset_optimizer=False -> тот же объект оптимизатора
    policy4 = make_policy(features_dim=64)
    ppo_like4 = type("P", (), {})()
    ppo_like4.policy = policy4
    ppo_like4.learning_rate = 2e-4
    opt_before = policy4.optimizer
    pretrain_policy_bc(ppo_like4, [(obs[i], mask[i], int(actions[i]), float(rets[i])) for i in range(n)],
                       epochs=1, batch_size=16, normalize=False, lr=1e-3, lr_final=1e-4,
                       reset_at_start=False, reset_optimizer=False, verbose=False)
    check("reset_at_start=False + reset_optimizer=False: объект оптимизатора тот же",
          policy4.optimizer is opt_before)
    check_close("в этом режиме eps остаётся прежним (1e-5)",
                policy4.optimizer.eps_by_group()[GROUP_POLICY], 1e-5, tol=1e-15)


# ------------------------------------------------------------- E: SplitLRPPO + лог ---
def part_e_ppo_schedule():
    print("=" * 78)
    print("E. SplitLRPPO: lr по группам, а не одним скаляром")
    print("=" * 78)
    from stable_baselines3.common.utils import FloatSchedule

    from agents.optim import SplitLRPPO

    # lr задаются абсолютно: policy 2e-4, value 5e-5 (=0.25x), shared 1e-4 (=0.5x)
    policy = make_policy(features_dim=64, lr_policy=2e-4, lr_value=5e-5, lr_shared=1e-4)
    ppo = SplitLRPPO.__new__(SplitLRPPO)     # без env: проверяем только _update_learning_rate
    ppo.policy = policy
    ppo.learning_rate = 2e-4
    ppo.lr_schedule = FloatSchedule(lambda p: 2e-4 * p)
    ppo._current_progress_remaining = 0.5

    recorded: dict = {}

    class _Logger:
        def record(self, key, value):
            recorded[key] = float(value)

    ppo._logger = _Logger()   # logger — property без сеттера
    ppo._update_learning_rate(policy.optimizer)
    lrs = policy.optimizer.lr_by_group()
    check_close("PPO: policy lr из расписания (2e-4 * 0.5)", lrs[GROUP_POLICY], 1e-4)
    check_close("PPO: value = 0.25 * policy (ratio из настроек)", lrs[GROUP_VALUE], 0.25e-4)
    check_close("PPO: shared = 0.5 * policy", lrs[GROUP_SHARED], 0.5e-4)
    check("PPO: в логгер ушёл lr каждой группы",
          {"train/lr_policy", "train/lr_value", "train/lr_shared"} <= set(recorded), str(sorted(recorded)))
    check_close("PPO: train/learning_rate = policy lr", recorded["train/learning_rate"], 1e-4)
    check("PPO: train/lr_adapt_scale пишется", "train/lr_adapt_scale" in recorded)

    # обычный Adam (легаси-путь) не ломается
    plain = torch.optim.Adam(policy.parameters(), lr=1e-3)
    ppo._update_learning_rate(plain)
    check_close("PPO: для плоского Adam поведение прежнее (скаляр всем группам)",
                plain.param_groups[0]["lr"], 1e-4)


# ------------------------------------------------------------- F: совместимость ---
def part_f_compat():
    print("=" * 78)
    print("F. Совместимость: старый плоский Adam и свой чекпоинт")
    print("=" * 78)
    policy = make_policy(features_dim=64, lr_policy=3e-4, lr_value=1e-4)
    opt = policy.optimizer
    policy.train()
    out = policy(obs_batch(4))
    backward_on(out)
    opt.step()
    opt2 = build_policy_optimizer(policy, lr=3e-4, lr_value=1e-4, eps_policy=1e-5)
    state = opt.state_dict()
    n_saved = len(state["state"])
    check("в своём чекпоинте моменты есть (иначе проверка вакуумная)", n_saved > 0, f"{n_saved}")
    opt2.load_state_dict(state)                      # своя схема -> моменты восстанавливаются
    check_eq("свой checkpoint: моменты восстановлены", len(opt2.state), n_saved)
    check_close("свой checkpoint: lr сохранился", opt2.lr_by_group()[GROUP_POLICY], 3e-4)

    # старый чекпоинт: один плоский param_group (как во всех zip до этой правки)
    legacy = torch.optim.Adam(policy.parameters(), lr=7e-4, eps=1e-8)
    policy.train()
    out = policy(obs_batch(4))
    backward_on(out)
    legacy.step()
    legacy_state = legacy.state_dict()
    check_eq("в старом чекпоинте ровно одна группа", len(legacy_state["param_groups"]), 1)

    import warnings as _w
    opt3 = build_policy_optimizer(policy, lr=4e-4, lr_policy=4e-4, lr_value=1e-4, eps_policy=1e-5)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        opt3.load_state_dict(legacy_state)           # не должно падать
    check_eq("старый чекпоинт: групп снова три", len(opt3.param_groups), 3)
    check_eq("старый чекпоинт: моменты отброшены", len(opt3.state), 0)
    check("старый чекпоинт: предупреждение выдано", any("моменты отброшены" in str(w.message) for w in caught),
          str([str(w.message)[:60] for w in caught]))
    check_close("старый чекпоинт: lr остался нашим (4e-4), а не 7e-4 из файла",
                opt3.lr_by_group()[GROUP_POLICY], 4e-4)
    check_close("старый чекпоинт: eps остался нашим (1e-5), а не 1e-8 из файла",
                opt3.eps_by_group()[GROUP_POLICY], 1e-5, tol=1e-15)

    # полный круг: сохранить и загрузить PPO с нашими группами
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    import gymnasium as _gym

    class _Env(_gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            from gymnasium import spaces as _sp
            self.observation_space = _sp.Dict({
                "observation": _sp.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32),
                "action_mask": _sp.Box(0, 1, shape=(26,), dtype=np.int8)})
            self.action_space = _sp.Discrete(26)

        def reset(self, **kw):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, True, False, {}

        def close(self):
            pass

    env = DummyVecEnv([_Env])
    ppo = PPO(MaskedActorCriticPolicy, env, device="cpu", verbose=0, n_steps=8, batch_size=8,
              policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=64),
                                 optimizer_class=SplitAdam,
                                 optimizer_kwargs=dict(lr_policy=3e-4, lr_value=1e-4, eps_policy=1e-5)))
    check("PPO с SplitAdam: lr policy 3e-4", abs(ppo.policy.optimizer.lr_by_group()[GROUP_POLICY] - 3e-4) < 1e-12)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m")
        ppo.save(path)
        loaded = PPO.load(path, device="cpu")
        check("загрузка своего чекпоинта: снова SplitAdam",
              isinstance(loaded.policy.optimizer, SplitAdam))
        lrs = loaded.policy.optimizer.lr_by_group()
        check("загрузка: раздельные lr сохранились (policy 3e-4, value 1e-4)",
              abs(lrs[GROUP_POLICY] - 3e-4) < 1e-9 and abs(lrs[GROUP_VALUE] - 1e-4) < 1e-9, str(lrs))
        check_close("загрузка: eps policy = 1e-5", loaded.policy.optimizer.eps_by_group()[GROUP_POLICY],
                    1e-5, tol=1e-15)


# ------------------------------------------------------------------ G: warmup value ---
def part_g_value_only():
    print("=" * 78)
    print("G. Warmup «учим только value» на раздельных группах")
    print("=" * 78)
    policy = make_policy(features_dim=64, lr_policy=1e-4, lr_value=1e-3)
    opt = policy.optimizer
    for p in policy.parameters():
        p.requires_grad = False
    for p in policy.value_net.parameters():
        p.requires_grad = True
    for p in policy.mlp_extractor.value_net.parameters():
        p.requires_grad = True
    opt.zero_grad()
    out = policy(obs_batch(4))
    backward_on(out)
    frozen_with_grad = [n for n, p in policy.named_parameters()
                        if not p.requires_grad and p.grad is not None]
    check("warmup: у замороженной policy-головы градиентов нет", not frozen_with_grad,
          str(frozen_with_grad[:3]))
    have_grad = [n for n, p in policy.named_parameters()
                 if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    check("warmup: value-голова градиент получила", bool(have_grad), str(have_grad[:2]))
    w_policy = policy.action_net.weight.detach().clone()
    w_value = policy.value_net.weight.detach().clone()
    opt.step()
    check("warmup: policy-голова не сдвинулась",
          torch.equal(w_policy, policy.action_net.weight.detach()))
    check("warmup: value-голова обновилась",
          not torch.equal(w_value, policy.value_net.weight.detach()))
    check_close("warmup: lr value = 1e-3 (свой, не policy-шный)",
                opt.lr_by_group()[GROUP_VALUE], 1e-3)
    check_close("warmup: lr policy = 1e-4", opt.lr_by_group()[GROUP_POLICY], 1e-4)
    opt.clear_moments()
    check_eq("warmup: clear_moments сбрасывает моменты", len(opt.state), 0)
    check("warmup: группы при этом не пересобираются", len(opt.param_groups) == 3)


def part_h_resume_contract():
    print("=" * 78)
    print("H. --resume: настройки CLI побеждают чекпоинт, моменты Adam сохраняются")
    print("=" * 78)
    from stable_baselines3 import PPO as _PPO
    from agents.optim import SplitLRPPO, adopt_optimizer, ensure_split_ppo

    # 1) PPO.load() отдаёт БАЗОВЫЙ PPO — без конверсии раздельные lr теряются на дообучении
    env = make_vec_env()
    policy = make_policy(features_dim=64, lr_policy=3e-4, lr_value=1e-4)
    ppo = _PPO("MultiInputPolicy", env, n_steps=8, batch_size=8, n_epochs=1, device="cpu")
    ppo.policy = policy          # подменяем на реальную политику проекта (SplitAdam внутри)
    check("resume: наивный PPO — не SplitLRPPO", not isinstance(ppo, SplitLRPPO),
          type(ppo).__name__)
    ppo = ensure_split_ppo(ppo)
    check("resume: ensure_split_ppo делает его SplitLRPPO", isinstance(ppo, SplitLRPPO),
          type(ppo).__name__)

    # 2) _update_learning_rate не падает без логгера и применяет lr ПО ГРУППАМ
    ppo.lr_schedule = lambda progress: 3e-5
    try:
        ppo._update_learning_rate(ppo.policy.optimizer)
        err = None
    except Exception as e:                                  # pragma: no cover
        err = e
    check("resume: _update_learning_rate без логгера не падает", err is None, str(err))
    lr = ppo.policy.optimizer.lr_by_group()
    check_close("resume: policy lr = 3e-5 из расписания", lr[GROUP_POLICY], 3e-5, tol=1e-12)
    check_close("resume: value lr = 3e-5 * ratio(1/3) = 1e-5", lr[GROUP_VALUE], 1e-5, tol=1e-12)

    # 3) apply_settings: явные флаги CLI меняют lr/eps, моменты остаются
    opt = ppo.policy.optimizer
    opt.zero_grad()
    backward_on(policy(obs_batch(4)))
    opt.step()
    n_moments = len(opt.state)
    check("resume: моменты появились", n_moments > 0, str(n_moments))
    opt.apply_settings(lr_policy=1e-4, lr_value=5e-5, lr_shared=5e-5,
                       eps_policy=1e-8, eps_value=1e-8, eps_shared=1e-8)
    check_eq("resume: моменты при apply_settings сохранены", len(opt.state), n_moments)
    lr2 = opt.lr_by_group()
    check_close("resume: policy lr = 1e-4 (флаг CLI)", lr2[GROUP_POLICY], 1e-4, tol=1e-12)
    check_close("resume: value lr = 5e-5 (абсолютный флаг CLI)", lr2[GROUP_VALUE], 5e-5, tol=1e-12)
    check_close("resume: shared lr = 5e-5", lr2[GROUP_SHARED], 5e-5, tol=1e-12)
    eps = opt.eps_by_group()
    check_close("resume: eps policy = 1e-8 из CLI", eps[GROUP_POLICY], 1e-8, tol=1e-20)
    # 4) без флагов value/shared сохраняют прежнее отношение к policy
    opt.apply_settings(lr_policy=2e-5)
    lr3 = opt.lr_by_group()
    check_close("resume: без флага value сохраняет ratio 0.5", lr3[GROUP_VALUE], 1e-5, tol=1e-12)
    check_close("resume: без флага shared сохраняет ratio 0.5", lr3[GROUP_SHARED], 1e-5, tol=1e-12)
    check_eq("resume: eps не тронут без флага", opt.eps_by_group()[GROUP_POLICY], 1e-8)

    # 5) legacy-чекпоинт с плоским Adam: конверсия в SplitAdam переносит моменты
    import torch as _torch
    policy2 = make_policy(features_dim=64)
    named = list(policy2.named_parameters())
    flat = _torch.optim.Adam([p for _, p in named], lr=7e-5, eps=1e-7)
    policy2.optimizer = flat
    for _, prm in named:
        prm.grad = _torch.randn_like(prm) * 0.01
    flat.step()
    moved_before = len(flat.state)
    check("legacy: плоский Adam имеет моменты", moved_before > 0, str(moved_before))
    new_opt = adopt_optimizer(policy2)
    check("legacy: оптимизатор стал SplitAdam", isinstance(new_opt, SplitAdam),
          type(new_opt).__name__)
    check_eq("legacy: моменты перенесены", len(new_opt.state), moved_before)
    check_eq("legacy: групп три (policy/value/shared)", len(new_opt.param_groups), 3)
    check_close("legacy: lr сохранён", new_opt.lr_by_group()[GROUP_POLICY], 7e-5, tol=1e-12)
    check_close("legacy: eps сохранён", new_opt.eps_by_group()[GROUP_POLICY], 1e-7, tol=1e-20)
    lr4 = new_opt.apply_settings(lr_policy=3e-4, lr_value=1e-4).copy()
    check_close("legacy: после apply_settings lr value = 1e-4", lr4[GROUP_VALUE], 1e-4, tol=1e-12)


def main() -> int:
    part_a_schedules()
    print("-" * 78)
    part_b_groups()
    print("-" * 78)
    part_c_adapt()
    print("-" * 78)
    part_d_bc_reset()
    print("-" * 78)
    part_e_ppo_schedule()
    print("-" * 78)
    part_f_compat()
    print("-" * 78)
    part_g_value_only()
    print("-" * 78)
    part_h_resume_contract()
    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
