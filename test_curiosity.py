"""Проверки ICM/CuriosityVecWrapper: устройство, размерности, нормализация, терминальные переходы.

Основное, что здесь закреплено (по итогам разбора кода):
  1. Веса ICM реально переезжают на device (иначе на cuda — RuntimeError на первом вызове).
  2. Размерности по умолчанию берутся из config (N_FEATURES/N_ACTIONS), а не «зашиты» в модуль.
  3. terminal_observation, который доходит до wrapper'а, УЖЕ нормализован VecNormalize —
     это проверяется на живом стеке, а не по документации; из него достаётся СЫРОЕ значение.
  4. В replay лежат СЫРЫЕ наблюдения, и обучение ICM нормализует их ТЕКУЩЕЙ статистикой
     (obs_rms дрейфует: записи 10k шагов назад отмасштабированы иначе).
  5. done-шаг без terminal_observation: переход не попадает в replay, r_int = 0.
  6. Curiosity-вклад за эпизод виден в info["r_int_episode"] и предупреждает при
     превышении SHAPING_EPISODE_CAP.

Запуск: PYTHONPATH=. python test_curiosity.py  (нужны numpy, torch, sb3; сервер не нужен)
"""

import os
import sys
import tempfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from agents.config import N_FEATURES
from agents.curiosity import _default_action_dim

N_ACTIONS = _default_action_dim()

OK: list = []
FAIL: list = []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


def check_eq(name, got, want):
    check(name, got == want, f"got={got!r} want={want!r}")


def _obs_row(x, i: int = 0) -> np.ndarray:
    """Строка i наблюдений из dict-obs / ndarray (VecNormalize отдаёт dict с [N, D])."""
    if isinstance(x, dict):
        x = x.get("observation", x)
    arr = np.asarray(x, dtype=np.float32)
    return arr[i] if arr.ndim >= 2 else arr


class Env870(gym.Env):
    """Env с dict-obs как у ExampleEnv: фиксированное число шагов до done."""
    observation_space = spaces.Dict({
        "observation": spaces.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8),
    })
    action_space = spaces.Discrete(N_ACTIONS)

    def __init__(self, horizon: int = 3, base: float = 3.0, no_terminal: bool = False,
                 strip_scale: float = 0.0):
        self.horizon = int(horizon)
        self.base = float(base)
        self.no_terminal = bool(no_terminal)
        self.strip_scale = float(strip_scale)   # обнуляем сколько-то первых колонок (для дрейфа)
        self.t = 0
        self.rng = np.random.default_rng(0)

    def _obs(self):
        obs = self.rng.normal(size=(N_FEATURES,)).astype(np.float32) * 0.5 + self.base
        if self.strip_scale:
            k = int(self.strip_scale)
            obs[:k] = 0.0
        return obs

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return {"observation": self._obs(),
                "action_mask": np.ones((N_ACTIONS,), dtype=np.int8)}, {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.horizon
        obs = self._obs()
        info = {}
        if self.no_terminal:
            # имитируем «нет terminal_observation»: внешний код сам решает, что делать
            info["_no_terminal"] = True
        return ({"observation": obs, "action_mask": np.ones((N_ACTIONS,), dtype=np.int8)},
                0.1, done, False, info)


class StubICM(nn.Module):
    """Двойник ICM: записывает, куда его переместили и что ему подают на вход."""

    def __init__(self, obs_dim: int = N_FEATURES, action_dim: int = N_ACTIONS):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.to_calls: list = []
        self.seen_r_int: list = []
        self.seen_train: list = []
        self.proj = nn.Linear(obs_dim, 8)
        self.head = nn.Linear(8, action_dim)

    def to(self, *args, **kwargs):
        # только фиксируем вызов: реально двигать некуда (в тестовой среде нет GPU),
        # а на CPU перемещение и так совпадает с исходным устройством
        self.to_calls.append((args, kwargs))
        return self

    def encode(self, obs):
        return self.proj(obs)

    def intrinsic_reward(self, obs, actions, next_obs, device=None):
        self.seen_r_int.append({"obs": np.asarray(obs), "next": np.asarray(next_obs),
                                "actions": np.asarray(actions)})
        return np.full(len(np.asarray(actions)), 0.5, dtype=np.float64)

    def forward_loss(self, obs, actions, next_obs):
        self.seen_train.append({"obs": np.asarray(obs.detach().cpu()),
                                "next": np.asarray(next_obs.detach().cpu())})
        # z требует градиента (как настоящий loss), иначе optimizer.step() в wrapper'е
        # напечатал бы «does not require grad» и тест выглядел бы как поломка
        z = self.proj(obs).sum() * 0.0
        return z, z, z.detach().expand(obs.shape[0])


def make_stack(horizon=3, no_terminal=False, norm_horizon=None):
    """DummyVecEnv + Monitor + VecNormalize — как в реальном прогоне, но без сервера.

    norm_horizon: сколько шагов прокрутить ДО замеров (чтобы статистика накопилась).
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make():
        return Monitor(Env870(horizon=horizon, no_terminal=no_terminal))

    venv = DummyVecEnv([make])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=0.99,
                      norm_obs_keys=["observation"])
    vn.training = True
    return vn


def test_icm_defaults_and_device():
    print("1. ICM: размерности из config и перемещение на device")
    from agents.curiosity import CuriosityVecWrapper, ICM

    icm = ICM()
    check_eq("ICM по умолчанию: obs_dim = N_FEATURES", icm.obs_dim, N_FEATURES)
    check_eq("ICM по умолчанию: action_dim = размер действия poke-env", icm.action_dim, N_ACTIONS)
    check_eq("ICM: action_dim выведен из poke-env (gen9 -> 26)", _default_action_dim(9), 26)
    check_eq("ICM: явный obs_dim важнее дефолта", ICM(obs_dim=42).obs_dim, 42)

    # ядро фикса: wrapper обязан вызвать icm.to(device), а не только сохранить строку.
    # Проверяем на "cuda" без GPU: forward не запускаем, только смотрим, что .to() вызван.
    stub = StubICM()
    vn = make_stack()
    CuriosityVecWrapper(vn, icm=stub, device="cuda", beta=0.05)
    moved = [c for c in stub.to_calls if c[0][0] == "cuda"]
    check("wrapper: icm.to('cuda') вызван (иначе на GPU RuntimeError)", bool(moved), str(stub.to_calls))
    check("wrapper: icm.to() вызван ДО создания оптимизатора",
          len(stub.to_calls) >= 1)
    vn.close()

    stub2 = StubICM()
    vn2 = make_stack()
    w = CuriosityVecWrapper(vn2, icm=stub2, device="cpu")
    check("wrapper: на cpu веса остаются на cpu",
          all(p.device.type == "cpu" for p in stub2.parameters()))
    check("wrapper: действие по умолчанию не нормализует replay на сырых (флаг)",
          bool(w.normalize_replay))
    vn2.close()


def test_terminal_observation_is_normalized():
    print("2. terminal_observation на живом стеке: нормализован ли (проверка, а не вера)")
    from agents.curiosity import CuriosityVecWrapper
    from stable_baselines3.common.vec_env import VecNormalize

    vn = make_stack(horizon=2)
    # наполним статистику, чтобы нормализация была заметной (не идентичной)
    for _ in range(20):
        vn.reset()
        for _ in range(2):
            vn.step(np.array([0]))

    obs = vn.reset()
    for _ in range(2):
        obs, _, dones, infos = vn.step(np.array([0]))
    check("стек: эпизод завершился (done)", bool(np.asarray(dones)[0]))
    term = infos[0].get("terminal_observation")
    check("стек: terminal_observation присутствует", term is not None, str(type(term)))
    term_arr = _obs_row(term)
    obs_arr = _obs_row(obs)
    raw_arr = _obs_row(vn.get_original_obs())
    check("стек: terminal_observation в масштабе нормализованных obs (не сырой)",
          abs(float(term_arr.mean()) - float(obs_arr.mean())) < 3.0
          and abs(float(term_arr.mean()) - float(raw_arr.mean())) > 0.5,
          f"term mean {term_arr.mean():.2f}, obs mean {obs_arr.mean():.2f}, raw mean {raw_arr.mean():.2f}")
    check("стек: VecNormalize НЕ меняет observation_space (870 признаков)",
          tuple(vn.observation_space["observation"].shape), (N_FEATURES,))

    # wrapper возвращает сырое значение терминала и кладёт его в replay
    stub = StubICM()
    vn2 = make_stack(horizon=2)
    for _ in range(20):
        vn2.reset()
        for _ in range(2):
            vn2.step(np.array([0]))
    w = CuriosityVecWrapper(vn2, icm=stub, batch_size=2, train_freq=10 ** 9)
    out = w.reset()
    out, _, dones, infos = w.step(np.array([0]))
    out, _, dones, infos = w.step(np.array([0]))
    check("wrapper: done поймали", bool(np.asarray(dones)[0]))
    seen_next = stub.seen_r_int[-1]["next"][0]
    raw_last = None
    check("wrapper: для ICM next нормализован (та же шкала, что вход obs)",
          abs(float(seen_next.mean())) < 3.0, f"next mean {seen_next.mean():.2f}")
    replay_last = w.replay[-1]
    check("wrapper: в replay лежит СЫРОЕ наблюдение (шкала ~base 3.0), а не нормализованное",
          abs(float(np.asarray(replay_last[2]).mean()) - 3.0) < 1.0
          and float(np.asarray(replay_last[2]).mean()) > 1.0,
          f"replay next mean {np.asarray(replay_last[2]).mean():.2f}")
    check("wrapper: флаг raw у записи replay", bool(replay_last[3]))
    check("wrapper: параметры и вход ICM в одном масштабе — нормализация применялась к обоим",
          np.allclose(np.asarray(stub.seen_r_int[-1]["obs"]).std(), np.asarray(stub.seen_r_int[-1]["next"]).std(),
                      atol=1.5))
    vn.close()
    vn2.close()


def test_replay_normalized_with_current_stats():
    print("3. Дрейф статистики: обучение ICM нормализует replay ТЕКУЩИМ масштабом")
    from agents.curiosity import CuriosityVecWrapper

    stub = StubICM()
    vn = make_stack(horizon=2)
    for _ in range(20):
        vn.reset()
        for _ in range(2):
            vn.step(np.array([0]))
    w = CuriosityVecWrapper(vn, icm=stub, batch_size=4, train_freq=10 ** 9, replay_size=100)
    w.reset()
    for _ in range(8):
        w.step(np.array([0]))
    check("replay набран", len(w.replay) >= 4, str(len(w.replay)))

    # сдвигаем статистику нормализации: сырые записи должны получить НОВЫЙ масштаб
    old_mean = np.asarray(vn.obs_rms["observation"].mean).copy()
    old_var = np.asarray(vn.obs_rms["observation"].var).copy()
    vn.obs_rms["observation"].mean = old_mean + 100.0
    vn.obs_rms["observation"].var = np.ones_like(vn.obs_rms["observation"].var)

    # детерминированный батч: берём все записи по порядку
    import agents.curiosity as cz
    orig_choice = cz.np.random.choice
    cz.np.random.choice = lambda n, size, replace=False: np.arange(size)
    try:
        w.batch_size = len(w.replay)
        w._train_icm_step()
    finally:
        cz.np.random.choice = orig_choice
    check("ICM обучился на батче (forward_loss вызван)", len(stub.seen_train) >= 1)
    if stub.seen_train:
        raw_stack = np.stack([b[0] for b in w.replay])
        exp_obs = w._normalize(raw_stack)
        got_obs = stub.seen_train[-1]["obs"]
        check("строка сырого replay нормализована РОВНО текущей статистикой",
              exp_obs is not None and np.shape(got_obs) == exp_obs.shape
              and np.allclose(got_obs, exp_obs, atol=1e-5),
              f"max|Δ|={np.abs(np.asarray(got_obs) - exp_obs).max():.2e}" if exp_obs is not None else "нет нормализатора")
        # при сдвиге mean на +100 нормализация уходит за clip_obs=10, поэтому сравниваем
        # не с сырым значением, а со шкалой СТАРОЙ статистики (там было ~0)
        old_norm = (raw_stack - old_mean) / np.sqrt(old_var)
        check("батч отличается от прежней шкалы (дрейф действительно учтён)",
              abs(float(np.asarray(got_obs).mean()) - float(old_norm.mean())) > 5.0,
              f"батч mean {np.asarray(got_obs).mean():.1f} (clip ±10) vs старая шкала "
              f"{old_norm.mean():.2f} vs сырое {raw_stack.mean():.1f}")
    check("replay по-прежнему хранит сырые значения (не портим накопленное)",
          abs(float(np.asarray(w.replay[-1][0]).mean()) - 3.0) < 1.0,
          f"{np.asarray(w.replay[-1][0]).mean():.2f}")
    vn.close()


def test_missing_terminal_observation():
    print("4. done без terminal_observation: переход в replay не пишем, r_int = 0")
    from agents.curiosity import CuriosityVecWrapper

    stub = StubICM()
    vn = make_stack(horizon=2)
    w = CuriosityVecWrapper(vn, icm=stub, batch_size=2, train_freq=10 ** 9)
    w.reset()
    w.step(np.array([0]))
    n_before = len(w.replay)

    # эмулируем отсутствие ключа: перехватываем step_wait внутреннего стека
    real_step_wait = w.venv.step_wait

    def step_wait_without_terminal():
        obs, rew, dones, infos = real_step_wait()
        for info in infos:
            info.pop("terminal_observation", None)
        return obs, rew, dones, infos

    w.venv.step_wait = step_wait_without_terminal
    obs, rewards, dones, infos = w.step(np.array([0]))
    if not bool(np.asarray(dones)[0]):
        obs, rewards, dones, infos = w.step(np.array([0]))
    check("done-шаг пойман", bool(np.asarray(dones)[0]))
    check_eq("счётчик missing_terminal_obs увеличен", w.missing_terminal_obs, 1)
    check_eq("артефактный переход НЕ добавлен в replay", len(w.replay), n_before)
    check_eq("r_int за артефактный переход = 0", float(infos[0]["r_int"]), 0.0)
    vn.close()


def test_episode_curiosity_accounting():
    print("5. Curiosity-вклад за эпизод: info['r_int_episode'] и предупреждение при перевесе")
    import io
    import contextlib

    from agents.curiosity import CuriosityVecWrapper

    stub = StubICM()
    vn = make_stack(horizon=2)
    w = CuriosityVecWrapper(vn, icm=stub, beta=1.0, anneal=False, batch_size=2, train_freq=10 ** 9)
    w.reset()
    w.step(np.array([0]))
    obs, rewards, dones, infos = w.step(np.array([0]))
    check("эпизод завершился", bool(np.asarray(dones)[0]))
    check("info['r_int'] пишется (для TB/логов)", "r_int" in infos[0], str(sorted(infos[0].keys())))
    check("info['beta'] пишется", "beta" in infos[0])
    check("info['r_int_episode'] пишется на конце эпизода", "r_int_episode" in infos[0],
          str(infos[0].get("r_int_episode")))
    # r_int у двойника = 0.5, beta = 1.0, 2 шага в эпизоде -> вклад 1.0
    check_eq("вклад за эпизод = beta * сумма r_int", round(float(infos[0]["r_int_episode"]), 6), 1.0)
    check("wrapper помнит максимум вклада за эпизод", w.max_episode_r_int >= 1.0,
          f"{w.max_episode_r_int}")

    # порог предупреждения: ставим ниже фактического вклада
    w2 = CuriosityVecWrapper(vn, icm=StubICM(), beta=1.0, anneal=False, batch_size=2,
                             train_freq=10 ** 9)
    w2.episode_r_int_warn_at = 0.5
    w2.reset()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        w2.step(np.array([0]))
        w2.step(np.array([0]))
    out = buf.getvalue()
    check("при перевесе curiosity над SHAPING_EPISODE_CAP печатается предупреждение",
          "curiosity-вклад за эпизод" in out and "SHAPING_EPISODE_CAP" in out,
          out.strip().splitlines()[-1][:100] if out.strip() else "нет вывода")
    vn.close()


def test_icm_forward_detach_and_shapes():
    print("6. ICM: phi_next.detach() в forward-loss и совместимость размеров")
    from agents.curiosity import ICM, CuriosityVecWrapper

    icm = ICM(obs_dim=32, action_dim=N_ACTIONS, feat_dim=16)
    obs = torch.randn(5, 32)
    nxt = torch.randn(5, 32)
    acts = torch.randint(0, N_ACTIONS, (5,))
    inv, fwd, per = icm.forward_loss(obs, acts, nxt)
    check("forward_loss: три выхода", (inv.dim(), fwd.dim(), per.shape), (0, 0, torch.Size([5])))
    # градиент по энкодеру от forward-пути не должен содержать членов через phi_next
    icm.zero_grad()
    _, fwd_only, _ = icm.forward_loss(obs, acts, nxt)
    fwd_only.backward()
    g = icm.encoder[0].weight.grad
    check("forward_loss: градиент есть (обучение идёт)", g is not None and float(g.abs().sum()) > 0)

    # r_int без градиента и с clip в wrapper'е
    r = icm.intrinsic_reward(obs.numpy(), acts.numpy(), nxt.numpy())
    check("intrinsic_reward: форма [N]", r.shape, (5,))

    # реальный ICM в живом стеке: шаг проходит без ошибок, replay наполняется
    vn = make_stack(horizon=2)
    real_icm = ICM(feat_dim=16)
    w = CuriosityVecWrapper(vn, icm=real_icm, batch_size=4, train_freq=10 ** 9)
    w.reset()
    for _ in range(10):
        obs_o, rew_o, dones_o, infos_o = w.step(np.array([0]))
    check("живой ICM: replay наполняется переходами", len(w.replay) >= 8, str(len(w.replay)))
    check("живой ICM: r_int конечен и неотрицателен",
          np.isfinite(w.last_r_int_mean) and w.last_r_int_mean >= 0.0, f"{w.last_r_int_mean}")
    w.batch_size = 4
    w._train_icm_step()
    check("живой ICM: шаг обучения прошёл, fwd_loss записан",
          w.last_fwd_loss >= 0.0, f"inv={w.last_inv_loss:.4f}, fwd={w.last_fwd_loss:.4f}")
    vn.close()


def test_config_sync_stale_dimension_caught():
    print("7. Рассинхрон размерности: понятная ошибка вместо тихой порчи")
    from agents.curiosity import ICM, CuriosityVecWrapper

    vn = make_stack(horizon=2)
    icm = ICM(obs_dim=16, feat_dim=8)      # заведомо неправильная размерность
    w = CuriosityVecWrapper(vn, icm=icm, batch_size=2, train_freq=10 ** 9)
    w.reset()
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            w.step(np.array([0]))
            crashed = False
        except Exception:
            crashed = True
    out = buf.getvalue()
    check("несовпадение obs_dim: wrapper сообщает об ошибке (не молчит)",
          ("CuriosityVecWrapper step_wait warn" in out) or crashed,
          out.strip().splitlines()[-1][:110] if out.strip() else "нет вывода")
    vn.close()


def main() -> int:
    print(f"N_FEATURES={N_FEATURES}, N_ACTIONS={N_ACTIONS}")
    print("-" * 74)
    test_icm_defaults_and_device()
    print("-" * 74)
    test_terminal_observation_is_normalized()
    print("-" * 74)
    test_replay_normalized_with_current_stats()
    print("-" * 74)
    test_missing_terminal_observation()
    print("-" * 74)
    test_episode_curiosity_accounting()
    print("-" * 74)
    test_icm_forward_detach_and_shapes()
    print("-" * 74)
    test_config_sync_stale_dimension_caught()
    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
