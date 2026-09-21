#!/usr/bin/env python3
"""Второй режим действий (`--action-mode embed`): выбор по кандидатам, а не по индексам.

Что проверяем:

  A) раскладка действий embed: свитч-действие j = j-й резерв КАНОНИЧЕСКОГО порядка
     (того же, в котором резервы лежат в bench-блоке признаков). Проверяем на живых объектах
     poke-env: маска разрешает ровно те j, для которых резерв существует и легален, а
     action_to_order -> order_to_action возвращает тот же j;
  B) инвариантность к порядку команды: перестановка команды в battle.team (тот самый
     произвол, из-за которого индексы свитчей ничего не значат) НЕ меняет ни маску embed,
     ни ордер для одного и того же монстра;
  C) срезы кандидатов: индексные карты совпадают с признаками конкретных кандидатов
     (base_power поля i == move_candidate_matrix[i][0]), срезы не пересекаются и вместе с
     контекстом покрывают obs ровно один раз;
  D) политика embed: логиты зависят ТОЛЬКО от (контекст, признаки кандидата) — перестановка
     признаков двух кандидатов переставляет их логиты, остальные не меняются. В режиме indices
     такого свойства нет (проверяем, что тест не проходит на старой политике);
  E) обучение: BC на мини-датасете и полный цикл PPO (collect_rollouts + train) работают,
     log_prob конечны, маска соблюдается (выбранное действие всегда разрешено), а softmax,
     entropy и log_prob считаются СТРОГО по разрешённым действиям (иначе importance ratio в
     PPO тихо станет неверным);
  F) режим — свойство модели: чекпоинт embed сохраняет action_mode, `action_mode_from_checkpoint`
     его читает, BC отказывается учиться на датасете чужого режима (и принимает свой), а resume
     не тащит embed-чекпоинт в миграцию «старой архитектуры» из-за крошечных легаси-веток.

Запуск: PYTHONPATH=. python test_slot_free_mode.py
"""
import logging
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from agents import action_space as A  # noqa: E402
from agents import config as _cfg  # noqa: E402,F401  (монки-патчи poke-env)
from agents import features as F  # noqa: E402
from agents.config import N_FEATURES  # noqa: E402

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


def make_battle(team_order=("swampert", "garchomp", "heatran", "corviknight"), moves=None):
    """Живой Battle с командой в заданном порядке (имена = позиции в battle.team)."""
    from poke_env.battle import Battle

    moves = moves or ("hydropump", "earthquake", "icebeam", "protect")
    b = Battle(battle_tag="battle-gen9randombattle-1", username="Me",
               logger=logging.getLogger("quiet"), gen=9)
    for msg in [["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""],
                ["", "start"],
                ["", "switch", "p1a: First", f"{team_order[0]}, L50, M", "300/300"],
                ["", "switch", "p2a: Skarm", "skarmory, L50, F", "300/300"]]:
        b.parse_message(msg)
    details = {"garchomp": "garchomp, L50, F", "heatran": "heatran, L50, M",
               "corviknight": "corviknight, L50, F", "blissey": "blissey, L50, F"}
    pokemon = [{"ident": "p1: First", "details": f"{team_order[0]}, L50, M", "condition": "300/300",
                "active": True,
                "stats": {"atk": 250, "def": 250, "spa": 250, "spd": 250, "spe": 250},
                "moves": list(moves), "baseAbility": "torrent", "item": "leftovers",
                "pokeball": "pokeball", "ability": "torrent"}]
    for i, name in enumerate(team_order[1:]):
        pokemon.append({"ident": f"p1: Slot{i}", "details": details.get(name, f"{name}, L50, M"),
                        "condition": "300/300", "active": False,
                        "stats": {"atk": 250, "def": 250, "spa": 250, "spd": 250, "spe": 250},
                        "moves": ["tackle"], "baseAbility": "roughskin", "item": "leftovers",
                        "pokeball": "pokeball", "ability": "roughskin"})
    b.parse_request({"active": [{"moves": [{"move": m.capitalize(), "id": m, "pp": 16, "maxpp": 16,
                                           "target": "normal", "disabled": False} for m in moves]}],
                     "side": {"id": "p1", "name": "Me", "pokemon": pokemon}, "rqid": 1})
    return b


class _ProbeEnv(gym.Env):
    """Мини-env с тем же Dict-obs, что у боевого (для VecEnv в тестах обучения)."""

    def __init__(self, dim=N_FEATURES):
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 2.0, shape=(dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(26,), dtype=np.int8)})
        self.action_space = spaces.Discrete(26)
        self._rng = np.random.default_rng(0)
        self._step = 0

    def _obs(self):
        obs = self._rng.normal(0, 0.3, size=N_FEATURES).astype(np.float32)
        mask = np.zeros(26, dtype=np.int8)
        mask[6:10] = 1                       # 4 приёма
        mask[0] = 1                          # один свитч
        mask[22:26] = 1                      # тера
        return {"observation": obs, "action_mask": mask}

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        self._step += 1
        done = self._step >= 8
        return self._obs(), 0.1, done, done, {}


def make_policy(mode="embed", **kw):
    from agents.policy import EmbeddedActorCriticPolicy, MaskedActorCriticPolicy
    cls = EmbeddedActorCriticPolicy if mode == "embed" else MaskedActorCriticPolicy
    obs_space = spaces.Dict({
        "observation": spaces.Box(-1.0, 2.0, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, shape=(26,), dtype=np.int8)})
    return cls(obs_space, spaces.Discrete(26), lr_schedule=lambda p: 2e-4,
               features_extractor_kwargs=dict(features_dim=64),
               net_arch=dict(pi=[64], vf=[64]), **kw)


def obs_of(battle, mask=None):
    obs = F.embed_battle_with_fusion(battle, None, None, our_team_fusions={}, opp_team_fusions={})
    if mask is None:
        mask = np.array(A.get_action_mask(battle), dtype=np.int8)
    return {"observation": torch.as_tensor(obs).unsqueeze(0),
            "action_mask": torch.as_tensor(mask).unsqueeze(0)}


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from poke_env.environment.singles_env import SinglesEnv

    # --------------------------------------------------- A) раскладка и round-trip ---
    print("=" * 78)
    print("A. embed: свитч-действие j = j-й резерв канонического порядка (и round-trip)")
    print("=" * 78)
    b = make_battle(team_order=("swampert", "garchomp", "heatran", "corviknight"))
    team_ids = [m.species for m in b.team.values()]
    reserves = [m.species for m in A.embed_reserves(b)]
    check("A: канонический порядок резервов отсортирован по виду",
          reserves == sorted(reserves), f"team={team_ids} резервы={reserves}")
    A.set_action_mode("embed")
    mask = A.get_action_mask(b)
    check("A: маска embed разрешает свитчи ровно по числу резервов",
          [i for i in range(6) if mask[i]] == list(range(len(reserves))), str(mask[:6]))
    check("A: индекс 5 (6-й монстр) в сингле всегда запрещён", mask[5] == 0)
    ok_round = True
    for j in range(len(reserves)):
        order = A.action_to_order(np.int64(j), b)
        back = int(A.order_to_action(order, b, fake=True, strict=False))
        # сверяем ИМЕННО монстра: poke-env в ордере печатает ник/позицию, а не вид
        if back != j or getattr(order, "order", None) is not A.embed_reserves(b)[j]:
            ok_round = False
            print(f"   j={j}: {order} -> {back}")
    check("A: action_to_order -> order_to_action возвращает тот же индекс и монстра", ok_round)
    check("A: приёмы нумеруются как раньше (6..9 по слотам known_moves)",
          [i for i in range(6, 10) if mask[i]] == [6, 7, 8, 9], str(mask[6:10]))
    A.set_action_mode("indices")
    mask_ind = SinglesEnv.get_action_mask(b)
    check("A: indices-маска ссылается на team-индексы (сдвиг относительно embed)",
          [i for i in range(6) if mask_ind[i]] != [i for i in range(6) if mask[i]],
          f"indices={[i for i in range(6) if mask_ind[i]]} embed={[i for i in range(6) if mask[i]]}")

    # --------------------------------------------- B) инвариантность к порядку team ---
    print("-" * 78)
    print("B. Порядок команды не влияет на embed (в indices — влияет)")
    print("-" * 78)
    A.set_action_mode("embed")
    orders = (("swampert", "garchomp", "heatran", "corviknight"),
              ("swampert", "corviknight", "garchomp", "heatran"),
              ("swampert", "heatran", "corviknight", "garchomp"))
    masks, targets = [], []
    for team_order in orders:
        bt = make_battle(team_order=team_order)
        masks.append([i for i in range(6) if A.get_action_mask(bt)[i]])
        targets.append([m.species for m in A.embed_reserves(bt)])
    check("B: маска embed одинакова при любом порядке команды", masks[0] == masks[1] == masks[2],
          str(masks))
    check("B: действие j всегда ведёт на один и тот же вид (канонический порядок)",
          targets[0] == targets[1] == targets[2], str(targets[0]))
    A.set_action_mode("indices")
    ind_targets = []
    for team_order in orders:
        bt = make_battle(team_order=team_order)
        row = []
        for i in range(6):
            if SinglesEnv.get_action_mask(bt)[i]:
                order = SinglesEnv.action_to_order(np.int64(i), bt)
                row.append((i, getattr(getattr(order, "order", None), "species", "?")))
        ind_targets.append(row)
    check("B: в indices тот же индекс ведёт на РАЗНЫЕ виды (это и есть проблема)",
          ind_targets[0] != ind_targets[1], f"{ind_targets[0]} vs {ind_targets[1]}")

    # --------------------------------------------------- C) срезы кандидатов ---
    print("-" * 78)
    print("C. Срезы кандидатов: карты индексов == признаки конкретных кандидатов")
    print("-" * 78)
    A.set_action_mode("embed")
    bt = make_battle(team_order=("swampert", "garchomp", "heatran", "corviknight"),
                     moves=("hydropump", "earthquake", "icebeam", "protect"))
    obs = F.embed_battle_with_fusion(bt, None, None, our_team_fusions={}, opp_team_fusions={})
    mv = F.move_candidate_matrix(obs, with_flags=False)
    check("C: срезы не пересекаются и покрывают obs вместе с контекстом", F.candidate_slices_ok(obs))
    # контекст + то, что модель берёт как кандидатов (приёмы и bench-блок) + колонки угрозы
    threats = F.SWITCH_THREAT_DIM * 5
    check("C: контекст + кандидаты покрывают все N_FEATURES",
          F.context_dim() + 4 * F.MOVE_FEATURES_DIM + 5 * F.BENCH_SLOT_DIM + threats == N_FEATURES,
          f"{F.context_dim()} + {4 * F.MOVE_FEATURES_DIM} + {5 * F.BENCH_SLOT_DIM} + {threats}")
    # base_power приёма i (bp/100) лежит в начале вектора кандидата i
    expect_bp = [m.base_power / 100 for m in F.move_slots_for_action(bt)]
    check("C: base_power кандидата i == его же признак",
          np.allclose(mv[:, 0], expect_bp), f"{np.round(mv[:, 0], 3)} vs {expect_bp}")
    # перестановка слотов приёмов переставляет кандидатов (а не колонки внутри кандидата)
    obs_swapped = obs.copy()
    idx = F.move_feature_index()
    obs_swapped[idx[0]] = obs[idx[1]]
    obs_swapped[idx[1]] = obs[idx[0]]
    mv2 = F.move_candidate_matrix(obs_swapped, with_flags=False)
    check("C: перестановка слотов приёмов переставляет кандидатов",
          np.allclose(mv2[0], mv[1]) and np.allclose(mv2[1], mv[0]))
    # свитч-кандидаты: bench-слот j идёт первым блоком вектора
    sw = F.switch_candidate_matrix(obs)
    b_start = F.obs_layout()["our_bench"][0]
    check("C: первый блок вектора свитча == его bench-слот",
          np.allclose(sw[0, :F.BENCH_SLOT_DIM], obs[b_start:b_start + F.BENCH_SLOT_DIM]))
    check("C: 5 свитч-кандидатов в том же порядке, что действия embed",
          len(A.embed_reserves(bt)) <= 5 and sw.shape[0] == 5,
          f"резервов {len(A.embed_reserves(bt))}")

    # --------------------------------------------------- D) инвариантность логитов ---
    print("-" * 78)
    print("D. Логиты зависят от кандидата, а не от его позиции")
    print("=" * 78)
    torch.manual_seed(0)
    pol = make_policy("embed")
    # веса «разгоняем»: у свежей политики ortho(gain=0.01) даёт логиты ~1e-6, и любая проверка
    # на равенство проходит тривиально. После normal_ логиты имеют нормальный масштаб.
    with torch.no_grad():
        for module in pol.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(0.0, 0.5)
                module.bias.normal_(0.0, 0.1)
    pol.eval()
    with torch.no_grad():
        logits_before, _ = pol.logits_and_values(obs_of(bt))
    base = logits_before[0]
    usable = [0, 1, 2, 3, 4, 6, 7, 8, 9, 22, 23, 24, 25]
    check("D: логиты кандидатов нетривиальны и конечны (тест имеет смысл)",
          float(base[usable].abs().max()) > 0.05 and bool(torch.isfinite(base[usable]).all()),
          f"max|logit|={float(base[usable].abs().max()):.2f}")
    # переставляем признаки 1-го и 2-го приёма (их 33 признака: 30 из головы + урон)
    obs_perm = obs.copy()
    cols = F.move_feature_index()
    obs_perm[cols[1]] = obs[cols[2]]
    obs_perm[cols[2]] = obs[cols[1]]
    obs_dict = {"observation": torch.as_tensor(obs_perm).unsqueeze(0),
                "action_mask": torch.ones(1, 26)}
    with torch.no_grad():
        logits_perm, _ = pol.logits_and_values(obs_dict)
    b_perm = logits_perm[0]
    check("D: логит приёма 7 == логит приёма 8 после перестановки их признаков",
          abs(float(base[7]) - float(b_perm[8])) < 1e-5 and abs(float(base[8]) - float(b_perm[7])) < 1e-5,
          f"{float(base[7]):.4f}/{float(base[8]):.4f} -> {float(b_perm[7]):.4f}/{float(b_perm[8]):.4f}")
    check("D: тера-логиты переставляются вместе с приёмами (23 <-> 24)",
          abs(float(base[23]) - float(b_perm[24])) < 1e-5 and abs(float(base[24]) - float(b_perm[23])) < 1e-5,
          f"{float(base[23]):.4f}/{float(base[24]):.4f} -> {float(b_perm[23]):.4f}/{float(b_perm[24]):.4f}")
    check("D: свитчи и не тронутые приёмы (0..4, 6, 9) не сдвинулись",
          all(abs(float(base[i]) - float(b_perm[i])) < 1e-5 for i in (0, 1, 2, 3, 4, 6, 9)))
    # для сравнения: у позиционной политики логиты привязаны к позиции
    pol_idx = make_policy("indices")
    with torch.no_grad():
        for module in pol_idx.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(0.0, 0.5)
                module.bias.normal_(0.0, 0.1)
    pol_idx.eval()
    with torch.no_grad():
        idx_base = pol_idx.logits_and_values(obs_of(bt))[0][0]
        idx_perm = pol_idx.logits_and_values(obs_dict)[0][0]
    # главное отличие: у embed логит ПЕРЕЕХАЛ вместе с признаками кандидата, у позиционной —
    # логит привязан к номеру слота и просто изменился (кандидат «внутри» него уже другой)
    moved_embed = abs(float(base[7]) - float(b_perm[8])) < 1e-5
    moved_idx = abs(float(idx_base[7]) - float(idx_perm[8])) < 1e-5
    check("D: embed переносит логит вместе с кандидатом, indices — нет",
          moved_embed and not moved_idx,
          f"embed: logit[7]={float(base[7]):.3f} == logit'[8]={float(b_perm[8]):.3f}; "
          f"indices: logit[7]={float(idx_base[7]):.3f} != logit'[8]={float(idx_perm[8]):.3f}")
    check("D: embed-политика не зависит от неиспользуемых легаси-веток",
          float(pol.features_extractor.net[0].weight.shape[1]) == float(N_FEATURES))

    # ------------------------------------------------------------ E) обучение ---
    print("-" * 78)
    print("E. Обучение: PPO-роллаут и BC на мини-датасете в режиме embed")
    print("-" * 78)
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO

    from agents.policy import EmbeddedActorCriticPolicy
    from agents.training import pretrain_policy_bc

    venv = DummyVecEnv([lambda: _ProbeEnv()])
    ppo = PPO(EmbeddedActorCriticPolicy, venv, device="cpu", n_steps=16, batch_size=8, n_epochs=2,
              verbose=0,
              policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=64),
                                 net_arch=dict(pi=[64], vf=[64])))
    ppo.learn(total_timesteps=32)
    check("E: PPO collect_rollouts + train в embed-режиме отработали", True)
    obs_t = {"observation": torch.randn(4, N_FEATURES), "action_mask": torch.ones(4, 26)}
    with torch.no_grad():
        actions, values, log_probs = ppo.policy.forward(obs_t)
    check("E: forward возвращает конечные log_prob/values",
          bool(torch.isfinite(log_probs).all()) and bool(torch.isfinite(values).all()))
    # маска соблюдается: запрещённые действия не выбираются
    mask = torch.zeros(4, 26)
    mask[:, 6] = 1
    mask[:, 2] = 1
    obs_t = {"observation": torch.randn(4, N_FEATURES), "action_mask": mask}
    with torch.no_grad():
        chosen, _, _ = ppo.policy.forward(obs_t)
    check("E: политика выбирает только разрешённые маской действия",
          set(chosen.tolist()) <= {6, 2}, str(chosen.tolist()))

    # BC на мини-датасете (в памяти)
    n = 64
    ds = []
    rng = np.random.default_rng(0)
    for i in range(n):
        ds.append((rng.normal(0, 0.3, size=N_FEATURES).astype(np.float32),
                   np.array([1 if k in (6, 7, 8, 9) else 0 for k in range(26)], dtype=np.int8),
                   int(rng.integers(6, 10)), float(rng.normal(0, 1))))
    venv2 = DummyVecEnv([lambda: _ProbeEnv()])
    ppo2 = PPO(EmbeddedActorCriticPolicy, venv2, device="cpu", n_steps=16, batch_size=8, n_epochs=1,
               verbose=0,
               policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=64),
                                  net_arch=dict(pi=[64], vf=[64])))
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        pretrain_policy_bc(ppo2, ds, epochs=1, batch_size=16, normalize=False,
                           reset_at_start=True, reset_optimizer=False, progress_every=0,
                           patience=99)
    line = [l for l in buf.getvalue().splitlines() if l.startswith("[BC epoch 0]")]
    check("E: BC в embed-режиме прошёл и посчитал loss", bool(line), line[0][:80] if line else "")

    # --- корректность распределения для PPO: softmax/log_prob/entropy ТОЛЬКО по разрешённым.
    # Если считать их по всем 26 слотам (или по кандидатам без маски), importance ratio тихо
    # станет неверным — это ровно то место, которого опасались при переменной длине кандидатов.
    pol = ppo.policy
    pol.eval()
    mask = np.zeros(26, dtype=np.int8)
    mask[6] = mask[8] = mask[2] = 1          # один свитч + два приёма
    with torch.no_grad():
        dist = pol.get_distribution(obs_of(bt, mask=mask))
        logits, _ = pol.logits_and_values(obs_of(bt, mask=mask))
        probs = dist.distribution.probs[0]
        ent = dist.distribution.entropy()[0]
    allowed = np.array([6, 8, 2])
    p_allowed = probs[allowed].numpy()
    logits_allowed = logits[0][allowed].numpy().astype(np.float64)
    z = np.exp(logits_allowed - logits_allowed.max())
    manual = z / z.sum()
    check("E: softmax идёт только по разрешённым действиям (сумма 1, запрещённые 0)",
          abs(float(probs.sum()) - 1.0) < 1e-5 and float(probs.numpy()[[i for i in range(26) if mask[i] == 0]].sum()) < 1e-9,
          f"sum={float(probs.sum()):.6f}")
    check("E: вероятности совпадают с softmax по разрешённым логитам",
          np.allclose(p_allowed, manual, atol=1e-5),
          f"dist={np.round(p_allowed, 4)} softmax={np.round(manual, 4)}")
    manual_ent = -float((manual * np.log(manual)).sum())
    check("E: entropy считается по разрешённым действиям", abs(float(ent) - manual_ent) < 1e-4,
          f"{float(ent):.5f} vs {manual_ent:.5f}")
    # log_prob из evaluate_actions (PPO) обязан совпасть с log_prob выбранных действий
    acts = torch.tensor([6, 8, 2])
    with torch.no_grad():
        obs_rep = {k: v.repeat(3, 1) for k, v in obs_of(bt, mask=mask).items()}
        _, lp_eval, _ = pol.evaluate_actions(obs_rep, acts)
        lp_direct = dist.distribution.log_prob(acts)
    check("E: evaluate_actions.log_prob == log_prob действий (важно для importance ratio)",
          torch.allclose(lp_eval, lp_direct, atol=1e-5),
          f"{np.round(lp_eval.numpy(), 5)} == {np.round(lp_direct.numpy(), 5)}")

    # --------------------------------------------------- F) режим как свойство модели ---
    print("-" * 78)
    print("F. Режим хранится в чекпоинте, датасет проверяется на режим")
    print("=" * 78)
    td = tempfile.mkdtemp(prefix="slotfree_")
    try:
        path = os.path.join(td, "embed_model.zip")
        ppo2.save(path)
        from agents.checkpoint_utils import action_mode_from_checkpoint
        check("F: чекпоинт embed опознаётся как embed",
              action_mode_from_checkpoint(path) == "embed", str(action_mode_from_checkpoint(path)))
        path_idx = os.path.join(td, "indices_model.zip")
        venv3 = DummyVecEnv([lambda: _ProbeEnv()])
        from agents.policy import MaskedActorCriticPolicy
        ppo3 = PPO(MaskedActorCriticPolicy, venv3, device="cpu", n_steps=16, batch_size=8, verbose=0)
        ppo3.save(path_idx)
        check("F: обычный чекпоинт опознаётся как indices",
              action_mode_from_checkpoint(path_idx) == "indices",
              str(action_mode_from_checkpoint(path_idx)))

        # датасет: сайдкар пишется с режимом, чужой режим отвергается
        from agents.training import read_dataset_meta, save_dataset, validate_bc_dataset
        A.set_action_mode("embed")
        ds_path = os.path.join(td, "ds_embed.npz")
        save_dataset(ds, ds_path)
        check("F: сайдкар датасета записал режим embed",
              (read_dataset_meta(ds_path) or {}).get("action_mode") == "embed",
              str(read_dataset_meta(ds_path)))
        info = validate_bc_dataset(ds_path, N_FEATURES)
        check("F: BC принимает датасет своего режима", info.get("action_mode") == "embed", str(info))
        A.set_action_mode("indices")
        try:
            validate_bc_dataset(ds_path, N_FEATURES)
            check("F: BC должен отказаться от датасета чужого режима", False, "исключения не было")
        except SystemExit as e:
            check("F: BC отказался от датасета чужого режима с объяснением",
                  "embed" in str(e) and "indices" in str(e), str(e).splitlines()[0][:80])
        # датасет без сайдкара = indices (обратная совместимость)
        os.remove(ds_path + ".meta.json")
        check("F: датасет без сайдкара считается indices (старые датасеты)",
              validate_bc_dataset(ds_path, N_FEATURES).get("action_mode") == "indices")

        # resume: размеРЫ легаси-веток embed-политики (features_dim 64) — не признак старой
        # архитектуры; иначе чекпоинт уходил в миграцию, режим не применялся, и env играл
        # в indices против embed-модели (тихое расхождение семантики)
        from agents.policy_player import resume_migration_target
        check("F: embed-чекпоинт не мигрирует из-за крошечных легаси-веток",
              resume_migration_target("embed", 64, N_FEATURES, 512) == (64, False),
              str(resume_migration_target("embed", 64, N_FEATURES, 512)))
        check("F: obs-несовпадение мигрирует и в embed",
              resume_migration_target("embed", 64, 418, 512) == (64, True))
        check("F: indices-чекпоинт мигрирует на features_dim из CLI",
              resume_migration_target("indices", 64, N_FEATURES, 512) == (512, True))
        check("F: indices-чекпоинт с текущей архитектурой не мигрирует",
              resume_migration_target("indices", 512, N_FEATURES, 512) == (512, False))
    finally:
        shutil.rmtree(td, ignore_errors=True)
        A.reset_action_mode()

    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
