#!/usr/bin/env python3
"""Признаки одинаковы на ВСЕХ стадиях: BC-претрейн, RL-обучение, оценка винрейта, инференс.

Проверяем три вещи, из-за которых модель «в бою ведёт себя не так, как в обучении»:
  1. РАЗМЕРНОСТЬ: obs_dim == N_FEATURES на каждом пути, маска 26, ничего не паддится/режется молча.
  2. ПОРЯДОК: раскладка колонок одна и та же — сегменты obs отзываются на те правки боя,
     на которые должны (погода, хазарды, тера, скамейка), а не «где-то рядом».
  3. НОРМАЛИЗАЦИЯ: то, что видит политика, одинаково на всех стадиях:
       обучение (VecNormalize) == оценка (LiveVecNormalizeAdapter) == инференс (VecNormStats с диска).

Плюс сквозной прогон тестовой модели по всем стадиям (S1..S5) с проверкой на РЕАЛЬНЫХ боях и
снимках боя:
  S1 сбор датасета (HeuristicRecorder)      -> obs из датасета == пересчёт из сырого снимка
  S2 BC-претрейн (1 эпоха)                  -> модель собрана под N_FEATURES, stats прогреты
  S3 RL-обучение (короткий роллаут)         -> obs в шаге == повторный пересчёт по снимку боя,
                                               буфер обучения == вход политики на шаге,
                                               вход == normalize(сырой obs шага)
  S4 оценка винрейта (реальный бой)         -> те же равенства на пути PolicyPlayer + адаптер
  S5 инференс (save -> load -> бой)         -> сырой obs тот же, нормализация с диска та же

Запуск:
  PYTHONPATH=. python test_features_consistency.py            # офлайн-части + live, если сервер
  PYTHONPATH=. python test_features_consistency.py --no-live   # только офлайн (без сервера)
  PYTHONPATH=. python test_features_consistency.py --only S3    # адресно (стадии накопительные)
  PYBOT_SHOWDOWN_DIR=/path/to/pokemon-showdown PYTHONPATH=. python test_features_consistency.py
Переменные: PYBOT_TEST_FORMAT (по умолчанию gen9randombattle для стокового сервера),
            PYBOT_TEST_FAST=1 (1 бой и короткий роллаут), PYBOT_SHOWDOWN_DIR (автозапуск сервера).
"""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import socket
import subprocess
import sys
import tempfile
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env.battle import Battle, PokemonType, SideCondition, Weather
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from agents import config as _cfg
from agents.config import N_FEATURES
from poke_env.data import GenData

GENDATA_TYPE_CHART = GenData.from_gen(9).type_chart
from agents.features import _TYPE_INDEX as TYPE_INDEX
from agents.damage import DAMAGE_BLOCK_SIZE, EFFECT_FLAGS, FLAGS_BASE, MIRROR_BASE, TEAM_BASE
from agents.features import (
    TYPE_MATCHUP_BASE, TYPE_MATCHUP_BLOCK_SIZE, _eff_types, _matchup_team_slots,
    _type_matchup_block, _type_multi_hot, embed_battle_with_fusion,
)
from agents.fusion_types import effective_types
from agents.type_utils import damage_multiplier_safe
from agents.players import HeuristicRecorder, PolicyPlayer
from agents.vecnorm_utils import (
    LiveVecNormalizeAdapter,
    features_fingerprint,
    load_vecnorm_stats,
    save_vecnormalize_with_meta,
)

OK: list = []
FAIL: list = []
SKIPPED: list = []
ROWS: list = []          # итоговая таблица: (стадия, dim, нормализация, max|Δ|, хеш)


# --------------------------------------------------------------------------- helpers ---
def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


def check_eq(name, got, want):
    check(name, got == want, f"got={got!r} want={want!r}")


def check_close(name, a, b, atol=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        check(name, False, f"формы не совпали: {a.shape} vs {b.shape}")
        return np.inf
    d = float(np.abs(a - b).max())
    check(name, d <= atol, f"max|Δ|={d:.3e}")
    return d


def check_subset(name, changed, lo, hi):
    """Изменённые индексы должны лежать внутри окна [lo, hi) — это проверка ПОРЯДКА колонок."""
    changed = np.asarray(sorted(changed), dtype=int)
    ok = bool(changed.size) and int(changed.min()) >= lo and int(changed.max()) < hi
    check(name, ok, f"изменено {changed.size} колонок, диапазон [{changed.min() if changed.size else '-'}, "
                    f"{changed.max() if changed.size else '-'}] ожидалось внутри [{lo}, {hi})")
    return changed


def stack_rows(batches: list) -> np.ndarray:
    """Собрать все наблюдения, поданные в политику, в матрицу (N, dim)."""
    rows = [b.reshape(1, -1) if b.ndim == 1 else b for b in batches if np.size(b)]
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, 0), np.float32)


def nearest_row_dist(rows: np.ndarray, ref: np.ndarray) -> float:
    """Для каждой строки rows — минимальное max|Δ| до строк ref (0, если совпала с точностью до float)."""
    if rows.shape[0] == 0 or ref.shape[0] == 0 or rows.shape[1] != ref.shape[1]:
        return float("inf")
    worst = 0.0
    for start in range(0, rows.shape[0], 64):
        chunk = rows[start:start + 64]
        d = np.abs(chunk[:, None, :] - ref[None, :, :]).max(axis=2).min(axis=1)
        worst = max(worst, float(d.max()))
    return worst


def check_windows(name, changed, windows, lo_note=""):
    """Изменённые индексы должны целиком лежать в объединении ожидаемых сегментов."""
    changed = np.asarray(sorted(changed), dtype=int)
    ok = bool(changed.size) and all(any(lo <= i < hi for lo, hi in windows) for i in changed)
    rng = f"[{changed.min()}, {changed.max()}]" if changed.size else "—"
    check(name, ok, f"изменено {changed.size} колонок, диапазон {rng}, ожидалось внутри "
                    f"{[w for w in windows]}{lo_note}")
    return changed


def safe_close(env):
    try:
        env.close()
    except Exception:
        pass


def h8(arr) -> str:
    return hashlib.md5(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).tobytes()).hexdigest()[:8]


TAG = "battle-gen9fusionmonsrandombattle-1"
FUSION = {"base_stats": {"hp": 70, "atk": 80, "def": 90, "spa": 100, "spd": 110, "spe": 120},
          "speed_range": (200, 320)}
PROTECT = (1.0, 0.0)


def make_battle():
    """Синтетический снимок боя: обе стороны, активные, скамейка, погода, хазарды, статус, бусты."""
    b = Battle(battle_tag=TAG, username="Me", logger=logging.getLogger("quiet"), gen=9)
    for msg in (["", "player", "p1", "Me", "", ""],
                ["", "player", "p2", "Opp", "", ""],
                ["", "start"]):
        b.parse_message(msg)
    feed = [
        ["", "switch", "p1a: Frost", "froslass, L50, M", "100/100"],
        ["", "switch", "p2a: Corv", "corviknight, L50, M", "100/100"],
        ["", "-weather", "RainDance"],
        ["", "-sidestart", "p1: Me", "move: Stealth Rock"],
        ["", "-sidestart", "p2: Opp", "move: Spikes"],
        ["", "-boost", "p1a: Frost", "spa", "2"],
        ["", "-status", "p2a: Corv", "brn"],
        ["", "-damage", "p1a: Frost", "78/100"],
        ["", "-damage", "p2a: Corv", "61/100"],
        # скамейка обеих сторон (для bench-блока 400 колонок)
        ["", "switch", "p1a: Aqua", "swampert, L50, M", "362/362"],
        ["", "switch", "p1a: Frost", "froslass, L50, M", "78/100"],
        ["", "switch", "p2a: Skarm", "skarmory, L50, F", "100/100"],
        ["", "switch", "p2a: Corv", "corviknight, L50, M", "61/100"],
        ["", "move", "p1a: Frost", "icebeam", "p2a: Corv"],
    ]
    for msg in feed:
        b.parse_message(msg)
    # |request| — единственный штатный способ дать покемонам приёмы и заполнить available_moves
    stats = {"atk": 100, "def": 90, "spa": 150, "spd": 100, "spe": 150}
    # |request| разбирается отдельным методом (parse_message его не знает)
    b.parse_request({
        "active": [{
            "moves": [{"id": "icebeam", "pp": 10, "maxpp": 16, "target": "normal", "disabled": False},
                      {"id": "shadowball", "pp": 15, "maxpp": 24, "target": "normal", "disabled": False},
                      {"id": "thunderbolt", "pp": 15, "maxpp": 24, "target": "normal", "disabled": False},
                      {"id": "protect", "pp": 10, "maxpp": 16, "target": "self", "disabled": False}],
            "canTerastallize": "Ice",
        }],
        "side": {"name": "Me", "id": "p1", "pokemon": [
            {"ident": "p1: Frost", "details": "froslass, L50, M", "condition": "78/100", "active": True,
             "stats": stats, "moves": ["icebeam", "shadowball", "thunderbolt", "protect"],
             "baseAbility": "cursedbody", "item": "heavydutyboots", "ability": "cursedbody",
             "teraType": "Ice"},
            {"ident": "p1: Aqua", "details": "swampert, L50, M", "condition": "362/362", "active": False,
             "stats": stats, "moves": ["earthquake", "icepunch"], "baseAbility": "torrent",
             "item": "leftovers", "ability": "torrent", "teraType": "Water"},
        ]},
    })
    return b


def inject_maps(player, our_fusion, opp_fusion, our_side="p1",
                our_team=None, opp_team=None, protect=PROTECT):
    """Кладёт фьюжн-карты/протекты туда, откуда их читают get_fusion_entry/get_protected_last_turn."""
    opp_side = "p2" if our_side == "p1" else "p1"
    store = {our_side: our_fusion, opp_side: opp_fusion,
             f"{our_side}_by_species": dict(our_team or {}),
             f"{opp_side}_by_species": dict(opp_team or {})}
    player._fusion_stats[TAG] = store
    player._protect_state[TAG] = {f"last_{our_side}": bool(protect[0]), f"last_{opp_side}": bool(protect[1])}
    return player


def core_obs(battle, our_fusion, opp_fusion, protect=PROTECT, our_team=None, opp_team=None):
    return embed_battle_with_fusion(
        battle, our_fusion, opp_fusion,
        our_protected_last_turn=protect[0], opp_protected_last_turn=protect[1],
        our_team_fusions=our_team, opp_team_fusions=opp_team,
    )


def player_obs(battle, player):
    """Путь PolicyPlayer/HeuristicRecorder: те же аргументы, но взятые ИХ геттерами."""
    return embed_battle_with_fusion(
        battle,
        player.get_fusion_entry(battle, is_ours=True),
        player.get_fusion_entry(battle, is_ours=False),
        our_protected_last_turn=player.get_protected_last_turn(battle, is_ours=True),
        opp_protected_last_turn=player.get_protected_last_turn(battle, is_ours=False),
        our_team_fusions=player.get_team_fusion_map(battle, is_ours=True) if hasattr(player, "get_team_fusion_map") else None,
        opp_team_fusions=player.get_team_fusion_map(battle, is_ours=False) if hasattr(player, "get_team_fusion_map") else None,
    )


def find_example_env(env, depth: int = 8):
    cur = env
    for _ in range(depth):
        if hasattr(cur, "battle1") and hasattr(cur, "embed_battle"):
            return cur
        cur = getattr(cur, "env", None)
        if cur is None:
            return None
    return None


# ------------------------------------------------------------------- часть 1: layout ---
LAYOUT = [
    ("our_moves(4x30)", 120), ("faint_hp", 4), ("status", 14), ("hazards", 8), ("switches", 4),
    ("boosts", 14), ("actual_stats", 12), ("ability", 40), ("weather_field", 10),
    ("trick/tailwind/screens", 9), ("speed", 1), ("revealed", 2), ("semi_invuln", 2),
    ("sub_damage", 2), ("restricted", 1), ("volatiles", 22), ("items", 22), ("bench", 400),
    ("vulnerability", 2), ("tera_meta", 3), ("is_tera", 2), ("tera_type", 19), ("protect", 2),
    ("damage", DAMAGE_BLOCK_SIZE), ("type_matchup", TYPE_MATCHUP_BLOCK_SIZE),
]


def layout_offsets():
    out = {}
    off = 0
    for name, size in LAYOUT:
        out[name] = (off, off + size)
        off += size
    return out, off


def part1_layout():
    print("=" * 78)
    print("ЧАСТЬ 1. Раскладка признаков: размерности и границы сегментов")
    print("=" * 78)
    from agents.training import MIN_PREFIX_OBS_DIM

    off, total = layout_offsets()
    check_eq("сумма сегментов == N_FEATURES", total, N_FEATURES)
    check_eq("блок урона лежит в хвосте: граница префикса", off["damage"][0], MIN_PREFIX_OBS_DIM)
    check_eq("735/715: граница префикса == MIN_PREFIX_OBS_DIM", MIN_PREFIX_OBS_DIM, 715)
    check_eq("802 = 715 + зеркальный блок (MIRROR_BASE)", MIN_PREFIX_OBS_DIM + MIRROR_BASE, 802)
    check_eq("TEAM_BASE - MIRROR_BASE == 12 (их приёмы по нашему активному)",
             TEAM_BASE - MIRROR_BASE, 12)
    check_eq("FLAGS_BASE - TEAM_BASE == 24 (их приёмы x наши 6 слотов)",
             FLAGS_BASE - TEAM_BASE, 24)
    check_eq("DAMAGE_BLOCK_SIZE - FLAGS_BASE == len(EFFECT_FLAGS)",
             DAMAGE_BLOCK_SIZE - FLAGS_BASE, len(EFFECT_FLAGS))
    check_eq("DAMAGE_BLOCK_SIZE", DAMAGE_BLOCK_SIZE, 155)
    check_eq("блок урона кончается там, где начинается блок типов соперника",
             off["type_matchup"][0], TYPE_MATCHUP_BASE)
    check_eq("TYPE_MATCHUP_BASE = префикс + блок урона",
             MIN_PREFIX_OBS_DIM + DAMAGE_BLOCK_SIZE, TYPE_MATCHUP_BASE)
    check_eq("блок типов соперника в хвосте (после него ничего нет)",
             off["type_matchup"][1], N_FEATURES)
    check_eq("TYPE_MATCHUP_BLOCK_SIZE = типы активного + 12 строк + 6 флагов",
             19 + 2 * 8 * 6 + 6, TYPE_MATCHUP_BLOCK_SIZE)

    # порядок: сдвиг одного признака должен менять колонки ТОЛЬКО своего сегмента
    b = make_battle()
    base = core_obs(b, FUSION, FUSION)
    check_eq("базовый obs: размерность", base.shape[0], N_FEATURES)
    check_eq("в фикстуре действительно есть хазард на нашей стороне (иначе проверка вакуумная)",
             float(base[off["hazards"][0]]), 1.0)

    b2 = copy.deepcopy(b)
    b2._weather = {Weather.SANDSTORM: 5}          # было RainDance
    diff = np.where(core_obs(b2, FUSION, FUSION) != base)[0]
    check_windows("погода меняет только сегмент weather_field", diff, [off["weather_field"]])
    check_eq("погода: сменились ровно 2 колонки (дождь ушёл, песчаная буря пришла)", int(diff.size), 2)

    b3 = copy.deepcopy(b)
    b3._side_conditions[SideCondition.SPIKES] = 3   # было 0 -> 1.0 в признаке
    diff3 = np.where(core_obs(b3, FUSION, FUSION) != base)[0]
    check_windows("хазарды меняют только сегмент hazards", diff3, [off["hazards"]])

    b4 = copy.deepcopy(b)
    if b4.active_pokemon is not None:
        b4.active_pokemon._terastallized = True
        b4.active_pokemon._terastallized_type = b4.active_pokemon.tera_type
    diff4 = np.where(core_obs(b4, FUSION, FUSION) != base)[0]
    # тера честно меняет и STAB-флаги приёмов, и блок урона (типы стали другими) —
    # поэтому окна: приёмы, tera-meta, is_tera, tera_type, damage
    check_windows("тера активного меняет только связанные сегменты", diff4,
                  [off["our_moves(4x30)"], off["tera_meta"], off["is_tera"], off["tera_type"],
                   off["damage"], off["type_matchup"]])
    check(f"тера: колонка our_is_tera[{off['is_tera'][0]}] изменилась",
          off["is_tera"][0] in diff4, f"diff={list(diff4)}")

    b5 = copy.deepcopy(b)
    bench = [m for m in b5.team.values() if m is not b5.active_pokemon]
    if bench:
        bench[0]._current_hp = max(int(bench[0]._current_hp) - 12, 1)
    diff5 = np.where(core_obs(b5, FUSION, FUSION) != base)[0]
    # HP резерва влияет и на bench-блок, и на сводку свитчей (avg_hp по резервам) — это её смысл
    check_windows("HP скамейки меняет только bench и сводку свитчей", diff5,
                  [off["switches"], off["bench"]])
    check(f"HP скамейки: изменилась колонка bench[{off['bench'][0]}..]", 
          any(off["bench"][0] <= i < off["bench"][1] for i in diff5), f"diff={list(diff5)}")

    # (e) типы соперника: сдвигаем тип активного оппонента -> меняются наши мультипликаторы по
    #     нему (our_moves), блок урона и блок типов; всё остальное трогать не должен
    b6 = copy.deepcopy(b)
    if b6.opponent_active_pokemon is not None:
        b6.opponent_active_pokemon._temporary_types = [PokemonType.WATER, None]
    diff6 = np.where(core_obs(b6, FUSION, FUSION) != base)[0]
    check_windows("тип активного оппонента меняет только приёмы/урон/блок типов", diff6,
                  [off["our_moves(4x30)"], off["damage"], off["type_matchup"]])
    check("тип оппонента: изменились колонки блока типов",
          any(off["type_matchup"][0] <= i < off["type_matchup"][1] for i in diff6),
          f"diff={list(diff6)}")

    # (f) матрица «тип соперника x наш покемон»: ПЕРЕСЧИТЫВАЕМ ВСЕ 12 строк и все 6 слотов
    from agents.features import _matchup_team_slots, _type_matchup_block
    block = _type_matchup_block(b)
    rows_base, conf_base = 19, 19 + 2 * 8 * 6
    ours = _matchup_team_slots(b.active_pokemon, b.team)
    theirs = _matchup_team_slots(b.opponent_active_pokemon, b.opponent_team)

    filled, empty, worst_m = 0, 0, 0.0
    checked: list = []
    for i, mon in enumerate(theirs):
        if mon is None:
            continue
        t1, t2, src = effective_types(mon, battle=b)
        for j, atk in enumerate((t1, t2)):
            row = rows_base + (i * 2 + j) * 8
            if atk is None:
                empty += 1
                check(f"блок типов: пустой слот соперника {i}/{j} — строка нулевая",
                      float(block[row]) == 0.0 and float(np.abs(block[row + 1:row + 8]).sum()) == 0.0,
                      f"row={block[row:row + 8].tolist()}")
                continue
            filled += 1
            check(f"блок типов: флаг типа {i}/{j}",
                  float(block[row]) == 1.0, f"флаг={float(block[row])}")
            want_scalar = (TYPE_INDEX[atk] if atk in TYPE_INDEX else 0) / 18.0
            check(f"блок типов: скаляр типа {i}/{j}",
                  abs(float(block[row + 1]) - want_scalar) < 1e-6,
                  f"got={float(block[row + 1]):.6f} want={want_scalar:.6f}")
            for k, mon2 in enumerate(ours):
                mt1, mt2 = _eff_types(mon2) if mon2 is not None else (None, None)
                if mt1 is None and mt2 is None:
                    continue
                want = damage_multiplier_safe(atk, mt1, mt2, type_chart=GENDATA_TYPE_CHART)
                got = float(block[row + 2 + k])
                worst_m = max(worst_m, abs(got - want))
                checked.append(f"соп{i}/{j}->наш{k}: {got:.2f}/{want:.2f}")
    check("блок типов: строки с типами есть (проверка не вакуумная)", filled >= 1,
          f"заполнено {filled}, пусто {empty}")
    check("блок типов: матрица == ручной пересчёт по типам (все строки x все слоты)",
          worst_m <= 1e-6, f"max|Δ|={worst_m:.3e}; " + " | ".join(checked[:8]))
    check("блок типов: есть хотя бы один не-нейтральный множитель (иначе матрица бесполезна)",
          any(abs(float(block[rows_base + r * 8 + 2 + k]) - 1.0) > 0.01
              for r in range(12) for k in range(6)),
          "все множители = 1.0")
    check("блок типов: multi-hot нашего активного == _type_multi_hot(активный)",
          bool(np.array_equal(np.asarray(block[:19]), _type_multi_hot(b.active_pokemon))),
          f"multi-hot={block[:19].tolist()}")
    check("блок типов: у нашего активного действительно есть типы в multi-hot",
          float(np.asarray(block[:19]).sum()) >= 1.0, f"сумма={float(np.asarray(block[:19]).sum())}")
    check("блок типов: флаги «тип подтверждён сервером» — 0/1 по слотам соперника",
          all(float(v) in (0.0, 1.0) for v in block[conf_base:conf_base + 6]),
          f"флаги={block[conf_base:conf_base + 6].tolist()}")
    check("блок типов: без серверного typechange тип считается выведенным (флаг 0)",
          float(block[conf_base]) == 0.0, f"флаг={float(block[conf_base])}")

    # сервер прислал тип (как при выходе на поле): флаг становится 1, а матрица — по этому типу
    block6 = _type_matchup_block(b6)          # b6: у активного оппонента _temporary_types = Water
    check("блок типов: typechange с сервера -> флаг подтверждения = 1",
          float(block6[conf_base]) == 1.0, f"флаг={float(block6[conf_base])}")
    want_scalar_w = (TYPE_INDEX[PokemonType.WATER] if PokemonType.WATER in TYPE_INDEX else 0) / 18.0
    check("блок типов: typechange с сервера -> в блоке именно серверный тип",
          abs(float(block6[rows_base + 1]) - want_scalar_w) < 1e-6,
          f"скаляр={float(block6[rows_base + 1]):.6f} хотели={want_scalar_w:.6f}")
    worst_w, checked_w = 0.0, 0
    for k, mon2 in enumerate(ours):
        mt1, mt2 = _eff_types(mon2) if mon2 is not None else (None, None)
        if mt1 is None and mt2 is None:
            continue                      # пустой слот команды: в блоке 0.0 (мон отсутствует)
        want = damage_multiplier_safe(PokemonType.WATER, mt1, mt2, type_chart=GENDATA_TYPE_CHART)
        worst_w = max(worst_w, abs(float(block6[rows_base + 2 + k]) - want))
        checked_w += 1
    check("блок типов: множители по серверному типу == ручной расчёт",
          worst_w <= 1e-6 and checked_w >= 1, f"слотов сверено {checked_w}, max|Δ|={worst_w:.3e}")

    ROWS.append(("раскладка (офлайн)", N_FEATURES, "—", 0.0, h8(base)))


# ------------------------------------------------------ часть 2: пути вычисления obs ---
def part2_paths():
    print("=" * 78)
    print("ЧАСТЬ 2. Один и тот же бой -> один и тот же obs на всех путях")
    print("=" * 78)
    from agents.env import ExampleEnv

    battle = make_battle()
    our_team = {"froslass": FUSION, "swampert": FUSION}
    opp_team = {"corviknight": FUSION, "skarmory": FUSION}
    ref = core_obs(battle, FUSION, FUSION, our_team=our_team, opp_team=opp_team)
    check_eq("эталон (dataset/BС-путь): размерность", ref.shape[0], N_FEATURES)
    ROWS.append(("dataset / BC (core)", ref.shape[0], "нет", 0.0, h8(ref)))

    # (1) путь обучения: ExampleEnv.embed_battle (как в роллауте)
    env = ExampleEnv(battle_format=_cfg.BATTLE_FORMAT, log_level=40, open_timeout=None,
                     start_listening=False)
    inject_maps(env.agent1, FUSION, FUSION, our_team=our_team, opp_team=opp_team)
    env.battle1 = battle
    env_obs = np.asarray(env.embed_battle(battle), dtype=np.float32)
    d = check_close("RL-обучение (ExampleEnv.embed_battle) == эталон", env_obs, ref)
    ROWS.append(("RL-обучение (env)", env_obs.shape[0], "VecNormalize снаружи", d, h8(env_obs)))

    # (2) путь инференса/оценки: PolicyPlayer.embed_battle (БЕЗ нормализатора = сырой obs)
    player_raw = PolicyPlayer(policy=None, battle_format=_cfg.BATTLE_FORMAT,
                              log_level=40, start_listening=False)
    inject_maps(player_raw, FUSION, FUSION, our_team=our_team, opp_team=opp_team)
    p_obs = np.asarray(player_raw.embed_battle(battle), dtype=np.float32)
    d = check_close("оценка/инференс (PolicyPlayer.embed_battle, без норм.) == эталон", p_obs, ref)
    ROWS.append(("оценка/инференс (PolicyPlayer)", p_obs.shape[0], "нет (сырой)", d, h8(p_obs)))

    # (3) путь сборщика датасета: HeuristicRecorder берёт аргументы своими геттерами
    recorder = HeuristicRecorder(dataset=[], raw_dataset=[], battle_format=_cfg.BATTLE_FORMAT,
                                 log_level=40, start_listening=False)
    inject_maps(recorder, FUSION, FUSION, our_team=our_team, opp_team=opp_team)
    r_obs = np.asarray(player_obs(battle, recorder), dtype=np.float32)
    d = check_close("сборщик датасета (HeuristicRecorder) == эталон", r_obs, ref)
    ROWS.append(("сборщик датасета", r_obs.shape[0], "нет", d, h8(r_obs)))

    # (4) критично: env видит ТЕ ЖЕ фьюжн-аргументы (потеря командных карт ломала bench/урон)
    env2 = ExampleEnv(battle_format=_cfg.BATTLE_FORMAT, log_level=40, open_timeout=None,
                      start_listening=False)
    env2.battle1 = battle
    inject_maps(env2.agent1, FUSION, FUSION, our_team=our_team, opp_team=opp_team)
    env_team_obs = np.asarray(env2.embed_battle(battle), dtype=np.float32)
    d = check_close("env передаёт командные фьюжн-карты так же, как датасет", env_team_obs, ref)
    # и наоборот: без карт obs обязан ОТЛИЧАТЬСЯ (иначе проверка вакуумная)
    env3 = ExampleEnv(battle_format=_cfg.BATTLE_FORMAT, log_level=40, open_timeout=None,
                      start_listening=False)
    env3.battle1 = battle
    inject_maps(env3.agent1, FUSION, FUSION, our_team=None, opp_team=None)
    no_team = np.asarray(env3.embed_battle(battle), dtype=np.float32)
    check("без командных карт obs отличается (проверка не вакуумная)",
          int((no_team != ref).sum()) > 0, f"различий {int((no_team != ref).sum())}")

    # (5) детерминизм: повторный вызов даёт тот же вектор (obs не зависит от скрытого состояния)
    again = np.asarray(env2.embed_battle(battle), dtype=np.float32)
    check_close("повторный вызов embed_battle бит-в-бит", again, ref, atol=0.0)

    for e in (env, env2, env3):
        safe_close(e)
    return battle, our_team, opp_team


# ------------------------------------------------------- часть 3: нормализация obs ---
def make_vecnormalize(dim=N_FEATURES, seed=0):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class E(gym.Env):
        observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 4.0, shape=(dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(26,), dtype=np.int8)})
        action_space = spaces.Discrete(26)

        def reset(self, *, seed=None, options=None):
            return {"observation": self.observation_space["observation"].sample(),
                    "action_mask": np.ones(26, dtype=np.int8)}, {}

        def step(self, a):
            return self.reset()[0], 0.0, False, False, {}

    vec = VecNormalize(DummyVecEnv([lambda: E()]), norm_obs=True, norm_reward=False,
                       gamma=0.99, norm_obs_keys=["observation"])
    rng = np.random.default_rng(seed)
    vec.obs_rms["observation"].update(rng.normal(size=(512, dim)).astype(np.float32))
    return vec


def part3_norm(tmpdir):
    print("=" * 78)
    print("ЧАСТЬ 3. Нормализация: обучение == оценка == инференс (один и тот же вектор)")
    print("=" * 78)
    battle = make_battle()
    raw = core_obs(battle, FUSION, FUSION)
    vec = make_vecnormalize()
    rng = np.random.default_rng(1)
    raw_batch = np.vstack([raw, rng.normal(size=(3, N_FEATURES)).astype(np.float32)])

    # путь обучения: VecNormalize.normalize_obs (dict-obs, ключ observation)
    train_norm = np.asarray(vec.normalize_obs({"observation": raw_batch.copy()})["observation"],
                            dtype=np.float32)
    check_eq("обучение: нормализованный obs сохраняет размерность", train_norm.shape[1], N_FEATURES)
    check("обучение: нормализация реально меняет значения",
          float(np.abs(train_norm - raw_batch).max()) > 1e-3,
          f"max|Δ|={float(np.abs(train_norm - raw_batch).max()):.3f}")

    # путь оценки: LiveVecNormalizeAdapter (та же живая статистика)
    adapter = LiveVecNormalizeAdapter(vec, target_dim=N_FEATURES)
    d = check_close("оценка (LiveVecNormalizeAdapter) == обучение", adapter.normalize(raw_batch), train_norm)

    # путь инференса: статистика с диска (VecNormStats) — index.py / play_trained.py
    vpath = os.path.join(tmpdir, "vecnormalize.pkl")
    saved = save_vecnormalize_with_meta(vec, vpath)
    check("инференс: VecNormalize сохранён с сайдкаром", bool(saved))
    disk = load_vecnorm_stats(vpath, N_FEATURES)
    check("инференс: статистика прочитана с диска", disk is not None)
    if disk is not None:
        d2 = check_close("инференс (VecNormStats с диска) == обучение", disk.normalize(raw_batch), train_norm)
        ROWS.append(("нормализация: инференс", N_FEATURES, disk.describe()[:40], d2, h8(train_norm)))
    try:
        meta = json.load(open(vpath + ".meta.json", encoding="utf-8"))
    except Exception:
        meta = {}
    check_eq("сайдкар статистики: obs_dim", int(meta.get("obs_dim", -1)), N_FEATURES)
    check("сайдкар статистики: хеш кода признаков совпадает с текущим кодом",
          meta.get("features_hash") == features_fingerprint(),
          f"{meta.get('features_hash')} vs {features_fingerprint()}")

    # путь play_trained.py: stats.normalize(player.embed_battle(battle)) вместо obs_normalizer
    if disk is not None:
        pl = PolicyPlayer(policy=None, battle_format=_cfg.BATTLE_FORMAT, log_level=40,
                          start_listening=False)
        inject_maps(pl, FUSION, FUSION)
        play_trained_style = disk.normalize(np.asarray(pl.embed_battle(battle), dtype=np.float32))
        pl2 = PolicyPlayer(policy=None, battle_format=_cfg.BATTLE_FORMAT, log_level=40,
                           start_listening=False, obs_normalizer=disk)
        inject_maps(pl2, FUSION, FUSION)
        index_style = np.asarray(pl2.embed_battle(battle), dtype=np.float32)
        d3 = check_close("инференс: play_trained-стиль == index-стиль (obs_normalizer)",
                         play_trained_style, index_style)
        d4 = check_close("инференс: obs на боевом снимке == нормализованный эталон",
                         index_style, disk.normalize(raw))
        ROWS.append(("нормализация: play_trained", N_FEATURES, "VecNormStats", max(d3, d4), h8(index_style)))
    vec.close()
    return vec, vpath


# ------------------------------------------------------------ часть 4: живой прогон ---
def port_open(host="localhost", port=8000, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_server(log):
    if port_open():
        log("Showdown уже слушает localhost:8000")
        return True
    d = os.environ.get("PYBOT_SHOWDOWN_DIR", "/tmp/ps/node_modules/pokemon-showdown")
    if not os.path.isfile(os.path.join(d, "pokemon-showdown")):
        log(f"нет сервера и нет каталога pokemon-showdown ({d}) — live-стадии пропускаю")
        return False
    os.makedirs(os.path.join(d, "logs", "repl"), exist_ok=True)
    log(f"поднимаю локальный Showdown из {d}...")
    subprocess.Popen(["node", "pokemon-showdown", "start", "--no-security"], cwd=d,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(1)
        if port_open():
            log("сервер поднят")
            return True
    log("сервер не поднялся за 30 с — live-стадии пропускаю")
    return False


class _EnvRecorder:
    """Обёртка вокруг ExampleEnv: пишет (снимок боя, сырой obs, фьюжн-аргументы) каждого шага."""

    def __init__(self, core_env):
        self.core = core_env
        self.log: list = []

    def snap(self, obs_dict):
        try:
            b = self.core.battle1
            if b is None:
                return
            tag = b.battle_tag
            store = self.core.agent1._fusion_stats.get(tag, {})
            side = getattr(b, "player_role", "p1") or "p1"
            opp = "p2" if side == "p1" else "p1"
            self.log.append({
                "battle": copy.deepcopy(b),
                "obs": np.asarray(obs_dict["observation"], dtype=np.float32).reshape(-1).copy(),
                "our_fusion": store.get(side),
                "opp_fusion": store.get(opp),
                "our_team": store.get(f"{side}_by_species"),
                "opp_team": store.get(f"{opp}_by_species"),
                "protect": (self.core.agent1._protect_state.get(tag, {}).get(f"last_{side}", False),
                            self.core.agent1._protect_state.get(tag, {}).get(f"last_{opp}", False)),
            })
        except Exception as e:  # noqa: BLE001
            print(f"    (recorder: пропустил снимок: {e})")


class _RecordingGym(gym.Wrapper):
    """gym.Wrapper, который на reset/step снимает состояние боя для сверки с core-функцией."""

    def __init__(self, env, recorder: _EnvRecorder):
        super().__init__(env)
        self.recorder = recorder

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.recorder.snap(out[0] if isinstance(out, tuple) else out)
        return out

    def step(self, action):
        out = self.env.step(action)
        self.recorder.snap(out[0])
        return out


def collect_dataset(fmt, n_battles, log):
    """Стадия 1: сыграть бои эвристикой и записать датасет (obs, mask, action) + сырые снимки."""
    from agents import training as tr
    tr.BATTLE_FORMAT = fmt  # HeuristicRecorder получает формат явно, но подстрахуемся
    recorder = HeuristicRecorder(dataset=[], raw_dataset=[], battle_format=fmt,
                                 max_concurrent_battles=2)
    opponent = SimpleHeuristicsPlayer(battle_format=fmt, max_concurrent_battles=2)
    t0 = time.time()
    asyncio.run(recorder.battle_against(opponent, n_battles=n_battles))
    battles = getattr(recorder, "battles", None) or getattr(recorder, "_battles", {}) or {}
    log(f"собрано боёв: {len(battles)}, переходов: {len(recorder.dataset)} "
        f"({time.time() - t0:.1f} с)")
    return recorder, battles


def fresh_usernames():
    """Свежие имена аккаунтов на прогон.

    Стоковый Showdown: занятое имя -> выдаёт другое, а имена со служебными символами
    (например с "_") молча урезает -> poke-env ждёт своего имени, не логинится и бой виснет.
    """
    import string
    from poke_env.ps_client.account_configuration import AccountConfiguration

    token = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    orig = AccountConfiguration.generate.__func__

    def gen(cls, key, rand=False):
        clean = "".join(ch for ch in key if ch.isalnum())[:12]
        return orig(cls, f"{clean}{token}", rand)

    AccountConfiguration.generate = classmethod(gen)


def live_stages(fmt, fast, tmpdir, log, run):
    from agents import training as tr
    from agents.checkpoint_utils import load_policy_compat
    from agents.env import ExampleEnv
    from agents.policy import MaskedActorCriticPolicy
    from agents.policy_player import _sync_model_n_envs
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # стоковый Showdown не знает fusion-формат: подменяем формат во всех модулях,
    # которые читают BATTLE_FORMAT из своего namespace (иначе env уйдёт в чужой формат
    # и challenge повиснет с TimeoutError "Agent is not challenging")
    fresh_usernames()
    _cfg.BATTLE_FORMAT = fmt
    tr.BATTLE_FORMAT = fmt
    from agents import env as _envmod
    _envmod.BATTLE_FORMAT = fmt
    os.environ.setdefault("PYBOT_SELF_PLAY_NORM", "0")   # быстрый прогон: без снапшотов-соперников

    order = ["S1", "S2", "S3", "S4", "S5"]
    log(f"live-стадии к запуску: {', '.join(run) if run else '—'}")
    if not run:
        return
    vec = None      # адресные прогоны (--only S1) не доходят до стадии вектора-нормализатора
    disk = None

    if "S1" in run:
    # ---------------------------------------------------------------- S1: датасет ---
        print("-" * 78)
        print("S1. Сбор датасета (стадия BC-претрейна)")
        recorder, battles = collect_dataset(fmt, 1 if fast else 2, log)
        if not recorder.dataset:
            check("S1: датасет не пустой", False, "боёв не сыграно?")
            return
        ds = recorder.dataset                      # (obs, mask, action, tag) x N
        obs_in_ds = np.stack([d[0] for d in ds]).astype(np.float32)
        mask_in_ds = np.stack([d[1] for d in ds])
        check_eq("S1: obs_dim в датасете", obs_in_ds.shape[1], N_FEATURES)
        check_eq("S1: маска действий", mask_in_ds.shape[1], 26)
        check("S1: obs конечны", bool(np.isfinite(obs_in_ds).all()))

        # ключевое: датасет == пересчёт из СЫРЫХ снимков текущим кодом признаков
        recomputed = tr._recompute_dataset_from_raw(recorder.raw_dataset, battles)
        rec_sorted = sorted(recomputed, key=lambda x: (x[3], h8(x[0])))
        ds_sorted = sorted(ds, key=lambda x: (x[3], h8(x[0])))
        same_tags = [r[3] for r in rec_sorted] == [d[3] for d in ds_sorted]
        check_eq("S1: пересчёт из сырого кэша даёт столько же переходов", len(recomputed), len(ds))
        if same_tags and recomputed:
            rec_obs = np.stack([r[0] for r in rec_sorted]).astype(np.float32)
            ds_obs = np.stack([d[0] for d in ds_sorted]).astype(np.float32)
            d = check_close("S1: obs из датасета == пересчёт из сырого снимка (1:1)", rec_obs, ds_obs, atol=0.0)
            d2 = check_close("S1: маски совпадают", np.stack([r[1] for r in rec_sorted]), np.stack([d[1] for d in ds_sorted]))
            ROWS.append(("S1 датасет (BC)", obs_in_ds.shape[1], "нет (сырой)", max(d, d2), h8(ds_obs)))
        else:
            check("S1: теги переходов совпадают для сравнения", same_tags)

        dataset_with_ret = tr._compute_bc_returns(ds, battles)
        npz = os.path.join(tmpdir, "dataset_test.npz")
        tr.save_dataset(dataset_with_ret, npz)
        info = tr.validate_bc_dataset(npz, N_FEATURES)
        check_eq("S1: валидатор BC принял датасет (obs_dim)", int(info["obs_dim"]), N_FEATURES)
        check("S1: в датасете есть ret (returns для BC)", bool(info.get("has_ret")))

    

    if "S2" in run:
    # --------------------------------------------------------- S2: BC-претрейн ---
        print("-" * 78)
        print("S2. BC-претрейн (1 эпоха, тестовая модель)")
        # env ровно тот, что в обучении (ExampleEnv.create_env -> Monitor(DecisionWrapper(ExampleEnv))).
        # Self-play оппоненты отключены только ради скорости прогона: они загружают 7 снапшотов
        # с миграцией 418->870 и на признаки не влияют.
        from agents import env as envmod
        envmod._make_self_play_opponents = lambda *a, **k: []
        rec = _EnvRecorder(None)          # core заполним из созданного env

        def make_training_env():
            wrapped = ExampleEnv.create_env()
            if rec.core is None:
                rec.core = find_example_env(wrapped)
            return _RecordingGym(wrapped, rec)

        raw_env = DummyVecEnv([make_training_env])
        vec = VecNormalize(raw_env, norm_obs=True, norm_reward=False, gamma=0.99,
                           norm_obs_keys=["observation"])
        vec.training = True
        ppo = PPO(MaskedActorCriticPolicy, vec, device="cpu", verbose=0, n_steps=64, batch_size=32,
                  n_epochs=1, policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=64)))
        check_eq("S2: модель собрана под N_FEATURES", int(ppo.observation_space["observation"].shape[0]),
                 N_FEATURES)
        first_w = ppo.policy.features_extractor.net[0].weight
        check_eq("S2: вход экстрактора признаков", int(first_w.shape[1]), N_FEATURES)

        # Что реально видит политика: forward-hook экстрактора признаков. Спаи на
        # policy.forward/evaluate_actions слепы — BC считает лосс через policy.extract_features.
        seen_bc: list = []

        def _grab_hook(module, args, output, bucket):
            try:
                tens = args[0]["observation"] if isinstance(args[0], dict) else args[0]
                arr = np.asarray(tens.detach().cpu(), dtype=np.float32)
                bucket.append(arr.reshape(-1) if arr.ndim == 1 else arr)
            except Exception:
                pass

        hook = ppo.policy.features_extractor.register_forward_hook(
            lambda m, a, o: _grab_hook(m, a, o, seen_bc))
        tr.pretrain_policy_bc(ppo, npz, epochs=1, batch_size=32, normalize=True,
                              contrastive=True, neg_weight=0.3)
        hook.remove()
        check("S2: BC прошёл, статистика нормализации прогрета",
              float(np.asarray(vec.obs_rms["observation"].count)) > 0,
              f"count={float(np.asarray(vec.obs_rms['observation'].count)):.0f}")
        check_eq("S2: статистика нормализации размерности N_FEATURES",
                 int(np.asarray(vec.obs_rms["observation"].mean).size), N_FEATURES)
        adapter = LiveVecNormalizeAdapter(vec, target_dim=N_FEATURES)
        bc_rows = stack_rows(seen_bc)
        check("S2: политика получала вход (hook сработал)", bc_rows.shape[0] > 0,
              f"батчей {len(seen_bc)}, строк {bc_rows.shape[0]}")
        check("S2: во входе политики нет NaN/Inf", bool(np.isfinite(bc_rows).all()),
              f"нечисловых значений: {int((~np.isfinite(bc_rows)).sum())}")
        # вход BC должен быть ровно normalize() от obs датасета: ни сырых, ни дважды нормализованных
        norm_ds = np.asarray(vec.normalize_obs({"observation": obs_in_ds.copy()})["observation"],
                             dtype=np.float32)
        dist = nearest_row_dist(bc_rows, norm_ds)
        check("S2: вход BC == нормализация obs из датасета (не сырой и не дважды)", dist <= 1e-5,
              f"макс. из ближайших max|Δ| по строкам = {dist:.3e}")
        raw_shift = float(np.abs(norm_ds - obs_in_ds).max())
        check("S2: нормализация не тождественна (иначе проверка вакуумная)", raw_shift > 1e-4,
              f"max|normed - raw| = {raw_shift:.3e}")
        if bc_rows.shape[0]:
            ROWS.append(("S2 BC (вход политики)", bc_rows.shape[1], "VecNormalize", dist,
                         h8(bc_rows[0])))

    

    if "S3" in run:
    # ------------------------------------------------------------ S3: RL-обучение ---
        print("-" * 78)
        print("S3. RL-обучение (короткий роллаут на том же env)")
        _sync_model_n_envs(ppo, vec)
        ppo.set_env(vec)

        def hook_inputs(bucket):
            return ppo.policy.features_extractor.register_forward_hook(
                lambda m, a, o: _grab_hook(m, a, o, bucket))

        # (A) реальная конфигурация обучения: статистика нормализации обновляется каждый шаг
        seen_rl: list = []
        h = hook_inputs(seen_rl)
        ppo.learn(total_timesteps=64 if not fast else 32, reset_num_timesteps=True, progress_bar=False)
        h.remove()
        check("S3: роллаут прошёл (шаги > 0)", int(ppo.num_timesteps) > 0, f"{ppo.num_timesteps}")
        check("S3: снимки боя записаны", len(rec.log) > 1, f"{len(rec.log)}")
        if not rec.log:
            return
        # (а) obs из env == пересчёт core по тому же снимку с теми же аргументами
        worst = 0.0
        for item in rec.log[:20]:
            ref = embed_battle_with_fusion(
                item["battle"], item["our_fusion"], item["opp_fusion"],
                our_protected_last_turn=float(item["protect"][0]),
                opp_protected_last_turn=float(item["protect"][1]),
                our_team_fusions=item["our_team"], opp_team_fusions=item["opp_team"])
            worst = max(worst, float(np.abs(ref - item["obs"]).max()))
        check("S3: obs на шаге == пересчёт core по снимку боя", worst <= 1e-6, f"max|Δ|={worst:.3e}")
        d = check_close("S3: нормализация обучения == адаптер оценки",
                        adapter.normalize(rec.log[0]["obs"]),
                        vec.normalize_obs({"observation": rec.log[0]["obs"][None, :]})["observation"][0])

        rl_rows = stack_rows(seen_rl)
        check("S3: политика получала вход на роллауте (hook сработал)", rl_rows.shape[0] > 0,
              f"батчей {len(seen_rl)}, строк {rl_rows.shape[0]}")
        check("S3: во входе политики на роллауте нет NaN/Inf", bool(np.isfinite(rl_rows).all()),
              f"нечисловых значений: {int((~np.isfinite(rl_rows)).sum())}")
        check_eq("S3: размерность входа политики", int(rl_rows.shape[1]), N_FEATURES)

        # (б) то, на чём УЧИТСЯ политика (буфер роллаута), == то, что она видела на шаге
        buf = getattr(ppo.rollout_buffer, "observations", None)
        if isinstance(buf, dict):
            arr = np.asarray(buf["observation"], dtype=np.float32)
        elif buf is not None:
            arr = np.asarray(buf, dtype=np.float32)
        else:
            arr = np.zeros((0, 0), np.float32)
        buf_obs = arr.reshape(-1, arr.shape[-1]) if arr.size else np.zeros((0, 0), np.float32)
        d_buf = nearest_row_dist(buf_obs, rl_rows)
        check("S3: obs в буфере обучения == вход политики на том же шаге", d_buf <= 1e-6,
              f"строк в буфере {buf_obs.shape[0]}, max|Δ| до ближайшего входа = {d_buf:.3e}")

        # (в) точное равенство с нормализацией сырых obs, снятых с env.
        # При включённой статистике mean/var меняются на каждом шаге (нормализация шага t идёт
        # уже по статистике, включающей obs_t) — такое равенство воспроизводимо только
        # с замороженной статистикой, что и проверяем: статистика та же, формула та же.
        stage_a_raw = np.stack([it["obs"] for it in rec.log]).astype(np.float32)
        vec.training = False
        rec.log.clear()
        seen_frozen: list = []
        h = hook_inputs(seen_frozen)
        ppo.learn(total_timesteps=32, reset_num_timesteps=False, progress_bar=False)
        h.remove()
        vec.training = True
        frozen = stack_rows(seen_frozen)
        raw_rows = np.stack([it["obs"] for it in rec.log]).astype(np.float32) if rec.log else np.zeros((0, 0), np.float32)
        if raw_rows.shape[0]:
            # первый вход роллаута — obs, снятый на последнем шаге прошлой стадии (перенос _last_obs)
            raw_rows = np.vstack([stage_a_raw[-1:], raw_rows])
        want_norm = np.asarray(vec.normalize_obs({"observation": raw_rows})["observation"],
                               dtype=np.float32) if raw_rows.size else np.zeros((0, 0), np.float32)
        check("S3: при замороженной статистике вход политики == normalize(сырой obs шага)",
              frozen.shape[0] > 0 and raw_rows.shape[0] > 0, f"строк входа {frozen.shape[0]}, obs с env {raw_rows.shape[0]}")
        worst_n = nearest_row_dist(frozen, want_norm)
        check("S3: нормализация роллаута == та же формула/статистика, что на шаге", worst_n <= 1e-6,
              f"max|Δ|={worst_n:.3e} (плюс перенесённый obs с прошлого роллаута)")
        d_norm_vs_raw = nearest_row_dist(frozen, raw_rows)
        check("S3: это нормализация, а не сырые obs (проверка не вакуумная)", d_norm_vs_raw > 1e-4,
              f"расстояние входа до сырых obs = {d_norm_vs_raw:.3e}")

        ROWS.append(("S3 RL (obs в шаге)", N_FEATURES, "VecNormalize", worst, h8(rec.log[0]["obs"] if rec.log else rl_rows[0])))
        ROWS.append(("S3 RL (вход политики)", int(frozen.shape[1]) if frozen.size else N_FEATURES,
                     "VecNormalize", max(worst_n, d_buf), h8(frozen[0] if frozen.size else rl_rows[0])))

    if "S4" in run:
    # ------------------------------------------------------- S4: оценка винрейта ---
        print("-" * 78)
        print("S4. Оценка винрейта (реальный бой против RandomPlayer)")

        class RecPlayer(PolicyPlayer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.records: list = []

            def embed_battle(self, battle):
                out = super().embed_battle(battle)     # внутри уже применён obs_normalizer
                try:
                    self.records.append({
                        "battle": copy.deepcopy(battle),
                        "seen": np.asarray(out, dtype=np.float32).copy(),
                        "raw": player_obs(battle, self),
                    })
                except Exception:
                    pass
                return out

        eval_player = RecPlayer(policy=ppo.policy, battle_format=fmt, max_concurrent_battles=2,
                                       obs_normalizer=adapter)
        rand = RandomPlayer(battle_format=fmt, max_concurrent_battles=2)
        asyncio.run(eval_player.battle_against(rand, n_battles=1))
        check("S4: решения в бою были", len(eval_player.records) > 0, f"{len(eval_player.records)}")
        if eval_player.records:
            rec0 = eval_player.records[0]
            check_eq("S4: obs модели в бою размерности N_FEATURES", rec0["seen"].shape[0], N_FEATURES)
            d_env = check_close("S4: сырой obs == путь env на том же снимке",
                                rec0["raw"], np.asarray(_offline_env_obs(rec0["battle"]), dtype=np.float32))
            d_norm = check_close("S4: то, что видит политика == адаптер(сырой obs)",
                                 rec0["seen"], adapter.normalize(rec0["raw"]))
            ROWS.append(("S4 оценка винрейта", rec0["seen"].shape[0], "LiveVecNormalizeAdapter",
                         max(d_env, d_norm), h8(rec0["seen"])))

    

    if "S5" in run:
    # ------------------------------------------------------------- S5: инференс ---
        print("-" * 78)
        print("S5. Инференс уже натренированной модели (save -> load -> бой)")
        model_path = os.path.join(tmpdir, "model_test")
        ppo.save(model_path)
        vpath = os.path.join(tmpdir, "vecnormalize_test.pkl")
        check("S5: статистика сохранена с сайдкаром", bool(save_vecnormalize_with_meta(vec, vpath)))
        _cfg.VECNORM_PATH = vpath
        import index as idx
        disk = idx.load_obs_normalizer(log=log)
        check("S5: index.load_obs_normalizer прочитал статистику", disk is not None)
        check_eq("S5: размерность статистики на инференсе",
                 int(np.asarray(load_vecnorm_stats(vpath, N_FEATURES).mean).size), N_FEATURES)
        ppo_mig, mig_info = load_policy_compat(model_path + ".zip", N_FEATURES)
        check("S5: модель загружена (migration-хелпер)", bool(mig_info.get("loaded")), str(mig_info))
        check("S5: migration-хелпер не полез мигрировать (размерность уже N_FEATURES)",
              mig_info.get("migrated_from") is None, str(mig_info))
        check_eq("S5: загруженная политика смотрит на N_FEATURES",
                 int(ppo_mig.observation_space["observation"].shape[0]), N_FEATURES)
        ppo_inf = PPO.load(model_path, device="cpu")
        check_eq("S5: obs_dim загруженной модели",
                 int(ppo_inf.observation_space["observation"].shape[0]), N_FEATURES)

        inf_player = RecPlayer(policy=ppo_inf.policy, battle_format=fmt,
                                      max_concurrent_battles=2, obs_normalizer=disk)
        rand2 = RandomPlayer(battle_format=fmt, max_concurrent_battles=2)
        asyncio.run(inf_player.battle_against(rand2, n_battles=1))
        check("S5: решения в бою были", len(inf_player.records) > 0, f"{len(inf_player.records)}")
        if inf_player.records and disk is not None:
            rec0 = inf_player.records[0]
            d_raw = check_close("S5: сырой obs инференса == путь env на том же снимке",
                                rec0["raw"], np.asarray(_offline_env_obs(rec0["battle"]), dtype=np.float32))
            d_norm = check_close("S5: obs политики == статистика с диска (та же нормализация)",
                                 rec0["seen"], disk.normalize(rec0["raw"]))
            d_disk_live = check_close("S5: статистика с диска == живой VecNormalize обучения на том же obs",
                                      disk.normalize(rec0["raw"]), vec.normalize_obs(
                                          {"observation": rec0["raw"][None, :]})["observation"][0])
            ROWS.append(("S5 инференс (после reload)", rec0["seen"].shape[0], "VecNormStats (диск)",
                         max(d_raw, d_norm, d_disk_live), h8(rec0["seen"])))

    # кросс-стадийная сводка на снимках BC-датасета (там есть фьюжн-аргументы)
    if recorder.raw_dataset:
        fields = tr._raw_entry_fields(recorder.raw_dataset[0])
        if fields:
            (bcopy, mask, action, tag, our_f, opp_f, our_p, opp_p, our_t, opp_t) = fields
            ref = core_obs(bcopy, our_f, opp_f, protect=(our_p or 0.0, opp_p or 0.0),
                           our_team=our_t, opp_team=opp_t)
            env_obs_s = np.asarray(_offline_env_obs(bcopy, tag=tag, our_f=our_f, opp_f=opp_f,
                                                    our_team=our_t, opp_team=opp_t,
                                                    protect=(our_p, opp_p)), dtype=np.float32)
            d1 = check_close("S5/S1: env-путь == core на боевом снимке датасета", env_obs_s, ref)
            d2 = 0.0
            if disk is not None:
                d2 = check_close("S5/S1: нормализация диска == живая, на том же снимке",
                                 disk.normalize(ref),
                                 vec.normalize_obs({"observation": ref[None, :]})["observation"][0])
            ROWS.append(("S1-S5: один бой -> один obs", ref.shape[0], "VecNormStats",
                         max(d1, d2), h8(ref)))

    if vec is not None:
        try:
            vec.close()
        except Exception:
            pass


def _offline_env_obs(battle, tag=None, our_f=None, opp_f=None, our_team=None, opp_team=None,
                     protect=(0.0, 0.0)):
    """obs через ExampleEnv.embed_battle на произвольном снимке (офлайн, без сервера)."""
    from agents.env import ExampleEnv
    env = ExampleEnv(battle_format=_cfg.BATTLE_FORMAT, log_level=40, open_timeout=None,
                     start_listening=False)
    try:
        env.battle1 = battle
        side = getattr(battle, "player_role", "p1") or "p1"
        inject_maps(env.agent1, our_f, opp_f, our_side=side, our_team=our_team, opp_team=opp_team,
                    protect=(float(protect[0] or 0.0), float(protect[1] or 0.0)))
        return env.embed_battle(battle)
    finally:
        safe_close(env)


def _live_run(only) -> list:
    order = ["S1", "S2", "S3", "S4", "S5"]
    if not only:
        return order
    wanted = [s for s in order if s in only or s.lower() in only]
    if not wanted:
        return []
    return order[: order.index(wanted[-1]) + 1]      # стадии накопительные


# ------------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description="Признаки одинаковы на всех стадиях")
    ap.add_argument("--no-live", action="store_true", help="не запускать стадии с сервером")
    ap.add_argument("--live", action="store_true", help="обязательно запустить live-стадии")
    ap.add_argument("--only", default="", help="адресный прогон: 1,2,3,S1,S2,S3,S4,S5")
    args = ap.parse_args()
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    def want(tag: str) -> bool:
        return not only or tag.upper() in only

    fmt = os.environ.get("PYBOT_TEST_FORMAT", "gen9randombattle")
    fast = os.environ.get("PYBOT_TEST_FAST", "") not in ("", "0")
    log = print

    print(f"N_FEATURES={N_FEATURES}, формат live-боёв={fmt}, быстрый режим={fast}"
          + (f", только: {sorted(only)}" if only else ""))
    if want("1"):
        part1_layout()
        print("-" * 78)
    if want("2"):
        part2_paths()
        print("-" * 78)
    with tempfile.TemporaryDirectory() as td:
        if want("3"):
            part3_norm(td)
        live_ok = False
        need_live = any(want(t) for t in ("S1", "S2", "S3", "S4", "S5")) or (not only and not args.no_live)
        if need_live and not args.no_live:
            print("-" * 78)
            print("ЧАСТЬ 4. Живой прогон тестовой модели по всем стадиям")
            print("-" * 78)
            live_ok = ensure_server(log)
            if live_ok:
                try:
                    live_stages(fmt, fast, td, log, _live_run(only))
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    check("live-стадии прошли без исключений", False, f"{type(e).__name__}: {e}")
            else:
                SKIPPED.append("live-стадии (нет Showdown-сервера)")

    print("=" * 78)
    print("ИТОГ: стадия | obs_dim | нормализация | max|Δ| с эталоном | хеш obs")
    print("-" * 78)
    for row in ROWS:
        print(f"  {row[0]:<34} {str(row[1]):>4} {str(row[2]):<24} {row[3]:>10.3e}  {row[4]}")
    print("=" * 78)
    if SKIPPED:
        print("ПРОПУЩЕНО: " + "; ".join(SKIPPED))
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)})."
          + (f" Пропущено: {len(SKIPPED)}" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
