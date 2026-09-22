#!/usr/bin/env python3
"""Слоты приёмов в признаках = раскладка действий poke-env (6..9 и 22..25).

Повод (найдено на ревью, воспроизведено здесь на живом poke-env): признаки приёмов
строились циклом `enumerate(battle.available_moves)`, а действия 6..9 poke-env нумерует
по ДРУГОМУ списку:

    # poke_env/environment/singles_env.py
    known_moves = list(battle.active_pokemon.moves.values())[:4]      # ВСЕ известные, 4 слота
    mvs = battle.available_moves if (len(available_moves) == 1
                                     and available_moves[0].id not in known_ids) else known_moves

`battle.available_moves` короче и без «дыр»: сервер не присылает выключенные приёмы
(PP=0, Disable, Taunt, Choice-lock, Heal Block, Torment). При выключенном приёме №2:
known = [A, B(disabled), C, D], available = [A, C, D], маска разрешает действия 6, 8, 9,
а признаки получали слот 1 = C, слот 2 = D, слот 3 = дефолт. То есть действие 8 (C) видело
характеристики D, действие 9 (D) — дефолтные -1/1, а «родные» признаки B никто не видел.

Что проверяем (без сервера: боевые объекты poke-env собираются из сообщений и request):
  A) `move_slots_for_action` совпадает с приёмом, который реально исполняет
     `SinglesEnv.action_to_order` для КАЖДОГО легального по маске действия 6..9/22..25;
  B) признаки лежат в тех же слотах: obs[0:4] (base_power) и obs[8:12] (wasted) описывают
     именно приёмы слотов; выключенный приём остаётся в своём слоте, не сдвигая соседей;
  C) выключенный приём помечается wasted=1.0; при всех доступных приёмах obs 1:1 как раньше;
  D) блок признаков урона (damage block) нумерует приёмы теми же слотами;
  E) редкий случай poke-env (доступен ровно один приём, которого нет среди известных) — слот 0;
  F) недораскрытый мувсет/нет активного покемона — не падаем;
  G) env._moves_for_action (бухгалтерия награды) и features.move_slots_for_action — одна логика.

Запуск: PYTHONPATH=. python test_move_slots.py
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import config as _cfg  # noqa: E402,F401  (монки-патчи poke-env)
from agents import features as F  # noqa: E402
from agents.features import embed_battle_with_fusion, move_slots_for_action  # noqa: E402
from poke_env.battle import Battle  # noqa: E402
from poke_env.environment.singles_env import SinglesEnv  # noqa: E402

OK = 0
FAIL: list = []

# Смещения в obs (голова начинается с блока приёмов, см. np.concatenate в features.py)
OFF_BASE_POWER = 0      # moves_base_power: 4 значения по base_power/100
OFF_DMG_MULT = 4        # moves_dmg_multiplier
OFF_WASTED = 8          # moves_wasted
OFF_ACCURACY = 12
OFF_PP = 16

MOVES = ("hydropump", "earthquake", "icebeam", "protect")   # bp/100 = 1.1 / 1.0 / 0.9 / 0.0


def check(name, cond, extra=""):
    global OK
    if cond:
        OK += 1
        print(f"OK   {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL.append(name)
        print(f"FAIL {name}" + (f": {extra}" if extra else ""))


def make_battle(our_moves=MOVES, disabled=(), can_tera=None, bench=(), opp="skarmory, L50, F"):
    """Настоящий poke-env Battle: сообщения + request (как в живом бою)."""
    b = Battle(battle_tag="battle-gen9randombattle-1", username="Me",
               logger=logging.getLogger("quiet"), gen=9)
    for msg in [["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""],
                ["", "start"],
                ["", "switch", "p1a: Aqua", "swampert, L50, M", "362/362"],
                ["", "switch", "p2a: Skarm", opp, "300/300"]]:
        b.parse_message(msg)
    pokemon = [{"ident": "p1: Aqua", "details": "swampert, L50, M", "condition": "362/362",
                "active": True,
                "stats": {"atk": 257, "def": 257, "spa": 257, "spd": 257, "spe": 257},
                "moves": list(our_moves), "baseAbility": "torrent", "item": "leftovers",
                "pokeball": "pokeball", "ability": "torrent"}]
    # имена/виды разные: poke-env заводит мон в team по ident, одинаковые виды путаются
    bench_mons = [("Rock", "garchomp, L50, F", "dragonclaw", "roughskin"),
                  ("Lava", "heatran, L50, M", "magmastorm", "flashfire")]
    for name, details, mv, ability in bench_mons[:len(bench)]:
        pokemon.append({"ident": f"p1: {name}", "details": details, "condition": "300/300",
                        "active": False,
                        "stats": {"atk": 250, "def": 250, "spa": 250, "spd": 250, "spe": 250},
                        "moves": [mv], "baseAbility": ability, "item": "leftovers",
                        "pokeball": "pokeball", "ability": ability})
    active = {"moves": [{"move": m.capitalize(), "id": m, "pp": 16, "maxpp": 16,
                         "target": "normal", "disabled": m in disabled} for m in our_moves]}
    if can_tera:
        active["canTerastallize"] = can_tera
    b.parse_request({"active": [active],
                     "side": {"id": "p1", "name": "Me", "pokemon": pokemon}, "rqid": 1})
    return b


def embed(b):
    return embed_battle_with_fusion(b, None, None, our_team_fusions={}, opp_team_fusions={})


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from agents.config import N_FEATURES

    # ---------------------------------------------------------------- A) слоты == действия ---
    print("=" * 78)
    print("A. Слот приёма совпадает с приёмом, который исполняет action_to_order")
    print("=" * 78)
    cases = (("все приёмы доступны", {}),
             ("earthquake disabled", {"disabled": ("earthquake",)}),
             ("hydropump disabled", {"disabled": ("hydropump",)}),
             ("icebeam disabled", {"disabled": ("icebeam",)}),
             ("protect disabled", {"disabled": ("protect",)}),
             ("disabled + тера доступна", {"disabled": ("icebeam",), "can_tera": "Water"}),
             ("disabled + скамейка", {"disabled": ("earthquake",), "bench": ("a", "b")}))
    for label, kwargs in cases:
        b = make_battle(**kwargs)
        slots = [m.id for m in move_slots_for_action(b)]
        mask = SinglesEnv.get_action_mask(b)
        move_actions = [a for a in range(6, 10) if mask[a]]
        tera_actions = [a for a in range(22, 26) if mask[a]]
        check(f"A[{label}]: слоты = known_moves активного",
              slots == [m.id for m in list(b.active_pokemon.moves.values())[:4]], str(slots))
        bad = []
        for a in move_actions + tera_actions:
            order = SinglesEnv.action_to_order(np.int64(a), b, strict=False)
            # для move-ордера str(order) = "/choose move <id> [terastallize]" — берём id из объекта
            inner = getattr(order, "order", None)
            executed = str(getattr(inner, "id", "") or str(order).split()[-1]).lower()
            want = slots[(a - 6) % 4]
            if executed != want:
                bad.append((a, executed, want))
        check(f"A[{label}]: для всех легальных действий приём == слот",
              not bad and bool(move_actions), f"действия={move_actions}+{tera_actions} {bad}")
        disabled_slots = [i for i, m in enumerate(slots)
                          if m not in [x.id for x in b.available_moves]]
        if disabled_slots:
            check(f"A[{label}]: выключенные слоты {disabled_slots} запрещены маской",
                  all((6 + i) not in move_actions for i in disabled_slots),
                  f"move_actions={move_actions}")

    # ---------------------------------------------------- B) признаки в тех же слотах ---
    print("-" * 78)
    print("B. base_power/dmg_multiplier/wasted слота i описывают приём слота i")
    print("-" * 78)
    # против heatran (Fire/Steel): earthquake сам по себе НЕ wasted (2x по Fire) — значит
    # единственная причина wasted=1 в слоте 1 — то, что приём выключен сервером.
    b = make_battle(disabled=("earthquake",), opp="heatran, L50, M")
    obs = embed(b)
    slots = [m.id for m in move_slots_for_action(b)]
    check("B: размерность obs", obs.shape == (N_FEATURES,), str(obs.shape))
    check("B: слоты = [hydropump, earthquake, icebeam, protect]",
          slots == list(MOVES), str(slots))
    check("B: base_power в слотах = [1.1, 1.0, 0.9, 0.0] (earthquake не сдвинулся)",
          np.allclose(obs[OFF_BASE_POWER:OFF_BASE_POWER + 4], [1.1, 1.0, 0.9, 0.0]),
          str(np.round(obs[OFF_BASE_POWER:OFF_BASE_POWER + 4], 3)))
    check("B: старый путь (available_moves) дал бы [1.1, 0.9, 0.0, -1.0] — рассинхрон",
          not np.allclose(obs[OFF_BASE_POWER:OFF_BASE_POWER + 4], [1.1, 0.9, 0.0, -1.0]))
    # heatran Fire/Steel: Water 2x, Ground 2x*2x = 4x (знаменитая 4x-слабость), Ice 0.5*0.5,
    # Normal 1*0.5 — ожидание фиксируем числом именно для проверки ПОРЯДКА слотов
    check("B: множители типов по слотам = [2.0, 4.0, 0.25, 0.5]",
          np.allclose(obs[OFF_DMG_MULT:OFF_DMG_MULT + 4], [2.0, 4.0, 0.25, 0.5]),
          str(np.round(obs[OFF_DMG_MULT:OFF_DMG_MULT + 4], 3)))
    from agents.features import _move_wasted_flag
    check("B: сам по себе earthquake против heatran НЕ wasted",
          float(_move_wasted_flag(b.active_pokemon.moves["earthquake"], b)) == 0.0)
    check("B: но в obs слот 1 (выключенный приём) = wasted 1.0",
          np.allclose(obs[OFF_WASTED:OFF_WASTED + 4], [0.0, 1.0, 0.0, 0.0]),
          str(np.round(obs[OFF_WASTED:OFF_WASTED + 4], 3)))
    check("B: pp_frac всех слотов = 1 (PP не теряются при выключенном приёме)",
          np.allclose(obs[OFF_PP:OFF_PP + 4], [1.0, 1.0, 1.0, 1.0]),
          str(np.round(obs[OFF_PP:OFF_PP + 4], 3)))

    # соседний случай: выключен ПОСЛЕДНИЙ приём — предыдущие слоты не должны поехать
    b_last = make_battle(disabled=("protect",), opp="heatran, L50, M")
    obs3 = embed(b_last)
    check("B: protect disabled -> base_power = [1.1, 1.0, 0.9, 0.0] (без сдвига)",
          np.allclose(obs3[OFF_BASE_POWER:OFF_BASE_POWER + 4], [1.1, 1.0, 0.9, 0.0]),
          str(np.round(obs3[OFF_BASE_POWER:OFF_BASE_POWER + 4], 3)))
    check("B: protect сам по себе не wasted, но выключен -> wasted слота 3 = 1.0",
          float(_move_wasted_flag(b_last.active_pokemon.moves["protect"], b_last)) == 0.0
          and abs(float(obs3[OFF_WASTED + 3]) - 1.0) < 1e-6,
          str(np.round(obs3[OFF_WASTED:OFF_WASTED + 4], 3)))
    check("B: доступные слоты при этом остались не-wasted",
          np.allclose(obs3[OFF_WASTED:OFF_WASTED + 3], [0.0, 0.0, 0.0]),
          str(np.round(obs3[OFF_WASTED:OFF_WASTED + 4], 3)))

    # ------------------------------------------------------- C) регресс: всё доступно ---
    print("-" * 78)
    print("C. Когда доступны все приёмы, obs совпадает с прежним поведением")
    print("-" * 78)
    b_full = make_battle()
    avail_ids = [m.id for m in b_full.available_moves]
    check("C: все 4 приёма доступны", avail_ids == list(MOVES), str(avail_ids))
    obs_full = embed(b_full)
    check("C: base_power как у available_moves-порядка (списки совпадают)",
          np.allclose(obs_full[OFF_BASE_POWER:OFF_BASE_POWER + 4], [1.1, 1.0, 0.9, 0.0]),
          str(np.round(obs_full[OFF_BASE_POWER:OFF_BASE_POWER + 4], 3)))
    check("C: wasted = [0, 1, 0, 0] (earthquake — иммун по Flying, как и раньше)",
          np.allclose(obs_full[OFF_WASTED:OFF_WASTED + 4], [0.0, 1.0, 0.0, 0.0]),
          str(np.round(obs_full[OFF_WASTED:OFF_WASTED + 4], 3)))

    # ------------------------------------------------------------- D) damage block ---
    print("-" * 78)
    print("D. Блок урона нумерует приёмы теми же слотами")
    print("-" * 78)
    b = make_battle(disabled=("earthquake",))
    seen = []
    real_estimate = F.estimate_damage

    def spy(attacker, defender, move, ctx, **kw):
        seen.append(getattr(move, "id", None))
        return real_estimate(attacker, defender, move, ctx, **kw)

    F.estimate_damage = spy
    try:
        F._damage_block(b, None, None)
    finally:
        F.estimate_damage = real_estimate
    check("D: урон считается по слотам действий (known_moves), а не по available_moves",
          seen[:4] == list(MOVES), f"seen={seen[:4]}")

    # ------------------------------------------------------------- E) редкий случай ---
    print("-" * 78)
    print("E. Редкий случай poke-env: единственный доступный приём, которого нет среди известных")
    print("=" * 78)

    class _Mon:
        def __init__(self, ids):
            self.moves = {i: type("M", (), {"id": i})() for i in ids}

    class _Fake:
        def __init__(self, known, avail):
            self.active_pokemon = _Mon(known)
            self.available_moves = [type("M", (), {"id": a})() for a in avail]

    check("E: слот 0 = этот приём",
          [m.id for m in move_slots_for_action(_Fake(["surf"], ["protect"]))] == ["protect"])
    check("E: если приём известен — порядок known_moves",
          [m.id for m in move_slots_for_action(_Fake(["surf", "protect"], ["protect"]))]
          == ["surf", "protect"])
    check("E: два доступных приёма -> known_moves",
          [m.id for m in move_slots_for_action(_Fake(["surf", "protect"], ["surf", "protect"]))]
          == ["surf", "protect"])

    # ------------------------------------------------------------- F) хвосты и пустота ---
    print("-" * 78)
    print("F. Недораскрытый мувсет и отсутствие активного покемона")
    print("-" * 78)
    b2 = make_battle(our_moves=("hydropump", "icebeam"))
    check("F: два известных приёма -> два слота", len(move_slots_for_action(b2)) == 2)
    obs2 = embed(b2)
    check("F: obs не падает, размерность та же, конечные значения",
          obs2.shape == (N_FEATURES,) and bool(np.isfinite(obs2).all()))
    check("F: слот 2 (нет приёма) остался дефолтным: base_power = -1",
          abs(float(obs2[OFF_BASE_POWER + 2]) + 1.0) < 1e-6, str(obs2[OFF_BASE_POWER:OFF_BASE_POWER + 4]))

    class _Empty:
        active_pokemon = None
        available_moves = []
        opponent_active_pokemon = None
        team = {}
        opponent_team = {}
        side_conditions = {}
        opponent_side_conditions = {}
        weather = {}
        fields = {}
        gen = 9
        battle_tag = "battle-x"
    check("F: без активного покемона слотов нет", move_slots_for_action(_Empty()) == [])

    # ------------------------------------------------------------------- G) env ---
    print("-" * 78)
    print("G. Бухгалтерия награды (env._moves_for_action) и признаки — одна логика")
    print("-" * 78)
    from agents.env import ExampleEnv
    env = ExampleEnv(battle_format="gen9randombattle", log_level=40,
                     open_timeout=None, start_listening=False)
    for kwargs in ({}, {"disabled": ("earthquake",)}, {"disabled": ("protect",)}):
        b = make_battle(**kwargs)
        check(f"G{kwargs}: env и features дают один список",
              [m.id for m in env._moves_for_action(b)]
              == [m.id for m in move_slots_for_action(b)])

    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
