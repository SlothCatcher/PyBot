"""Регресс-тест: расчёт типовой эффективности не должен маскировать иммунитет.

Причина: чарт gen9 знает только 18 типов, а `PokemonType.damage_multiplier` кидает
KeyError, если тип защиты в чарте отсутствует (???/STELLAR у фьюжнов). Старый монки-патч
в `agents/config.py` ловил KeyError и возвращал 1.0 на весь расчёт — иммунитет
Electric vs Ground (0.0) превращался в "нейтрально", и модель спамила бесполезный приём.

Запуск (нужен poke-env):
    python test_type_multiplier_fix.py
"""
import sys

from poke_env.battle.pokemon_type import PokemonType as PT
from poke_env.data import GenData

from agents.type_utils import damage_multiplier_safe, damage_multiplier_safe_ex

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


# ---------------------------------------------------------------------------
# E2E на мок-бое: _move_wasted_flag + moves_dmg_multiplier внутри obs
# ---------------------------------------------------------------------------
def _mk_mon(name, t1, t2, move_ids):
    import types as _types
    from poke_env.battle import Move
    return _types.SimpleNamespace(
        type_1=t1, type_2=t2, ability=None, item=None, status=None,
        boosts={}, species=name, current_hp_fraction=1.0, current_hp=300, max_hp=300,
        fainted=False, active=True, level=100, types=[x for x in (t1, t2) if x],
        base_stats={"hp": 80, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        stats={"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
        moves={m: Move(m, gen=9) for m in move_ids}, effects={}, volatiles={},
        possible_abilities=[], terastallized=False, tera_type=None, is_dynamaxed=False,
        gender=None, weight=50.0, height=1.0, status_counter=0,
    )


def _fake_battle(move_ids, opp_types, our_types=(PT.FIRE, None)):
    import types as _types
    from poke_env.battle import Move
    opp = _mk_mon("fakemon", opp_types[0], opp_types[1], move_ids)
    our = _mk_mon("usmon", our_types[0], our_types[1], move_ids)
    b = _types.SimpleNamespace(
        opponent_active_pokemon=opp, active_pokemon=our,
        available_moves=[Move(m, gen=9) for m in move_ids],
        side_conditions={}, opponent_side_conditions={},
        weather={}, fields={}, team={"usmon": our}, opponent_team={"fakemon": opp},
        gen=9, turn=1, battle_tag="test", player_role="p1",
    )
    return b, [Move(m, gen=9) for m in move_ids]


def test_wasted_flag_and_obs():
    from agents.features import _move_wasted_flag, embed_battle_with_fusion

    moves = ["thunderbolt", "flamethrower", "earthquake", "protect"]
    # thunderbolt по земляному фьюжну с нераспознанным вторым типом = 0 урона
    for label, opp_types, want_wasted, want_mult in [
        ("GROUND", (PT.GROUND, None), 1.0, 0.0),
        ("GROUND + ???", (PT.GROUND, PT.THREE_QUESTION_MARKS), 1.0, 0.0),
        ("GROUND + STELLAR", (PT.GROUND, PT.STELLAR), 1.0, 0.0),
        ("WATER + ??? (контроль)", (PT.WATER, PT.THREE_QUESTION_MARKS), 0.0, 2.0),
    ]:
        b, mv = _fake_battle(moves, opp_types)
        flag = _move_wasted_flag(mv[0], b)
        obs = embed_battle_with_fusion(b, None, None)
        mult = float(obs[4])  # moves_dmg_multiplier[0]
        check(f"E2E thunderbolt vs {label}: wasted", flag, want_wasted)
        check(f"E2E thunderbolt vs {label}: obs dmg_mult", mult, want_mult)

    # earthquake по летающему с неизвестным вторым типом
    b, mv = _fake_battle(["earthquake", "protect", "thunderbolt", "flamethrower"],
                         (PT.FLYING, PT.THREE_QUESTION_MARKS))
    check("E2E earthquake vs FLYING+???: wasted", _move_wasted_flag(mv[0], b), 1.0)
    # obs не должен поменять размер
    from agents.config import N_FEATURES
    check("E2E obs shape == N_FEATURES", embed_battle_with_fusion(b, None, None).shape[0], N_FEATURES)


def main() -> int:
    chart = GenData.from_gen(9).type_chart
    print(f"gen9 chart: {len(chart)} типов, STELLAR={'STELLAR' in chart}, "
          f"THREE_QUESTION_MARKS={'THREE_QUESTION_MARKS' in chart}")
    print("-" * 70)

    # 1. Диагноз: ванильный poke-env кидает KeyError на неизвестном ВТОРОМ типе.
    #    ВАЖНО: проверяем ДО импорта agents.config, иначе монки-патч уже подменён.

    try:
        PT.ELECTRIC.damage_multiplier(PT.GROUND, PT.THREE_QUESTION_MARKS, type_chart=chart)
        print("!! ванильный вызов не кинул KeyError — poke-env изменился, проверь версию")
        vanilla_keyerror = False
    except KeyError as e:
        print(f"OK   подтверждено: ванильный damage_multiplier кидает KeyError({e}) для GROUND + ???")
        vanilla_keyerror = True

    # 2. Применяем монки-патчи и проверяем, что больше нет 1.0 на этот случай
    from agents import config as _cfg  # noqa: F401
    check("патч: ELECTRIC vs GROUND+??? = иммунитет (было 1.0)",
          PT.ELECTRIC.damage_multiplier(PT.GROUND, PT.THREE_QUESTION_MARKS, type_chart=chart), 0.0)
    check("патч: ELECTRIC vs GROUND = 0.0",
          PT.ELECTRIC.damage_multiplier(PT.GROUND, None, type_chart=chart), 0.0)
    check("патч: ELECTRIC vs GROUND+STELLAR = 0.0",
          PT.ELECTRIC.damage_multiplier(PT.GROUND, PT.STELLAR, type_chart=chart), 0.0)
    check("патч: ELECTRIC vs WATER = 2.0",
          PT.ELECTRIC.damage_multiplier(PT.WATER, None, type_chart=chart), 2.0)
    check("патч: FIRE vs GRASS+STEEL = 4.0",
          PT.FIRE.damage_multiplier(PT.GRASS, PT.STEEL, type_chart=chart), 4.0)
    check("патч: NORMAL vs GHOST = 0.0",
          PT.NORMAL.damage_multiplier(PT.GHOST, None, type_chart=chart), 0.0)

    # 3. Хелпер: покомпонентность
    check("safe: ELECTRIC vs (GROUND, None)", damage_multiplier_safe(PT.ELECTRIC, PT.GROUND), 0.0)
    check("safe: ELECTRIC vs (GROUND, ???)", damage_multiplier_safe(PT.ELECTRIC, PT.GROUND, PT.THREE_QUESTION_MARKS), 0.0)
    check("safe: ELECTRIC vs (GROUND, STELLAR)", damage_multiplier_safe(PT.ELECTRIC, PT.GROUND, PT.STELLAR), 0.0)
    check("safe: ELECTRIC vs (??? , None) остаётся нейтралом", damage_multiplier_safe(PT.ELECTRIC, PT.THREE_QUESTION_MARKS), 1.0)
    check("safe: ELECTRIC vs (WATER, ???) = 2.0 (неизвестный компонент не ломает известный)",
          damage_multiplier_safe(PT.ELECTRIC, PT.WATER, PT.THREE_QUESTION_MARKS), 2.0)
    check("safe: NORMAL vs (GHOST, ???) = 0.0", damage_multiplier_safe(PT.NORMAL, PT.GHOST, PT.THREE_QUESTION_MARKS), 0.0)
    check("safe: FIRE vs (GRASS, STEEL) = 4.0", damage_multiplier_safe(PT.FIRE, PT.GRASS, PT.STEEL), 4.0)
    check("safe: Fire vs (Water, ???) = 0.5 (неизвестный компонент не ломает известный)",
          damage_multiplier_safe(PT.FIRE, PT.WATER, PT.THREE_QUESTION_MARKS), 0.5)

    # 4. Флаг unknown
    mult, unknown = damage_multiplier_safe_ex(PT.ELECTRIC, PT.GROUND, PT.THREE_QUESTION_MARKS)
    check("unknown-флаг для (GROUND, ???)", (mult, unknown), (0.0, True))
    mult, unknown = damage_multiplier_safe_ex(PT.ELECTRIC, PT.GROUND)
    check("unknown-флаг для (GROUND)", (mult, unknown), (0.0, False))
    mult, unknown = damage_multiplier_safe_ex(PT.ELECTRIC, PT.THREE_QUESTION_MARKS)
    check("unknown-флаг для (???, None)", (mult, unknown), (1.0, True))

    # 5. None-типы не роняют расчёт
    check("safe: move_type=None", damage_multiplier_safe(None, PT.GROUND), 1.0)
    check("safe: def_type_1=None", damage_multiplier_safe(PT.ELECTRIC, None), 1.0)

    # 6. E2E на мок-бое: флаг wasted и признак moves_dmg_multiplier в obs
    test_wasted_flag_and_obs()

    print("-" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
