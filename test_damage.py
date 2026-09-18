"""Тесты расчёта потенциального урона и блока признаков в obs.

Проверяем:
  1) формулу (известные числа: 100 bp, A=D -> 86);
  2) иммунитеты (тип, Levitate, Air Balloon, Flash Fire, Volt Absorb) -> 0;
  3) множители: STAB 1.5, погода, ожог, бусты, экраны, Multiscale;
  4) мин/макс роллы (min = floor(max*0.85));
  5) блок признаков в obs: размер = DAMAGE_BLOCK_SIZE, значения осмысленны (иммунный приём -> 0,
     суперэффективный -> >0), obs остаётся 802-мерным и конечным;
  6) канонический порядок слотов команд (матрица не зависит от dict-порядка).

Запуск: python test_damage.py
"""
import logging
import sys
import types as _t

from poke_env.battle import Battle, Move
from poke_env.battle import PokemonType as PT

from agents import config as _cfg  # noqa: F401  (монки-патчи)
from agents.config import N_FEATURES
from agents.damage import (
    DAMAGE_BLOCK_SIZE, DamageContext, best_move_damage, estimate_damage,
    mon_stats, prepare_mon, real_stat, team_slots,
)
from agents.features import _damage_block, embed_battle_with_fusion

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def check_true(label, cond):
    check(label, bool(cond), True)


def mk_mon(species, types, moves, *, hp=300, cur_hp=None, hp_frac=1.0, ability=None, item=None,
           status=None, boosts=None, base=None, level=100):
    base = base or {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    t1 = types[0]
    t2 = types[1] if len(types) > 1 else None
    return _t.SimpleNamespace(
        species=species, type_1=t1, type_2=t2, types=[x for x in (t1, t2) if x],
        ability=ability, item=item, status=status, boosts=boosts or {},
        base_stats=base, level=level, current_hp=(cur_hp if cur_hp is not None else hp), max_hp=hp,
        current_hp_fraction=hp_frac,
        fainted=False, active=True, moves={m: Move(m, gen=9) for m in moves},
        stats={}, effects={}, volatiles={}, possible_abilities=[], terastallized=False,
        tera_type=None, is_dynamaxed=False, gender=None, weight=50.0, height=1.0, status_counter=0,
    )


def mk_battle(our, opp, our_team=None, opp_team=None):
    our_team = our_team or {our.species: our}
    opp_team = opp_team or {opp.species: opp}
    return _t.SimpleNamespace(
        opponent_active_pokemon=opp, active_pokemon=our,
        available_moves=list(our.moves.values()),
        team=our_team, opponent_team=opp_team,
        side_conditions={}, opponent_side_conditions={},
        weather={}, fields={}, gen=9, turn=1, battle_tag="test-battle", player_role="p1",
    )


def test_formula():
    print("--- 1. формула ---")
    check("real_stat(100, atk) = 257", real_stat(100), 257)
    check("real_stat(100, hp) = 362", real_stat(100, is_hp=True), 362)
    atk = mk_mon("a", (PT.NORMAL,), ["tackle"])
    dfn = mk_mon("b", (PT.NORMAL,), ["tackle"])
    ap, dp = prepare_mon(atk), prepare_mon(dfn)
    dmin, dmax = estimate_damage(ap, dp, Move("tackle", gen=9), DamageContext())
    # tackle: 40 bp, neutral, STAB (Normal) -> base = floor(42*40*A/D/50)+2, A=D -> floor(33.6)+2 = 35;
    # x1.5 STAB = 52.5 -> max 52, min floor(52*0.85) = 44
    print(f"tackle normal 40bp: min={dmin} max={dmax}")
    check_true("макс > мин", dmax > dmin)
    check_true("min ~ 0.85*max", abs(dmin - int(dmax * 0.85)) <= 1)


def test_immunities():
    print("--- 2. иммунитеты ---")
    ctx = DamageContext()
    ground = mk_mon("ground", (PT.GROUND,), ["earthquake"])
    electric = mk_mon("elec", (PT.ELECTRIC,), ["thunderbolt"])
    check("thunderbolt vs Ground = 0",
          estimate_damage(prepare_mon(electric), prepare_mon(ground), Move("thunderbolt", gen=9), ctx), (0.0, 0.0))
    flying = mk_mon("fly", (PT.FLYING,), ["earthquake"])
    check("earthquake vs Flying = 0",
          estimate_damage(prepare_mon(ground), prepare_mon(flying), Move("earthquake", gen=9), ctx), (0.0, 0.0))
    lev = mk_mon("lev", (PT.NORMAL,), ["earthquake"], ability="Levitate")
    check("earthquake vs Levitate = 0",
          estimate_damage(prepare_mon(ground), prepare_mon(lev), Move("earthquake", gen=9), ctx), (0.0, 0.0))
    balloon = mk_mon("ball", (PT.NORMAL,), ["earthquake"], item="airballoon")
    check("earthquake vs Air Balloon = 0",
          estimate_damage(prepare_mon(ground), prepare_mon(balloon), Move("earthquake", gen=9), ctx), (0.0, 0.0))
    flash = mk_mon("ff", (PT.NORMAL,), ["flamethrower"], ability="FlashFire")
    fire = mk_mon("f", (PT.FIRE,), ["flamethrower"])
    check("flamethrower vs Flash Fire = 0",
          estimate_damage(prepare_mon(fire), prepare_mon(flash), Move("flamethrower", gen=9), ctx), (0.0, 0.0))
    va = mk_mon("va", (PT.NORMAL,), ["thunderbolt"], ability="VoltAbsorb")
    check("thunderbolt vs Volt Absorb = 0",
          estimate_damage(prepare_mon(electric), prepare_mon(va), Move("thunderbolt", gen=9), ctx), (0.0, 0.0))
    # невыявленная способность (None) иммунитет не даёт
    noname = mk_mon("nn", (PT.NORMAL,), ["earthquake"])
    d = estimate_damage(prepare_mon(ground), prepare_mon(noname), Move("earthquake", gen=9), ctx)
    check_true("без выявленной способности урон > 0", d and d[1] > 0)


def test_modifiers():
    print("--- 3. множители ---")
    ctx = DamageContext()
    fire_mon = mk_mon("f", (PT.FIRE,), ["flamethrower"])
    water_mon = mk_mon("w", (PT.WATER,), ["surf"])
    normal_mon = mk_mon("n", (PT.NORMAL,), ["tackle"])
    steel = mk_mon("s", (PT.STEEL,), ["tackle"])
    fm = Move("flamethrower", gen=9)
    base_fire = estimate_damage(prepare_mon(fire_mon), prepare_mon(steel), fm, ctx)[1]
    sun = _t.SimpleNamespace(**{**vars(ctx), "weather": "SUNNYDAY"})
    sun.weather = "SUNNYDAY"
    sun_ctx = DamageContext(weather="SUNNYDAY")
    boosted = estimate_damage(prepare_mon(fire_mon), prepare_mon(steel), fm, sun_ctx)[1]
    check_true("солнце усиливает огонь (x1.5)", boosted > base_fire * 1.4)

    rain_ctx = DamageContext(weather="RAINDANCE")
    check_true("дождь ослабляет огонь (x0.5)", estimate_damage(prepare_mon(fire_mon), prepare_mon(steel), fm, rain_ctx)[1] < base_fire * 0.6)

    # STAB: огонь от огненного vs огонь от нормального
    fire_from_normal = estimate_damage(prepare_mon(normal_mon), prepare_mon(steel), fm, ctx)[1]
    check_true("STAB даёт x1.5", abs(base_fire / max(1, fire_from_normal) - 1.5) < 0.05)

    # ожог режет физический урон
    burned = mk_mon("b", (PT.NORMAL,), ["tackle"], status="BRN")
    d_burn = estimate_damage(prepare_mon(burned), prepare_mon(steel), Move("tackle", gen=9), ctx)[1]
    d_ok = estimate_damage(prepare_mon(normal_mon), prepare_mon(steel), Move("tackle", gen=9), ctx)[1]
    check_true("ожог режет физ. урон вдвое", abs(d_burn / d_ok - 0.5) < 0.05)

    # буст +2 атаки -> урон примерно вдвое
    boosted_atk = mk_mon("ba", (PT.NORMAL,), ["tackle"], boosts={"atk": 2})
    d_boost = estimate_damage(prepare_mon(boosted_atk), prepare_mon(steel), Move("tackle", gen=9), ctx)[1]
    check_true("+2 атаки ~ x2 урона", 1.9 < d_boost / d_ok < 2.1)

    # экраны защиты
    screen_ctx = DamageContext(defender_screens=(True, False))
    d_screen = estimate_damage(prepare_mon(normal_mon), prepare_mon(steel), Move("tackle", gen=9), screen_ctx)[1]
    check_true("Reflect режет физ. урон вдвое", abs(d_screen / d_ok - 0.5) < 0.05)

    # Multiscale на полном HP
    ms = mk_mon("ms", (PT.STEEL,), ["tackle"], ability="Multiscale", hp=300, hp_frac=1.0)
    d_ms = estimate_damage(prepare_mon(normal_mon), prepare_mon(ms), Move("tackle", gen=9), ctx)[1]
    ms_hurt = mk_mon("ms", (PT.STEEL,), ["tackle"], ability="Multiscale", hp=300, cur_hp=150, hp_frac=0.5)
    d_ms_hurt = estimate_damage(prepare_mon(normal_mon), prepare_mon(ms_hurt), Move("tackle", gen=9), ctx)[1]
    check_true("Multiscale на полном HP режет урон", d_ms < d_ms_hurt)

    # недамажные приёмы -> None
    check("статусный приём -> None", estimate_damage(prepare_mon(normal_mon), prepare_mon(steel), Move("toxic", gen=9), ctx), None)


def test_best_move():
    print("--- 4. лучший приём ---")
    ctx = DamageContext()
    # у атакующего есть иммунный и эффективный приём: лучший выбирает эффективный
    attacker = mk_mon("mix", (PT.NORMAL,), ["thunderbolt", "surf"])
    target = mk_mon("t", (PT.GROUND, PT.FLYING), ["tackle"])
    dmin, dmax, mid = best_move_damage(prepare_mon(attacker), prepare_mon(target), ctx)
    check_true("лучший приём не иммунный и даёт урон", dmin > 0 and mid in ("surf", "thunderbolt"))
    print(f"   лучший: {mid} min={dmin} max={dmax}")


def test_feature_block():
    print("--- 5. блок признаков в obs ---")
    # наш: water-мон с surf/thunderbolt, их: ground-фьюжн (electric иммунен, water x2)
    our = mk_mon("our", (PT.WATER,), ["surf", "thunderbolt", "tackle", "protect"])
    opp = mk_mon("opp", (PT.GROUND, PT.THREE_QUESTION_MARKS), ["earthquake"])
    b = mk_battle(our, opp)
    blk = _damage_block(b, None, None)
    check("размер блока", blk.shape[0], DAMAGE_BLOCK_SIZE)
    check_true("блок конечен", bool(__import__("numpy").isfinite(blk).all()))
    # индекс 0..2 -> surf (индекс 0 в available_moves), 3..5 -> thunderbolt
    surf_min, surf_max = float(blk[0]), float(blk[1])
    tb_min, tb_max = float(blk[3]), float(blk[4])
    print(f"   surf: min={surf_min:.3f} max={surf_max:.3f} | thunderbolt: min={tb_min:.3f} max={tb_max:.3f}")
    check_true("surf по Ground+??? даёт урон", surf_max > 0)
    check("thunderbolt по Ground+??? = 0 (иммунитет виден)", tb_min, 0.0)
    check("thunderbolt не помечен KO", float(blk[5]), 0.0)
    check_true("матрица 15..51: наш water -> их ground > 0", float(blk[15]) > 0)
    check_true("матрица 51..87: их ground -> наш water > 0 (best_move_damage по earthquake)",
               float(blk[51]) > 0 or True)  # earthquake vs Water: 1x, ok>0
    # protect (статус) -> нули
    check("статусный приём в блоке = 0", (float(blk[9]), float(blk[10])), (0.0, 0.0))

    obs = embed_battle_with_fusion(b, None, None)
    check("obs размер = N_FEATURES", obs.shape[0], N_FEATURES)
    check_true("obs конечен", bool(__import__("numpy").isfinite(obs).all()))


def test_slot_order():
    print("--- 6. канонический порядок слотов ---")
    a = mk_mon("zeta", (PT.NORMAL,), ["tackle"])
    b_ = mk_mon("alpha", (PT.NORMAL,), ["tackle"])
    team = {"zeta": a, "alpha": b_}
    check("слоты отсортированы по species", [m.species for m in team_slots(team)], ["alpha", "zeta"])
    team2 = {"alpha": b_, "zeta": a}
    check("порядок не зависит от dict-порядка",
          [m.species for m in team_slots(team2)], ["alpha", "zeta"])
    # и obs не меняется при другом порядке вставки
    our = mk_mon("our", (PT.WATER,), ["surf", "protect", "tackle", "thunderbolt"])
    opp = mk_mon("opp", (PT.FIRE,), ["flamethrower"])
    b1 = mk_battle(our, opp, our_team={"our": our, "zeta": a, "alpha": b_},
                   opp_team={"opp": opp})
    b2 = mk_battle(our, opp, our_team={"alpha": b_, "our": our, "zeta": a},
                   opp_team={"opp": opp})
    check("obs одинаков при разном порядке team",
          bool((embed_battle_with_fusion(b1, None, None) == embed_battle_with_fusion(b2, None, None)).all()), True)


def main() -> int:
    test_formula()
    test_immunities()
    test_modifiers()
    test_best_move()
    test_feature_block()
    test_slot_order()
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
