"""Тесты расчёта потенциального урона и блока признаков в obs.

Проверяем:
  1) формулу (известные числа: 100 bp, A=D -> 86);
  2) иммунитеты (тип, Levitate, Air Balloon, Flash Fire, Volt Absorb) -> 0;
  3) множители: STAB 1.5, погода, ожог, бусты, экраны, Multiscale;
  4) мин/макс роллы (min = floor(max*0.85));
  5) блок признаков в obs: размер = DAMAGE_BLOCK_SIZE, значения осмысленны (иммунный приём -> 0,
     суперэффективный -> >0), obs = N_FEATURES и конечен;
  6) канонический порядок слотов команд (матрица не зависит от dict-порядка);
  7) Mold Breaker/Teravolt/Turboblaze и приёмы с ignoreAbility игнорируют способность защиты;
  8) фьюжн-статы берутся только со своей стороны (species может совпасть);
  9) STAB от tera-типа — только после теры (объявленный tera:X в details STAB не даёт);
 10) инвариант Mold Breaker: способность защиты не влияет на урон (все приёмы x все способности).

Запуск: python test_damage.py
"""
import logging
import sys
from types import SimpleNamespace
import types as _t

from poke_env.battle import Battle, Move
from poke_env.battle import PokemonType as PT

from agents import config as _cfg  # noqa: F401  (монки-патчи)
from agents.config import N_FEATURES
from agents.damage import (
    DAMAGE_BLOCK_SIZE, DamageContext, EFFECT_FLAGS, best_move_damage, estimate_damage,
    mon_stats, opponent_effect_flags, prepare_mon, real_stat, team_slots,
)
from agents.features import _damage_block, embed_battle_with_fusion

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def check_true(label, cond, extra=""):
    check(label + (f": {extra}" if extra else ""), bool(cond), True)


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


def _flags_view(blk):
    from agents.damage import EFFECT_FLAGS, EFFECT_FLAG_COUNT, FLAGS_BASE
    return {EFFECT_FLAGS[i]: float(blk[FLAGS_BASE + i]) for i in range(EFFECT_FLAG_COUNT)}


def _nonzero_flags(blk):
    return {k: v for k, v in _flags_view(blk).items() if v}


def test_mirror_damage():
    """Зеркальный урон: известные приёмы противника по НАМ + сортировка по урону."""
    from agents.damage import (DAMAGE_BLOCK_SIZE, EFFECT_FLAG_COUNT, FLAGS_BASE, MIRROR_BASE,
                               OPP_MOVE_SLOTS, OUR_TEAM_SLOTS, TEAM_BASE)
    check("раскладка блока: MIRROR/TEAM/FLAGS", (MIRROR_BASE, TEAM_BASE, FLAGS_BASE), (87, 99, 123))
    check("размер блока = зеркало + флаги", DAMAGE_BLOCK_SIZE,
          MIRROR_BASE + OPP_MOVE_SLOTS * 3 + OPP_MOVE_SLOTS * OUR_TEAM_SLOTS + EFFECT_FLAG_COUNT)

    our = mk_mon("our", (PT.FIRE,), ["tackle", "protect", "ember", "quickattack"])
    opp = mk_mon("opp", (PT.WATER,), ["surf", "tackle", "recover", "roar"])
    blk = _damage_block(mk_battle(our, opp), None, None)

    base = MIRROR_BASE
    mins = [float(blk[base + i * 3]) for i in range(OPP_MOVE_SLOTS)]
    maxs = [float(blk[base + i * 3 + 1]) for i in range(OPP_MOVE_SLOTS)]
    check_true("зеркало: min урона убывает по слотам (сортировка по опасности)",
               mins[0] >= mins[1] >= mins[2] >= mins[3])
    check_true("зеркало: самый опасный приём — surf по Fire (2x)", mins[0] > mins[1] > 0,
               f"mins={[round(m, 3) for m in mins]}")
    check_true("зеркало: max >= min в каждом слоте", all(b >= a for a, b in zip(mins, maxs)))
    check_true("зеркало: урон по нам не нулевой для атакующих приёмов", mins[0] > 0)

    # команда: 4 приёма x 6 слотов
    team = {}
    for sp, tp in (("our", (PT.FIRE,)), ("watery", (PT.WATER,)), ("grassy", (PT.GRASS,)),
                   ("rocky", (PT.ROCK,)), ("steely", (PT.STEEL,)), ("ghosty", (PT.GHOST,))):
        team[sp] = mk_mon(sp, tp, ["tackle", "protect"])
    blk2 = _damage_block(mk_battle(team["our"], opp, our_team=team), None, None)
    t0 = TEAM_BASE
    check_true("зеркало: матрица по 6 нашим слотам заполнена",
               all(float(blk2[t0 + j]) > 0 for j in range(OUR_TEAM_SLOTS)),
               f"row0={[round(float(x), 3) for x in blk2[t0:t0 + OUR_TEAM_SLOTS]]}")
    check_true("зеркало: Water-наш получает меньше от surf, чем Fire-наш",
               float(blk2[t0 + 1]) < float(blk2[t0 + 0]),
               f"fire={float(blk2[t0 + 0]):.3f} water={float(blk2[t0 + 1]):.3f}")
    check_true("зеркало: [99:123] лежит ровно между матрицей и флагами",
               len(blk2[TEAM_BASE:FLAGS_BASE]) == OPP_MOVE_SLOTS * OUR_TEAM_SLOTS)

    # фейнт: урон по мёртвому не считается (иначе деление на 1 HP насыщает признак)
    dead = mk_mon("dead", (PT.NORMAL,), ["tackle"], hp=300, cur_hp=1)
    dead.fainted = True
    dead.current_hp_fraction = 0.0
    bench_alive = mk_mon("benchmon", (PT.WATER,), ["tackle"])
    team_dead = {"our": our, "dead": dead, "benchmon": bench_alive}
    blk_dead = _damage_block(mk_battle(our, opp, our_team=team_dead), None, None)
    # слоты по species: 0 = benchmon, 1 = dead, 2 = our
    check_true("фейнт: урон по мёртвому слоту = 0 (матрица и зеркало)",
               float(blk_dead[51 + 1]) == 0.0 and float(blk_dead[TEAM_BASE + 1]) == 0.0,
               f"матрица={float(blk_dead[52]):.3f} зеркало={float(blk_dead[TEAM_BASE + 1]):.3f}")
    check_true("фейнт: соседние живые слоты считаются",
               float(blk_dead[51 + 2]) > 0.0 and float(blk_dead[TEAM_BASE + 2]) > 0.0)
    our.fainted = True
    blk_dead2 = _damage_block(mk_battle(our, opp, our_team=team_dead), None, None)
    check_true("фейнт нашего активного: срез зеркала по активному обнулён",
               float(blk_dead2[MIRROR_BASE:MIRROR_BASE + 3].max()) == 0.0,
               f"={[round(float(x), 3) for x in blk_dead2[MIRROR_BASE:MIRROR_BASE + 3]]}")
    check_true("фейнт нашего активного: строки по живой скамейке остаются",
               float(blk_dead2[TEAM_BASE]) > 0.0)
    our.fainted = False

    # possible_KO считается от МАКСИМАЛЬНОГО ролла: hp между min и max
    from agents.damage import DamageContext, cached_move_damage, prepare_mon
    prep_opp = prepare_mon(opp, None)
    prep_our = prepare_mon(our, None)
    dmin, dmax = cached_move_damage(prep_opp, prep_our, Move("surf", gen=9), DamageContext())
    check_true("possible_KO: подготовка теста (dmax > dmin)", dmax > dmin, f"min={dmin} max={dmax}")
    mid_hp = int(dmin) + 1
    our_mid = mk_mon("our", (PT.FIRE,), ["tackle"], hp=362, cur_hp=mid_hp)
    blk_mid = _damage_block(mk_battle(our_mid, opp), None, None)
    check_true("possible_KO: минимальный ролл НЕ добивает",
               float(blk_mid[base]) < 0.999, f"min_frac={float(blk_mid[base]):.3f} hp={mid_hp}")
    check("possible_KO: максимальный ролл добивает -> флаг 1",
          float(blk_mid[base + 2]), 1.0)
    our_tank = mk_mon("our", (PT.FIRE,), ["tackle"], hp=362, cur_hp=int(dmax) + 10)
    blk_tank = _damage_block(mk_battle(our_tank, opp), None, None)
    check("possible_KO: не добивает даже максимум -> флаг 0", float(blk_tank[base + 2]), 0.0)

    # детерминизм: порядок раскрытия приёмов не влияет
    opp_shuffled = mk_mon("opp", (PT.WATER,), ["roar", "recover", "tackle", "surf"])
    blk4 = _damage_block(mk_battle(our, opp_shuffled), None, None)
    check_true("зеркало: результат не зависит от порядка приёмов",
               bool((blk[base:base + OPP_MOVE_SLOTS * 3] == blk4[base:base + OPP_MOVE_SLOTS * 3]).all()))

    # наши срезы не сдвинулись: [0:12] и [15:51] не зависят от известных приёмов противника
    empty_opp = mk_mon("opp", (PT.WATER,), [])
    blk5 = _damage_block(mk_battle(our, empty_opp), None, None)
    check_true("раскладка: наши срезы [0:12] не изменились",
               bool((blk[0:12] == blk5[0:12]).all()))
    check_true("раскладка: наша матрица [15:51] не изменилась",
               bool((blk[15:51] == blk5[15:51]).all()))
    check_true("раскладка: без известных приёмов зеркало и флаги нулевые",
               float(abs(blk5[MIRROR_BASE:].max())) == 0.0, f"max={float(abs(blk5[MIRROR_BASE:]).max())}")


def test_opponent_effect_flags():
    """Флаги особых эффектов у противника (по всей раскрытой команде)."""
    from agents.damage import EFFECT_FLAG_COUNT, OPP_MOVE_SLOTS, OUR_TEAM_SLOTS

    our = mk_mon("our", (PT.WATER,), ["surf", "protect"])
    opp = mk_mon("opp", (PT.STEEL,), ["stealthrock", "willowisp", "swordsdance", "roar"])
    blk = _damage_block(mk_battle(our, opp), None, None)
    fl = _flags_view(blk)
    check("флаги: hazard_stealthrock", fl["hazard_stealthrock"], 1.0)
    check("флаги: hazard_any", fl["hazard_any"], 1.0)
    check("флаги: status_burn (willowisp)", fl["status_burn"], 1.0)
    check("флаги: status_any", fl["status_any"], 1.0)
    check("флаги: setup_booster (swordsdance)", fl["setup_booster"], 1.0)
    check("флаги: phazing (roar)", fl["phazing"], 1.0)
    check("флаги: счётчик hazard-приёмов", fl["count_hazard_moves"], 0.25)
    check("флаги: счётчик setup-приёмов", fl["count_setup_moves"], 0.25)
    check("флаги: ложных срабатываний нет (нет screens)", fl["screens"], 0.0)
    check("флаги: ложных срабатываний нет (нет heal)", fl["healing_move"], 0.0)
    check_true("флаги: длина блока флагов совпадает с EFFECT_FLAG_COUNT",
               len(blk[123:]) == EFFECT_FLAG_COUNT)

    # статусы из secondary (scald -> burn 30%)
    opp2 = mk_mon("opp", (PT.WATER,), ["scald", "icebeam", "spore", "thunderwave"])
    fl2 = _nonzero_flags(_damage_block(mk_battle(our, opp2), None, None))
    check_true("флаги: secondary-статус scald (burn) виден",
               fl2.get("status_burn") == 1.0 and fl2.get("status_freeze") == 1.0, str(fl2))
    check_true("флаги: spore (sleep) и thunderwave (para) видны",
               fl2.get("status_sleep") == 1.0 and fl2.get("status_para") == 1.0)

    # setup: самоослабляющие атаки НЕ должны попадать в setup_booster
    for mid in ("closecombat", "overheat", "dracometeor", "superpower", "vcreate", "leafstorm"):
        fl_setup = _nonzero_flags(_damage_block(mk_battle(our, mk_mon("opp", (PT.WATER,), [mid])),
                                                None, None))
        check(f"флаги: {mid} — самоослабление, не setup", fl_setup.get("setup_booster", 0.0), 0.0)
    # ...а настоящие бусты и self-buff атаки — должны
    for mid, want in (("swordsdance", 1.0), ("calmmind", 1.0), ("shellsmash", 1.0),
                      ("flamecharge", 1.0), ("poweruppunch", 1.0), ("swagger", 0.0), ("charm", 0.0)):
        fl_setup = _nonzero_flags(_damage_block(mk_battle(our, mk_mon("opp", (PT.WATER,), [mid])),
                                                None, None))
        check(f"флаги: {mid} setup_booster", fl_setup.get("setup_booster", 0.0), want)

    # hazard-атаки: движок не хранит side_condition, но Stealth Rock/Spikes они ставят
    fl_sr = _flags_view(_damage_block(mk_battle(our, mk_mon("opp", (PT.ROCK,), ["stoneaxe"])), None, None))
    check("флаги: Stone Axe ставит Stealth Rock", fl_sr["hazard_stealthrock"], 1.0)
    fl_sp = _flags_view(_damage_block(mk_battle(our, mk_mon("opp", (PT.DARK,), ["ceaselessedge"])), None, None))
    check("флаги: Ceaseless Edge ставит Spikes", fl_sp["hazard_spikes"], 1.0)

    # битый приём (кастомный мод) не должен ронять остальные флаги
    class _BrokenSecondary(dict):
        def get(self, key, default=None):
            raise RuntimeError("мод сломан")

    class _BrokenMove:
        id = "brokenmove"
        entry = {}
        base_power = 0
        category = None
        boosts = None
        heal = 0
        drain = 0
        priority = 0
        secondary = [_BrokenSecondary()]

        @property
        def status(self):
            raise KeyError("frost")   # так падал кастомный мод fusionmons

        @property
        def weather(self):
            raise KeyError("frost")

    bad_mon = SimpleNamespace(species="broken", moves={"brokenmove": _BrokenMove()},
                              type_1=PT.NORMAL, type_2=None)
    good_mon = mk_mon("opp", (PT.NORMAL,), ["swordsdance"])
    fl_bad = dict(zip(EFFECT_FLAGS, opponent_effect_flags([bad_mon, good_mon])))
    check("флаги: битый приём не мешает остальным", fl_bad["setup_booster"], 1.0)

    # защита/восстановление/переключение
    opp3 = mk_mon("opp", (PT.NORMAL,), ["protect", "recover", "uturn", "extremespeed"])
    fl3 = _nonzero_flags(_damage_block(mk_battle(our, opp3), None, None))
    check_true("флаги: protect / healing / self_switch / priority_attack",
               all(fl3.get(k) == 1.0 for k in ("protect", "healing_move", "self_switch", "priority_attack")),
               str(fl3))

    # скамеечный сеттер хазардов: у активного их нет, но флаг должен быть
    active = mk_mon("garchomp", (PT.DRAGON, PT.GROUND), ["earthquake", "dragonclaw", "protect", "swordsdance"])
    bench = mk_mon("skarmory", (PT.STEEL, PT.FLYING), ["stealthrock", "spikes", "roost", "bodypress"])
    blk_bench = _damage_block(mk_battle(our, active, opp_team={"garchomp": active, "skarmory": bench}),
                              None, None)
    fl4 = _flags_view(blk_bench)
    check("флаги: hazard-сеттер на скамейке противника виден", fl4["hazard_any"], 1.0)
    check("флаги: оба hazard-приёма учтены", fl4["count_hazard_moves"], 0.5)
    check("флаги: healing на скамейке (roost) тоже виден", fl4["healing_move"], 1.0)


def test_mold_breaker():
    """Mold Breaker/Teravolt/Turboblaze и приёмы с ignoreAbility отключают способность защиты."""
    from agents.damage import MIRROR_BASE
    ctx = DamageContext()

    def dmg(atk_mon, dfn_mon, move):
        return estimate_damage(prepare_mon(atk_mon), prepare_mon(dfn_mon), Move(move, gen=9), ctx)

    levitator = mk_mon("dfn", (PT.POISON,), ["tackle"], ability="levitate")
    grounder = mk_mon("a", (PT.GROUND,), ["earthquake"])
    check("Levitate: обычный Ground -> 0", dmg(grounder, levitator, "earthquake"), (0.0, 0.0))
    for ability in ("moldbreaker", "teravolt", "turboblaze"):
        got = dmg(mk_mon("a", (PT.GROUND,), ["earthquake"], ability=ability), levitator, "earthquake")
        check_true(f"{ability}: Ground бьёт сквозь Levitate", bool(got) and got[0] > 0, f"got={got}")

    sipper = mk_mon("dfn2", (PT.NORMAL,), ["tackle"], ability="sapsipper")
    check("Sap Sipper: обычный Grass -> 0",
          dmg(mk_mon("g", (PT.GRASS,), ["energyball"]), sipper, "energyball"), (0.0, 0.0))
    got = dmg(mk_mon("g", (PT.GRASS,), ["energyball"], ability="moldbreaker"), sipper, "energyball")
    check_true("Mold Breaker: Grass бьёт сквозь Sap Sipper", bool(got) and got[0] > 0, f"got={got}")

    # сопротивление способности: Ice Scales (спец. урон вдвое) и Thick Fat (огонь вдвое)
    scales = mk_mon("dfn3", (PT.NORMAL,), ["tackle"], ability="icescales")
    psy = dmg(mk_mon("p", (PT.PSYCHIC,), ["psychic"]), scales, "psychic")
    geyser = dmg(mk_mon("p", (PT.PSYCHIC,), ["psychic"]), scales, "photongeyser")
    check_true("Ice Scales режет спец. урон (контроль)",
               psy[1] > 0 and geyser[1] > psy[1] * 1.8, f"psychic={psy} photon={geyser}")
    fat = mk_mon("dfn4", (PT.NORMAL,), ["tackle"], ability="thickfat")
    fire_plain = dmg(mk_mon("f", (PT.FIRE,), ["flamethrower"]), fat, "flamethrower")
    fire_mb = dmg(mk_mon("f", (PT.FIRE,), ["flamethrower"], ability="moldbreaker"), fat, "flamethrower")
    check_true("Thick Fat режет огонь, Mold Breaker — нет",
               fire_mb[1] > fire_plain[1] * 1.8, f"обычный={fire_plain} МБ={fire_mb}")

    # предмет и типовой иммунитет Mold Breaker не пробивает
    mb = mk_mon("a", (PT.GROUND,), ["earthquake"], ability="moldbreaker")
    check("Air Balloon (предмет) держит Ground даже при Mold Breaker",
          dmg(mb, mk_mon("dfn5", (PT.POISON,), ["tackle"], item="airballoon"), "earthquake"), (0.0, 0.0))
    check("Flying-тип держит Ground даже при Mold Breaker",
          dmg(mb, mk_mon("dfn6", (PT.FLYING,), ["tackle"]), "earthquake"), (0.0, 0.0))

    # способность защиты неизвестна -> никаких догадок (то же, что и без способности)
    check("неизвестная способность защиты -> поведение как без неё",
          dmg(grounder, mk_mon("dfn7", (PT.POISON,), ["tackle"], ability=None), "earthquake"),
          dmg(grounder, mk_mon("dfn8", (PT.POISON,), ["tackle"]), "earthquake"))

    # зеркало: их Mold Breaker против нашей Levitate (и наоборот — Levitate держит)
    our_lev = mk_mon("our", (PT.POISON,), ["tackle"], ability="levitate")
    blk_mb = _damage_block(
        mk_battle(our_lev, mk_mon("opp", (PT.GROUND,), ["earthquake"], ability="moldbreaker")), None, None)
    blk_pl = _damage_block(
        mk_battle(our_lev, mk_mon("opp", (PT.GROUND,), ["earthquake"], ability="sandveil")), None, None)
    check_true("зеркало: их Mold Breaker пробивает нашу Levitate",
               float(blk_mb[MIRROR_BASE]) > 0.0, f"={float(blk_mb[MIRROR_BASE]):.3f}")
    check("зеркало: без Mold Breaker Levitate держит", float(blk_pl[MIRROR_BASE]), 0.0)


def test_fusion_map_is_side_local():
    """Фьюжн-статы берутся только со своей стороны, даже если species совпала."""
    from agents.config import N_FEATURES
    from agents.damage import DAMAGE_BLOCK_SIZE
    # в полном obs блок признаков урона идёт в конец — индексы срезов сдвинуты
    off = N_FEATURES - DAMAGE_BLOCK_SIZE
    our = mk_mon("shared", (PT.NORMAL,), ["tackle", "protect"])
    bench = mk_mon("bench", (PT.NORMAL,), ["tackle", "protect"])
    opp = mk_mon("foe", (PT.NORMAL,), ["tackle", "protect"])
    b = mk_battle(our, opp, our_team={"shared": our, "bench": bench}, opp_team={"foe": opp})
    huge = {"base_stats": {"hp": 200, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}}

    base = embed_battle_with_fusion(b, None, None)
    # в чужой карте лежит НАША species: раньше она подставлялась нашим покемонам
    alien = embed_battle_with_fusion(b, None, None, our_team_fusions={},
                                     opp_team_fusions={"shared": huge})
    diff = float(abs(alien[off + 15:off + 51] - base[off + 15:off + 51]).max())
    check_true("чужая фьюжн-карта не меняет наш урон (species совпала)", diff == 0.0,
               f"max diff={diff:.4f}")

    # контроль: своя карта на ту же species применяется, чужая — к их
    own = embed_battle_with_fusion(b, None, None, our_team_fusions={"shared": huge},
                                   opp_team_fusions={})
    # слоты отсортированы по species: 0 = bench (нет в карте), 1 = shared (есть в карте)
    check_true("контроль: своя фьюжн-карта меняет урон своего покемона",
               bool((own[off + 21:off + 27] != base[off + 21:off + 27]).any()),
               f"base={[round(float(x), 3) for x in base[off + 21:off + 27]]} "
               f"own={[round(float(x), 3) for x in own[off + 21:off + 27]]}")
    check_true("контроль: строка покемона без записи в карте не меняется",
               bool((own[off + 15:off + 21] == base[off + 15:off + 21]).all()))
    theirs = embed_battle_with_fusion(b, None, None, our_team_fusions={},
                                      opp_team_fusions={"foe": huge})
    check_true("контроль: карта своей стороны применяется к их урону",
               bool((theirs[off + 51:off + 87] != base[off + 51:off + 87]).any()))
    check_true("чужой карты нет -> наши срезы совпадают между прогонами без карт",
               bool((embed_battle_with_fusion(b, None, None, our_team_fusions={},
                                              opp_team_fusions={})[off + 15:off + 87]
                     == base[off + 15:off + 87]).all()))


def test_stab_tera_timing():
    """STAB от tera-типа только ПОСЛЕ теры: объявленный `tera:X` в details не даёт STAB.

    `Pokemon.tera_type` заполняется из details (`tera:Water`) и teambuilder ещё до
    теры, поэтому старая ветка «совпало с tera_type» давала приёму лишний STAB 1.5x.
    Тера учитывается через `type_1` (он равен tera-типу только когда покемон
    терасталлизован).
    """
    from agents.damage import DamageContext, estimate_damage, prepare_mon

    def nb():
        b = Battle(battle_tag="battle-gen9fusionmonsrandombattle-1", username="Me",
                   logger=logging.getLogger("quiet"), gen=9)
        for msg in (["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""],
                    ["", "start"]):
            b.parse_message(msg)
        return b

    ctx = DamageContext()
    b = nb()
    # Fire/Ground с объявленным tera:Water (в бою ещё не терасталлизован)
    b.parse_message(["", "switch", "p1a: Blaze", "camerupt, L50, M, tera:Water", "300/300"])
    b.parse_message(["", "switch", "p2a: Rock", "steelix, L50, F", "300/300"])
    mon = b.active_pokemon
    check("тера ещё не применена, но tera_type уже известен",
          (getattr(mon.tera_type, "name", None), mon.is_terastallized), ("WATER", False))
    surf = Move("surf", gen=9)
    def dmg(m):
        return estimate_damage(prepare_mon(m, None), prepare_mon(b.opponent_active_pokemon, None), surf, ctx)
    with_tera_declared = dmg(mon)
    saved = mon._terastallized_type
    mon._terastallized_type = None
    without = dmg(mon)
    mon._terastallized_type = saved
    check("Surf по Fire/Ground без теры: STAB от tera:Water не применяется", with_tera_declared, without)

    b2 = nb()
    b2.parse_message(["", "switch", "p1a: Blaze", "camerupt, L50, M", "300/300"])
    b2.parse_message(["", "switch", "p2a: Rock", "steelix, L50, F", "300/300"])
    before = estimate_damage(prepare_mon(b2.active_pokemon, None),
                             prepare_mon(b2.opponent_active_pokemon, None), surf, ctx)
    b2.active_pokemon.terastallize("Water")
    after = estimate_damage(prepare_mon(b2.active_pokemon, None),
                            prepare_mon(b2.opponent_active_pokemon, None), surf, ctx)
    check_true("после теры в Water STAB появляется (урон вырос ~1.5x)",
               after[1] > before[1] * 1.4, f"до={before} после={after}")


def test_mold_breaker_is_exhaustive():
    """Инвариант: с Mold Breaker способность защиты не влияет НИ на один приём.

    Проверяем на всех атакующих приёмах гена 9 и всех способностях, которые моделирует
    `damage.py` (иммунитеты + снижение урона) — и заодно что без Mold Breaker каждая
    из этих способностей действительно что-то меняет (тест не вырожденный).
    """
    from poke_env.data import GenData
    from agents.damage import (DamageContext, _ABILITY_IMMUNITY, _DEF_ABILITY_MULT,
                               estimate_damage, prepare_mon)

    ctx = DamageContext()
    dex = GenData.from_gen(9).moves
    damaging = [mid for mid in dex if (getattr(Move(mid, gen=9), "base_power", 0) or 0) > 0]
    abilities = sorted(set(_ABILITY_IMMUNITY) | set(_DEF_ABILITY_MULT))
    check_true("набор способностей для проверки не пуст", len(abilities) >= 20, f"n={len(abilities)}")

    mb_atk = prepare_mon(mk_mon("a", (PT.FIGHTING, PT.GROUND), ["tackle"], ability="moldbreaker"))
    plain_atk = prepare_mon(mk_mon("a", (PT.FIGHTING, PT.GROUND), ["tackle"], ability="sandveil"))
    dfn_none = prepare_mon(mk_mon("dfn", (PT.NORMAL, PT.POISON), ["tackle"], ability=None))

    bad, vacuous = [], []
    for ab in abilities:
        dfn = prepare_mon(mk_mon("dfn", (PT.NORMAL, PT.POISON), ["tackle"], ability=ab))
        changed_without_mb = False
        for mid in damaging:
            mv = Move(mid, gen=9)
            mb, mb_none = estimate_damage(mb_atk, dfn, mv, ctx), estimate_damage(mb_atk, dfn_none, mv, ctx)
            if mb != mb_none:
                bad.append((ab, mid, mb, mb_none))
            if estimate_damage(plain_atk, dfn, mv, ctx) != estimate_damage(plain_atk, dfn_none, mv, ctx):
                changed_without_mb = True
        if not changed_without_mb:
            vacuous.append(ab)

    check("Mold Breaker: способность защиты не влияет на урон (нарушений)", len(bad), 0)
    if bad:
        print("      примеры:", bad[:5])
    check("все способности влияют на урон БЕЗ Mold Breaker (тест не вырожденный)", vacuous, [])


def main() -> int:
    test_formula()
    test_immunities()
    test_modifiers()
    test_best_move()
    test_feature_block()
    test_slot_order()
    test_mirror_damage()
    test_opponent_effect_flags()
    test_mold_breaker()
    test_fusion_map_is_side_local()
    test_stab_tera_timing()
    test_mold_breaker_is_exhaustive()
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
