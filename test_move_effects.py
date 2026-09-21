#!/usr/bin/env python3
"""Полная оценка приёмов: что модель реально знает о каждом приёме.

Требование (ee): помимо урона модель должна различать
  * лечение — сколько процентов HP (и доля урона, возвращаемая в HP — drain),
  * шанс наложения статуса, В ТОМ ЧИСЛЕ НА СЕБЯ (Rest),
  * форсированное применение (свитч цели — phaze, уже было),
  * возрождение союзного покемона (Revival Blessing) — с какой долей HP,
  * приоритет (было), отдача/recoil (было),
  * сколько ходов приём заряжается (Solar Beam/Fly/Geomancy) и сколько перезаряжается (Hyper Beam),
  * сколько стадий бустов/дебаффов и ПО КАКИМ СТАТАМ (включая accuracy/evasion),
  * какие эффекты способностей задействует приём (Bulletproof/Sharpness/Strong Jaw/Iron Fist/
    Mega Launcher/Wind Rider/Dancer/Powder),
  * роняет ли приём своего покемона (Explosion/Memento/Healing Wish),
  * какой хазард ставит (Stealth Rock/Spikes/Toxic Spikes/Sticky Web).

Проверяем и значения (сверка с данными Showdown), и форму доставки в obs, и главное —
что старые колонки не сдвинулись: слепок смещений 55 прежних блоков заморожен ниже
(сгенерирован из коммита fc48248) и сверяется с текущей раскладкой.

Запуск: PYTHONPATH=. python test_move_effects.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import config as C  # noqa: E402
from agents import dims as D  # noqa: E402
from agents import features as F  # noqa: E402

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


# Слепок из fc48248 (obs 991): имя блока -> (смещение, ширина). Ни одна строка не должна
# сдвинуться: все новые признаки обязаны лежать В ХВОСТЕ.
FROZEN_OLD_LAYOUT = [
    ("moves_base_power", 0, 4),
    ("moves_dmg_multiplier", 4, 4),
    ("moves_wasted", 8, 4),
    ("moves_accuracy", 12, 4),
    ("moves_pp_frac", 16, 4),
    ("moves_boost_own", 20, 20),
    ("moves_drop_opp", 40, 20),
    ("moves_hazard_clear", 60, 8),
    ("moves_heal", 68, 4),
    ("moves_status_prob", 72, 4),
    ("moves_priority", 76, 4),
    ("moves_stab", 80, 4),
    ("moves_recoil", 84, 4),
    ("moves_phaze", 88, 4),
    ("moves_contact", 92, 4),
    ("moves_sound", 96, 4),
    ("moves_multihit", 100, 4),
    ("moves_type", 104, 4),
    ("moves_category", 108, 12),
    ("faint_hp", 120, 4),
    ("our_status", 124, 7),
    ("opp_status", 131, 7),
    ("our_hazards", 138, 4),
    ("opp_hazards", 142, 4),
    ("our_switches", 146, 2),
    ("opp_switches", 148, 2),
    ("our_boosts", 150, 7),
    ("opp_boosts", 157, 7),
    ("our_actual_stats", 164, 6),
    ("opp_actual_stats", 170, 6),
    ("our_ability", 176, 20),
    ("opp_ability", 196, 20),
    ("weather", 216, 5),
    ("field", 221, 5),
    ("trick_room_tailwind", 226, 3),
    ("our_screens", 229, 3),
    ("opp_screens", 232, 3),
    ("speed_advantage", 235, 1),
    ("revealed", 236, 2),
    ("semi_invuln", 238, 2),
    ("sub_damaged", 240, 2),
    ("restricted", 242, 1),
    ("our_volatiles", 243, 11),
    ("opp_volatiles", 254, 11),
    ("our_item", 265, 11),
    ("opp_item", 276, 11),
    ("our_bench", 287, 200),
    ("opp_bench", 487, 200),
    ("vulnerability", 687, 2),
    ("tera_flags", 689, 3),
    ("is_tera", 692, 2),
    ("our_tera_type", 694, 19),
    ("protect", 713, 2),
    ("damage", 715, 155),
    ("type_matchup", 870, 121),
]


def mv(mid):
    from poke_env.battle.move import Move
    return Move(mid, gen=9)


def boost_slot(move, kind, stat):
    keys = F._BOOST_KEYS_EXT
    vec = F._move_boost_magnitude(move, kind)
    return float(vec[keys.index(stat)])


def move_col(name, slot=0):
    """Колонка блока `name` внутри вектора кандидата-приёма (учёт слотов 0..3)."""
    col = 0
    for bname, width, kind in F.OBS_BLOCKS:
        if kind != "per_move":
            continue
        per = width // 4
        if bname == name:
            return col + slot * per
        col += per
    raise KeyError(name)


def battle_obs(battle):
    return F.embed_battle_with_fusion(battle, None, None, our_team_fusions={}, opp_team_fusions={})


def main() -> int:
    from test_slot_free_mode import make_battle, make_policy, obs_of
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 78)
    print("0. Размерности: одно число во всех источниках")
    print("=" * 78)
    lay = F.obs_layout()
    check("dims.N_FEATURES == config.N_FEATURES == features.N_FEATURES",
          D.N_FEATURES == C.N_FEATURES == F.N_FEATURES, f"{D.N_FEATURES}/{C.N_FEATURES}/{F.N_FEATURES}")
    check("сумма блоков OBS_BLOCKS == N_FEATURES", lay["_total"][1] == D.N_FEATURES,
          f"{lay['_total'][1]} vs {D.N_FEATURES}")
    check("obs боя ровно N_FEATURES", battle_obs(make_battle()).shape == (D.N_FEATURES,),
          str(battle_obs(make_battle()).shape))
    check("вектор кандидата-приёма содержит новые блоки (65 = 33 + 32)",
          (F.MOVE_FEATURES_DIM, F.MOVE_CAND_DIM) == (65, 67), f"{F.MOVE_FEATURES_DIM}/{F.MOVE_CAND_DIM}")
    check("MOVE_EFFECT_BLOCK_SIZE == 128", F.MOVE_EFFECT_BLOCK_SIZE == 128, str(F.MOVE_EFFECT_BLOCK_SIZE))

    print("-" * 78)
    print("1. Рост только в хвост: старые смещения не сдвинулись")
    print("-" * 78)
    cur = {}
    off = 0
    for name, width, _kind in F.OBS_BLOCKS:
        cur[name] = (off, width)
        off += width
    shifted = [(n, o, w, cur.get(n)) for n, o, w in FROZEN_OLD_LAYOUT if cur.get(n) != (o, w)]
    check("все 55 прежних блоков на своих местах", not shifted, f"сдвинулось: {shifted[:3]}")
    tail_blocks = [(n, cur[n][0]) for n, _w, k in F.OBS_BLOCKS if k == "per_move"][-10:]
    check("первые 991 признаков — прежние, хвост начинается на 991",
          min(o for _n, o in tail_blocks) == 991, str(tail_blocks[:2]))
    check("порядок хвоста == MOVE_EFFECT_TAIL_BLOCKS",
          tuple(n for n, _o in tail_blocks) == F.MOVE_EFFECT_TAIL_BLOCKS,
          str(tuple(n for n, _o in tail_blocks)))
    check("хвост заканчивается на N_FEATURES", max(cur[n][0] + cur[n][1] for n, _o in tail_blocks) == D.N_FEATURES)

    print("-" * 78)
    print("2. Значения по данным Showdown: бусты/дебаффы, статус на себя, зарядка...")
    print("-" * 78)
    check("Swords Dance: +2 atk себе", boost_slot(mv("swordsdance"), "own", "atk") == 2.0,
          str(boost_slot(mv("swordsdance"), "own", "atk")))
    check("Howl: +1 atk (цель allies — наша сторона, а не соперник)",
          boost_slot(mv("howl"), "own", "atk") == 1.0 and boost_slot(mv("howl"), "opp", "atk") == 0.0,
          f"own={boost_slot(mv('howl'), 'own', 'atk')} opp={boost_slot(mv('howl'), 'opp', 'atk')}")
    check("Calm Mind: +1 spa и +1 spd",
          boost_slot(mv("calmmind"), "own", "spa") == 1.0 and boost_slot(mv("calmmind"), "own", "spd") == 1.0)
    check("Close Combat: СВОИ def/spd -1 (видно, в отличие от старого флага)",
          boost_slot(mv("closecombat"), "own", "def") == -1.0
          and boost_slot(mv("closecombat"), "own", "spd") == -1.0)
    check("Hone Claws: atk +1 и accuracy +1 (ось accuracy была невидима)",
          boost_slot(mv("honeclaws"), "own", "atk") == 1.0
          and boost_slot(mv("honeclaws"), "own", "accuracy") == 1.0)
    check("Growl: atk соперника -1", boost_slot(mv("growl"), "opp", "atk") == -1.0)
    check("Sand Attack: accuracy соперника -1", boost_slot(mv("sandattack"), "opp", "accuracy") == -1.0)
    check("Memento: atk/spa соперника -2 И роняет своего",
          boost_slot(mv("memento"), "opp", "atk") == -2.0 and boost_slot(mv("memento"), "opp", "spa") == -2.0
          and F._move_faint_user_flag(mv("memento")) == 1.0)
    check("бусты в own не попадают в opp (и наоборот)",
          boost_slot(mv("swordsdance"), "opp", "atk") == 0.0
          and boost_slot(mv("growl"), "own", "atk") == 0.0)
    check("Rest: статус НА СЕБЯ 100%", F._move_status_self_prob(mv("rest")) == 1.0)
    check("Spore: статус на СЕБЯ не приписан (это статус цели)",
          F._move_status_self_prob(mv("spore")) == 0.0
          and F._move_status_prob(mv("spore")) == 1.0, str(F._move_status_prob(mv("spore"))))
    check("Solar Beam: 1 ход зарядки", F._move_charge_turns(mv("solarbeam")) == 1.0)
    check("Fly/Dig: тоже зарядка", F._move_charge_turns(mv("fly")) == 1.0 and F._move_charge_turns(mv("dig")) == 1.0)
    check("Hyper Beam: перезарядка следующего хода", F._move_recharge_flag(mv("hyperbeam")) == 1.0)
    check("Tackle: ни зарядки, ни перезарядки",
          F._move_charge_turns(mv("tackle")) == 0.0 and F._move_recharge_flag(mv("tackle")) == 0.0)
    check("Revival Blessing: возрождение союзника на 50% HP", F._move_revive_frac(mv("revivalblessing")) == 0.5)
    check("Giga Drain: drain 50% урона в HP", F._move_drain_pct(mv("gigadrain")) == 0.5)
    check("Explosion/Healing Wish: роняют своего",
          F._move_faint_user_flag(mv("explosion")) == 1.0 and F._move_faint_user_flag(mv("healingwish")) == 1.0)
    check("Stealth Rock/Spikes/Toxic Spikes/Sticky Web: какой хазард ставится",
          list(map(int, F._move_hazard_set_flags(mv("stealthrock")))) == [1, 0, 0, 0]
          and list(map(int, F._move_hazard_set_flags(mv("spikes")))) == [0, 1, 0, 0]
          and list(map(int, F._move_hazard_set_flags(mv("toxicspikes")))) == [0, 0, 1, 0]
          and list(map(int, F._move_hazard_set_flags(mv("stickyweb")))) == [0, 0, 0, 1])
    check("Tackle: хазард не ставит", not any(F._move_hazard_set_flags(mv("tackle"))))
    flags = {n: int(f) for n, f in zip(F._ABILITY_MOVE_FLAGS, F._move_ability_flags(mv("bulletseed")))}
    check("Bullet Seed: bullet (Bulletproof)", flags.get("bullet") == 1, str(flags))
    check("Slash: slicing (Sharpness)", int(F._move_ability_flags(mv("slash"))[1]) == 1)
    check("Ice Punch: punch (Iron Fist)", int(F._move_ability_flags(mv("icepunch"))[3]) == 1)
    check("Dark Pulse: pulse (Mega Launcher)", int(F._move_ability_flags(mv("darkpulse"))[4]) == 1)
    check("Whirlwind: wind (Wind Rider)", int(F._move_ability_flags(mv("whirlwind"))[5]) == 1)
    check("Swords Dance: dance (Dancer)", int(F._move_ability_flags(mv("swordsdance"))[6]) == 1)
    check("Sleep Powder: powder (Powder)", int(F._move_ability_flags(mv("sleeppowder"))[7]) == 1)
    check("Ancient Power: secondary 10% -> ожидаемый буст 0.1 по 5 статам",
          abs(boost_slot(mv("ancientpower"), "own", "atk") - 0.1) < 1e-6,
          str(boost_slot(mv("ancientpower"), "own", "atk")))
    check("Ice Punch: статус цели 0.1 (не собран как собственный буст)",
          abs(F._move_status_prob(mv("icepunch")) - 0.1) < 1e-6
          and abs(boost_slot(mv("icepunch"), "own", "atk")) < 1e-9,
          f"status={F._move_status_prob(mv('icepunch'))}")

    print("-" * 78)
    print("3. Доставка в obs: колонки, пустые слоты, отсутствие сдвигов")
    print("-" * 78)
    b = make_battle(moves=("solarbeam", "swordsdance", "explosion", "protect"))
    obs = battle_obs(b)
    idx = F.move_feature_index()
    c_charge = move_col("moves_charge_turns")
    c_recharge = move_col("moves_recharge")
    c_faint = move_col("moves_faint_user")
    c_boost = move_col("moves_boost_own_stages")
    check("Solar Beam в слоте 0: charge=1 в obs",
          float(obs[idx[0, c_charge]]) == 1.0, str(float(obs[idx[0, c_charge]])))
    check("Swords Dance в слоте 1: +2 atk в obs",
          float(obs[idx[1, c_boost + F._BOOST_KEYS_EXT.index("atk")]]) == 2.0)
    check("Explosion в слоте 2: faint_user=1 в obs",
          float(obs[idx[2, c_faint]]) == 1.0)
    check("Protect в слоте 3: зарядки/перезарядки/бустов нет",
          float(obs[idx[3, c_charge]]) == 0.0 and float(obs[idx[3, c_recharge]]) == 0.0
          and not any(obs[idx[3, c_boost:c_boost + 7]]))
    b1 = make_battle(moves=("tackle",))
    obs1 = battle_obs(b1)
    empty_cols = np.concatenate([obs1[idx[slot, :]] for slot in (1, 2, 3)])
    check("пустые слоты приёмов: новые блоки строго нулевые (без мусора)",
          float(np.abs(empty_cols[-32:]).max()) == 0.0, str(float(np.abs(empty_cols[-32:]).max())))
    check("новые блоки читаются только из своего слота (нет перекрёстного залипания)",
          float(obs[idx[0, c_boost]]) == 0.0 and float(obs[idx[1, c_charge]]) == 0.0)
    check("срезы кандидатов согласованы (candidate_slices_ok)",
          bool(F.candidate_slices_ok(obs)))
    pol = make_policy("embed")
    logits, values = pol.logits_and_values(obs_of(b))
    check("политика embed принимает obs с новыми признаками и считает логиты 26",
          tuple(logits.shape) == (1, 26) and tuple(values.shape) == (1, 1),
          f"{tuple(logits.shape)}/{tuple(values.shape)}")

    print("-" * 78)
    print("4. Все приёмы данных Showdown обрабатываются без исключений")
    print("-" * 78)
    from poke_env.data import GenData
    from poke_env.battle.move import Move
    data = GenData.from_gen(9).moves
    stats = {"n": 0, "charge": 0, "recharge": 0, "revive": 0, "faint": 0, "drain": 0, "hazard": 0, "self_status": 0}
    errs = 0
    for mid in data:
        try:
            m = Move(mid, gen=9)
            stats["n"] += 1
            stats["charge"] += int(F._move_charge_turns(m) > 0)
            stats["recharge"] += int(F._move_recharge_flag(m) > 0)
            stats["revive"] += int(F._move_revive_frac(m) > 0)
            stats["faint"] += int(F._move_faint_user_flag(m) > 0)
            stats["drain"] += int(F._move_drain_pct(m) > 0)
            stats["hazard"] += int(any(F._move_hazard_set_flags(m)))
            stats["self_status"] += int(F._move_status_self_prob(m) > 0)
            F._move_boost_magnitude(m, "own"); F._move_boost_magnitude(m, "opp"); F._move_ability_flags(m)
        except Exception as e:  # noqa: BLE001 — тест обязан ловить любой сбой
            errs += 1
            if errs <= 3:
                print(f"     ошибка на {mid}: {type(e).__name__}: {e}")
    check(f"все приёмы обработаны без исключений (n={stats['n']})", errs == 0, f"ошибок {errs}")
    check("найдены приёмы с зарядкой (>=10)", stats["charge"] >= 10, str(stats))
    check("найден хотя бы один приём с возрождением", stats["revive"] >= 1, str(stats))
    check("найдены приёмы с отдачей HP (drain >= 10)", stats["drain"] >= 10, str(stats))
    check("найдены хазард-приёмы (>=4)", stats["hazard"] >= 4, str(stats))
    check("Rest попал в self-status", stats["self_status"] >= 1, str(stats))

    print("-" * 78)
    print("5. Прежние сигналы не поехали (регресс уже работавших признаков)")
    print("-" * 78)
    check("heal: Roost 50% (было и осталось)", abs(F._move_heal_pct(mv("roost")) - 0.5) < 1e-9,
          str(F._move_heal_pct(mv("roost"))))
    check("priority: Extreme Speed положительный, Tackle нулевой",
          F._move_priority(mv("extremespeed")) > 0 and F._move_priority(mv("tackle")) == 0.0,
          f"{F._move_priority(mv('extremespeed')):.3f}")
    check("recoil: Double-Edge > 0", F._move_recoil_pct(mv("doubleedge")) > 0)
    check("phaze: Whirlwind форсирует свитч", F._move_phaze_flag(mv("whirlwind")) == 1.0)
    check("wasted: нейтральный Tackle — не wasted; Earthquake по Flying-цели — wasted (иммунитет)",
          F._move_wasted_flag(mv("tackle"), b) == 0.0
          and F._move_wasted_flag(mv("earthquake"), b) == 1.0,
          f"tackle={F._move_wasted_flag(mv('tackle'), b)} eq={F._move_wasted_flag(mv('earthquake'), b)}")
    check("wasted: хил при полном HP — wasted",
          F._move_wasted_flag(mv("recover"), b) == 1.0)
    check("multihit: Bullet Seed", F._move_multihit_flag(mv("bulletseed")) == 1.0)
    check("contact: Close Combat", F._move_contact_flag(mv("closecombat")) == 1.0)
    check("sound: Roar", F._move_sound_flag(mv("roar")) == 1.0)

    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
