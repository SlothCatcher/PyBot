"""
Верификатор features.py — гоняется без сервера, на heuristic_dataset.npz + моках.
Запуск:  python test_features_verify.py
         python test_features_verify.py --samples 5000
"""
import argparse
import numpy as np
import pathlib
import sys

from agents.config import N_FEATURES, VECNORM_PATH
from agents.features import (
    _TYPE_LIST, _STATUSES, _VOLATILES, _KEY_ITEMS, _RESERVE_SLOT_SIZE,
    _WEATHERS, _FIELDS, MAX_RESERVES, embed_battle_with_fusion,
)


def check_dataset(samples: int = 5000):
    print("="*70)
    print("[1] heuristic_dataset.npz — форма и границы Box(-1,4)")
    print("="*70)
    p = pathlib.Path("models/heuristic_dataset.npz")
    if not p.exists():
        print(f"  SKIP: {p} не найден")
        return False
    data = np.load(p)
    obs, mask, action = data["obs"], data["mask"], data["action"]
    print(f"  obs shape {obs.shape}, mask {mask.shape}, action [{action.min()}, {action.max()}]")
    print(f"  N_FEATURES config={N_FEATURES}, dataset={obs.shape[1]}, match={obs.shape[1]==N_FEATURES} -> {'OK' if obs.shape[1]==N_FEATURES else 'FAIL'}")
    if obs.shape[1] != N_FEATURES:
        print("  FAIL: пересобери датасет или поправь N_FEATURES")
        return False
    n_nan = np.isnan(obs).sum()
    n_inf = np.isinf(obs).sum()
    print(f"  NaN {n_nan}, Inf {n_inf} -> {'OK' if n_nan==0 and n_inf==0 else 'FAIL'}")
    o_min, o_max, o_mean = obs.min(), obs.max(), obs.mean()
    print(f"  obs min {o_min:.3f} max {o_max:.3f} mean {o_mean:.3f} (Box -1..4)")
    out_of_box = ((obs < -1-1e-6) | (obs > 4+1e-6)).sum()
    print(f"  out-of-Box элементов {out_of_box} ({100*out_of_box/obs.size:.4f}%) -> {'OK' if out_of_box==0 else 'WARN'}")
    mask_sums = mask.sum(axis=1)
    print(f"  mask sum per sample: min {mask_sums.min()} max {mask_sums.max()} mean {mask_sums.mean():.1f} zeros {(mask_sums==0).sum()}")
    if (mask_sums==0).sum()>0:
        print("  WARN: есть сэмплы с пустой маской (должны идти в DefaultBattleOrder)")
    bad = 0
    for i in range(min(samples, len(obs))):
        a = int(action[i])
        if a < 0 or a >= mask.shape[1]:
            bad+=1
        elif mask[i, a]==0:
            bad+=1
    print(f"  action вне маски (первые {min(samples,len(obs))}): {bad} -> {'OK' if bad==0 else 'FAIL'}")
    groups = [
        ("moves_base(4)", 0,4), ("moves_dmg(4)",4,8), ("wasted(4)",8,12), ("acc(4)",12,16), ("pp(4)",16,20),
        ("fainted+hp(4)",20,24), ("our_status(7)",24,31), ("opp_status(7)",31,38),
        ("our_haz(4)",38,42), ("opp_haz(4)",42,46), ("our_sw(2)",46,48), ("opp_sw(2)",48,50),
        ("our_boost(5)",50,55), ("opp_boost(5)",55,60), ("weather(5)",60,65), ("field(5)",65,70),
        ("speed_adv(1)",70,71), ("revealed(2)",71,73), ("semi(2)",73,75), ("sub(2)",75,77), ("restr(1)",77,78),
        ("our_vol(11)",78,89), ("opp_vol(11)",89,100), ("our_item(11)",100,111), ("opp_item(11)",111,122),
        ("our_bench(135)",122,257), ("opp_bench(135)",257,392), ("vuln(2)",392,394), ("tera_flags(3)",394,397), ("tera_type(19)",397,416), ("protect(2)",416,418),
    ]
    print("\n  Проверка групп (mean должен быть !=0, std>0 для живых фич):")
    # semi/sub/tera_type — редкие: в heuristic_dataset 0.0 ожидаемо, т.к. SimpleHeuristics почти не юзает Substitute/SkyDrop/Tera
    # и старый poke_env не парсил tera. После фикса детекторы оживают в синтетике/живых боях, dataset остаётся DEAD исторически.
    RARE_GROUPS = {"semi(2)", "sub(2)", "tera_type(19)"}
    for name, a,b in groups:
        seg = obs[:samples, a:b]
        m, s, mn, mx = seg.mean(), seg.std(), seg.min(), seg.max()
        if s < 1e-6:
            dead = "RARE (ok, see synth)" if name in RARE_GROUPS else "DEAD"
        else:
            dead = "ok"
        print(f"    {name:18s} [{a:3d}:{b:3d}] mean {m:6.3f} std {s:5.3f} [{mn:5.2f},{mx:5.2f}] {dead}")
    print("\n[2] VecNormalize")
    try:
        import pickle
        vec = pickle.load(open(VECNORM_PATH,"rb"))
        dim = vec.observation_space["observation"].shape[0]
        print(f"  vecnormalize dim {dim} -> {'OK' if dim==N_FEATURES else 'FAIL'}")
        rms = vec.obs_rms["observation"]
        print(f"  obs_rms mean[:3] {rms.mean[:3]}, var[:3] {rms.var[:3]}, count {rms.count}")
        sample = obs[:2]
        normed = vec.normalize_obs({"observation": sample})["observation"]
        print(f"  normalize sample[0,:3] {sample[0,:3]} -> {normed[0,:3]}")
        print(f"  normed mean {normed.mean():.3f} std {normed.std():.3f} (ожидается ~0/1)")
    except Exception as e:
        print(f"  SKIP: не удалось загрузить {VECNORM_PATH}: {e}")
    print()
    return True


class FakePokemon:
    def __init__(self, **kw):
        self.species = kw.get("species","Pikachu")
        self.base_species = kw.get("base_species","Pikachu")
        self.type_1 = kw.get("type_1", _TYPE_LIST[0])
        self.type_2 = kw.get("type_2", None)
        self.current_hp_fraction = kw.get("hp", 1.0)
        self.fainted = kw.get("fainted", False)
        self.status = kw.get("status", None)
        self.boosts = kw.get("boosts", {})
        self.effects = kw.get("effects", {})
        self.moves = kw.get("moves", {})
        self.base_stats = kw.get("base_stats", {"spe":100, "hp":100,"atk":100,"def":100,"spa":100,"spd":100})
        self.level = kw.get("level", 100)
        self.item = kw.get("item", "")
        self.ability = kw.get("ability", None)
        self.active = kw.get("active", False)
        self.tera_type = kw.get("tera_type", None)
        self.is_terastallized = kw.get("is_terastallized", False)
        self.current_hp = kw.get("current_hp", 100)
        self.max_hp = kw.get("max_hp", 100)


class FakeMove:
    def __init__(self, id="tackle", base_power=40, type=None, accuracy=100, pp=16, max_pp=16, status=None):
        self.id = id
        self.base_power = base_power
        self.type = type or _TYPE_LIST[0]
        self.accuracy = accuracy
        self.current_pp = pp
        self.max_pp = max_pp
        self.status = status


def check_embed_synthetic():
    print("="*70)
    print("[3] Синтетические батлы — embed_battle_with_fusion")
    print("="*70)
    from poke_env.battle import Status, SideCondition, Weather, Field, Effect, PokemonType
    from poke_env.data import GenData

    class MockBattle:
        def __init__(self):
            self.gen = 9
            self.weather = {}
            self.fields = {}
            self.side_conditions = {}
            self.opponent_side_conditions = {}
            self.team = {}
            self.opponent_team = {}
            self.active_pokemon = None
            self.opponent_active_pokemon = None
            self.available_moves = []
            self.available_switches = []
            self.player_role = "p1"
            self.battle_tag = "test-battle"
            self.trapped = False
            self.can_tera = True

    def run_one(name, battle, our_fus=None, opp_fus=None, our_prot=0, opp_prot=0):
        try:
            obs = embed_battle_with_fusion(battle, our_fusion=our_fus, opp_fusion=opp_fus,
                                           our_protected_last_turn=our_prot, opp_protected_last_turn=opp_prot)
            ok = obs.shape==(418,) and np.isfinite(obs).all()
            print(f"  {name:40s} shape {obs.shape} finite {np.isfinite(obs).all()} min {obs.min():.2f} max {obs.max():.2f} -> {'OK' if ok else 'FAIL'}")
            return obs
        except Exception as e:
            print(f"  {name:40s} EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            return None

    b = MockBattle()
    run_one("пустой (нет активных)", b)

    b = MockBattle()
    b.side_conditions = {SideCondition.STEALTH_ROCK:1, SideCondition.SPIKES:2}
    b.opponent_side_conditions = {SideCondition.TOXIC_SPIKES:1}
    b.team = {"a": FakePokemon(active=True, hp=0.5)}
    b.opponent_team = {"b": FakePokemon(active=True, hp=0.8)}
    b.active_pokemon = list(b.team.values())[0]
    b.opponent_active_pokemon = list(b.opponent_team.values())[0]
    b.active_pokemon.type_1 = PokemonType.WATER; b.active_pokemon.type_2 = None
    b.opponent_active_pokemon.type_1 = PokemonType.FIRE
    b.available_moves = [FakeMove("hydropump", 110, PokemonType.WATER)]
    run_one("hazards + 1 мув", b)

    b = MockBattle()
    b.active_pokemon = FakePokemon(active=True, boosts={"atk":2,"def":-1,"spa":1}, status=Status.BRN, hp=0.33)
    b.opponent_active_pokemon = FakePokemon(active=True, boosts={"spe":3}, status=Status.PSN, hp=0.9)
    b.team = {"a": b.active_pokemon}
    b.opponent_team = {"b": b.opponent_active_pokemon}
    b.available_moves = [FakeMove("swordsdance",0, PokemonType.NORMAL, status=Status.BRN), FakeMove("flareblitz",120, PokemonType.FIRE)]
    run_one("boosts+status", b)

    b = MockBattle()
    b.active_pokemon = FakePokemon(active=True, effects={Effect.SUBSTITUTE:1, Effect.LEECH_SEED:1})
    b.opponent_active_pokemon = FakePokemon(active=True, effects={Effect.CONFUSION:1})
    b.team = {"a": b.active_pokemon}
    b.opponent_team = {"b": b.opponent_active_pokemon}
    b.available_moves = [FakeMove("substitute",0, PokemonType.NORMAL)]
    obs_sub = run_one("volatiles/sub", b)
    if obs_sub is not None:
        # semi/sub/tera — проверяем что детекторы живые в синтетике
        print(f"    -> sub our {obs_sub[75]:.2f} opp {obs_sub[76]:.2f} (ожидается >0 для our)")
        print(f"    -> semi our {obs_sub[73]:.2f} (ожидается 0 тут, см. следующий тест)")
    # semi via _preparing_move (чиним DEAD: FLY/DIG и т.п. отсутствуют в Effect, ловим через preparing)
    b = MockBattle()
    class FakePokemonPrep(FakePokemon):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._preparing_move = kw.get("_preparing_move", None)
    prep_mon = FakePokemonPrep(active=True, _preparing_move=object(), effects={})
    b.active_pokemon = prep_mon
    b.opponent_active_pokemon = FakePokemon(active=True)
    b.team = {"a": prep_mon}; b.opponent_team = {"b": b.opponent_active_pokemon}
    b.available_moves = [FakeMove("solarbeam",120, PokemonType.GRASS)]
    obs_prep = run_one("semi via preparing (Fly/SolarBeam)", b)
    if obs_prep is not None:
        print(f"    -> semi our {obs_prep[73]:.2f} opp {obs_prep[74]:.2f} (ожидается 1.0/0.0)")
    # tera via fallback _last_details / _terastallized_type (чиним DEAD: poke_env не парсил tera)
    b = MockBattle()
    tera_mon = FakePokemon(active=True, hp=1.0, tera_type=None)
    tera_mon._last_details = "Pikachu, L83, tera:Fire"
    tera_mon._terastallized_type = None
    b.active_pokemon = tera_mon
    b.opponent_active_pokemon = FakePokemon(active=True)
    b.team = {"a": tera_mon}; b.opponent_team = {"b": b.opponent_active_pokemon}
    b.available_moves = [FakeMove("tackle",40, PokemonType.NORMAL)]
    obs_tera = run_one("tera via _last_details fallback", b)
    if obs_tera is not None:
        tera_sum = float(obs_tera[397:416].sum())
        print(f"    -> tera_type sum {tera_sum:.1f} (ожидается 1.0, был DEAD до фикса)")

    b = MockBattle()
    fire = PokemonType.FIRE; grass = PokemonType.GRASS; water = PokemonType.WATER
    b.active_pokemon = FakePokemon(active=True, type_1=fire, hp=1.0)
    b.opponent_active_pokemon = FakePokemon(active=True, type_1=grass, hp=1.0)
    b.team = {
        "active": FakePokemon(active=True, type_1=fire),
        "r1": FakePokemon(active=False, type_1=grass, hp=1.0, fainted=False),
        "r2": FakePokemon(active=False, type_1=water, hp=0.2, fainted=False),
        "r3": FakePokemon(active=False, type_1=fire, hp=0, fainted=True),
        "r4": FakePokemon(active=False, type_1=grass, hp=1.0),
        "r5": FakePokemon(active=False, type_1=grass, hp=1.0),
    }
    b.opponent_team = {
        "a": FakePokemon(active=True, type_1=grass),
        "b": FakePokemon(active=False, type_1=water, hp=1.0),
    }
    b.active_pokemon = b.team["active"]; b.active_pokemon.active=True
    b.opponent_active_pokemon = b.opponent_team["a"]; b.opponent_active_pokemon.active=True
    b.available_moves = [FakeMove("flamethrower",90,fire)]
    obs = run_one("bench+vuln (огонь vs трава)", b)
    if obs is not None:
        vuln_our, vuln_opp = obs[392], obs[393]
        # our bench: 3x Grass + 1x Water vs Grass-активный: Water 2x уязвим -> 0.25; opp bench Water vs Fire -> 0.0
        print(f"    vulnerability our {vuln_our:.2f} opp {vuln_opp:.2f} (ожидается 0.25/0.00 — корректно, Grass→Water 2x)")

    b = MockBattle()
    b.active_pokemon = FakePokemon(active=True, base_stats={"spe":150}, hp=1.0)
    b.opponent_active_pokemon = FakePokemon(active=True, base_stats={"spe":100}, hp=1.0)
    b.team = {"a": b.active_pokemon}; b.opponent_team = {"b": b.opponent_active_pokemon}
    b.available_moves = [FakeMove("tackle",40, PokemonType.NORMAL)]
    fus = {"base_stats":{"spe":200}, "speed_range":(300,330)}
    run_one("fusion speed_range", b, our_fus=fus, opp_fus=None, our_prot=1, opp_prot=0)

    print("\n[4] Баг env: battle is self.battle1 vs player_role")
    class FakeEnv:
        def __init__(self):
            class B1: battle_tag="test"; player_role="p1"
            class B2: battle_tag="test"; player_role="p2"
            self.battle1 = B1()
            self.battle2 = B2()
            self.agent1 = type("A",(),{"_fusion_stats":{"test":{"p1":{"speed_range":(1,1)}}}, "_protect_state":{"test":{"last_p1":True}}})()
            self.agent2 = type("A",(),{"_fusion_stats":{"test":{"p2":{"speed_range":(2,2)}}}, "_protect_state":{"test":{"last_p2":True}}})()
    fake_env = FakeEnv()
    b_copy = type("B",(),{"battle_tag":"test","player_role":"p1"})()
    is_old_bug = (fake_env.agent1 if b_copy is fake_env.battle1 else fake_env.agent2) == fake_env.agent2
    # проверяем фикс: по player_role должен выбраться agent1 (читаем файл, не импортируем — иначе нужен stable_baselines3)
    try:
        src = pathlib.Path("agents/env.py").read_text()
        uses_player_role = "player_role" in src and "battle is self.battle1" not in src.split("def embed_battle")[1].split("def ")[0] if "def embed_battle" in src else "player_role" in src
        # более надёжно: ищем фиксатор в embed_battle
        import re
        embed_src = re.search(r"def embed_battle.*?(?=\n    def |\nclass |\Z)", src, re.S)
        embed_text = embed_src.group(0) if embed_src else src
        # убран первичный 'battle is self.battle1', оставлен только fallback в except + primary player_role
        has_fix = "player_role" in embed_text and 'getattr(battle, "player_role"' in embed_text
        has_primary_is = False
        # проверяем первую строку 'source = ' — должна быть player_role, а не is
        for line in embed_text.splitlines():
            if "source =" in line and "battle" in line:
                has_primary_is = "battle is self.battle1" in line
                break
        fixed_picks_agent1 = (fake_env.agent1 if getattr(b_copy, "player_role", "p1") == "p1" else fake_env.agent2) == fake_env.agent1
        print(f"  Старый код `is`: выберет agent2? {is_old_bug} -> {'BUG (ожидаемо)' if is_old_bug else 'ok'}")
        print(f"  env.py embed_battle primary использует player_role? {has_fix and not has_primary_is} -> {'OK' if (has_fix and not has_primary_is) else 'FAIL'}")
        print(f"  env.py primary 'is' баг? {has_primary_is} -> {'FAIL' if has_primary_is else 'OK'}")
        print(f"  Фикс выбирает agent1 для p1? {fixed_picks_agent1} -> {'OK' if fixed_picks_agent1 else 'FAIL'}")
        if has_fix and not has_primary_is and fixed_picks_agent1:
            print("  Вывод: env.py ПОЧИНЕН (player_role вместо is).")
        else:
            print("  Вывод: env.py ещё не починен — нужен player_role как в players.py")
    except Exception as e:
        print(f"  SKIP env fix check: {e}")
        print(f"  Копия p1 battle: старый код выберет agent2? {is_old_bug} -> {'BUG' if is_old_bug else 'ok'}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()
    ok1 = check_dataset(samples=args.samples)
    check_embed_synthetic()
    print("="*70)
    print("Готово. Если где-то FAIL/DEAD — пришли лог, поправлю.")
    print("Запускай на Windows так же: python test_features_verify.py --samples 5000")
    print("="*70)
