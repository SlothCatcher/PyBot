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
    for name, a,b in groups:
        seg = obs[:samples, a:b]
        m, s, mn, mx = seg.mean(), seg.std(), seg.min(), seg.max()
        dead = "DEAD" if s < 1e-6 else "ok"
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
    run_one("volatiles/sub", b)

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
        print(f"    vulnerability our {vuln_our:.2f} opp {vuln_opp:.2f} (ожидается our 0.0, opp ~1.0)")

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
    is_old_ok = (fake_env.agent1 if b_copy is fake_env.battle1 else fake_env.agent2) == fake_env.agent2
    print(f"  Копия p1 battle: старый код выберет agent2? {is_old_ok} -> {'BUG' if is_old_ok else 'ok'} (должен быть agent1)")
    print(f"  Вывод: в env.py нужно заменить `is` на `battle.player_role` как в players.py")
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
