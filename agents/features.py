import re
import numpy as np
from poke_env.battle import Field, SideCondition, Status, Weather
from poke_env.battle import Effect

_STATUSES = [None, Status.BRN, Status.PAR, Status.SLP, Status.FRZ, Status.PSN, Status.TOX]

_HAZARD_MOVES = {
    "spikes": (SideCondition.SPIKES, 3),
    "stealthrock": (SideCondition.STEALTH_ROCK, 1),
    "toxicspikes": (SideCondition.TOXIC_SPIKES, 2),
    "stickyweb": (SideCondition.STICKY_WEB, 1),
}

_SCREEN_MOVES = {
    "reflect": SideCondition.REFLECT,
    "lightscreen": SideCondition.LIGHT_SCREEN,
    "auroraveil": SideCondition.AURORA_VEIL,
}

_IMMUNITY_ABILITIES = {
    "levitate": "ground",
    "flashfire": "fire",
    "voltabsorb": "electric",
    "lightningrod": "electric",
    "motordrive": "electric",
    "waterabsorb": "water",
    "stormdrain": "water",
    "sapsipper": "grass",
}

_WEATHERS = [None, Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM, Weather.SNOW]
_FIELDS = [None, Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN]

_KEY_ITEMS = [
    "choiceband", "choicespecs", "choicescarf",
    "lifeorb", "focussash", "leftovers",
    "assaultvest", "heavydutyboots", "rockyhelmet",
    "airballoon",
]

_SEMI_INVULN_CHARGE_NAMES = [
    "FLY", "DIG", "DIVE", "BOUNCE", "PHANTOM_FORCE", "PHANTOMFORCE",
    "SHADOW_FORCE", "SHADOWFORCE", "SKY_DROP", "SKYDROP",
    "SOLAR_BEAM", "SOLARBEAM", "SOLAR_BLADE", "SOLARBLADE",
    "SKULL_BASH", "SKULLBASH", "SKY_ATTACK", "SKYATTACK",
    "RAZOR_WIND", "RAZORWIND", "FREEZE_SHOCK", "FREEZESHOCK",
    "ICE_BURN", "ICEBURN", "METEOR_BEAM", "METEORBEAM",
    "ELECTRO_SHOT", "ELECTROSHOT", "GEOMANCY",
]
_SEMI_INVULN_CHARGE_EFFECTS = [
    getattr(Effect, name) for name in _SEMI_INVULN_CHARGE_NAMES if hasattr(Effect, name)
]



_VOLATILES = [
    Effect.LEECH_SEED, Effect.CONFUSION, Effect.SUBSTITUTE, Effect.TAUNT,
    Effect.ENCORE, Effect.DISABLE, Effect.CURSE, Effect.YAWN,
    Effect.PERISH1, Effect.PERISH2, Effect.PERISH3,
]

_STATS_TABLE_RE = re.compile(
    r"<td>(\d+)</td><td>(\d+)</td><td>(\d+)</td><td>(\d+)</td><td>(\d+)</td><td>(\d+)</td>"
)
_SPEED_RANGE_RE = re.compile(r"Possible speed:\s*(\d+)-(\d+)")

_WEATHER_MOVES = {"sunnyday": Weather.SUNNYDAY, "raindance": Weather.RAINDANCE, "sandstorm": Weather.SANDSTORM, "snowscape": Weather.SNOW}
_TERRAIN_MOVES = {"electricterrain": Field.ELECTRIC_TERRAIN, "grassyterrain": Field.GRASSY_TERRAIN, "mistyterrain": Field.MISTY_TERRAIN, "psychicterrain": Field.PSYCHIC_TERRAIN}

from poke_env.battle import PokemonType

_EXCLUDED_TYPE_NAMES = {"THREE_QUESTION_MARKS"}
_TYPE_LIST = sorted(
    [t for t in PokemonType if t.name not in _EXCLUDED_TYPE_NAMES],
    key=lambda t: t.name,
)
_TYPE_INDEX = {t: i for i, t in enumerate(_TYPE_LIST)}

MAX_RESERVES = 5  # 6 покемонов в команде минус 1 активный
_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]
# hazard clear: [own, opp]
_HAZARD_CLEAR_MAP = {
    "rapidspin": (1.0, 0.0),
    "mortalspin": (1.0, 0.0),
    "tidyup": (1.0, 0.0),
    "defog": (1.0, 1.0),
    "courtchange": (1.0, 1.0),
}
_PHAZE_MOVES = {"roar","whirlwind","dragontail","circlethrow","yawn"}  # yawn not phaze but force switch later
_SCREEN_SIDE = [SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL]

def _revealed_moves_frac(pokemon) -> float:
    if pokemon is None:
        return 0.0
    return min(len(pokemon.moves) / 4.0, 1.0)

def _is_move_restricted(battle) -> float:
    pokemon = battle.active_pokemon
    if pokemon is None or len(pokemon.moves) == 0:
        return 0.0
    return 1.0 if len(battle.available_moves) < len(pokemon.moves) else 0.0

def _substitute_damaged(pokemon) -> float:
    """0.0 без куклы; >0 если Substitute стоит. 1.0 ~ 25 HP куклы, 0.5 если точное HP неизвестно."""
    if pokemon is None:
        return 0.0
    if Effect.SUBSTITUTE not in pokemon.effects:
        try:
            if not any(getattr(k, "name", str(k)).lower() == "substitute" for k in pokemon.effects.keys()):
                return 0.0
        except Exception:
            return 0.0
    value = pokemon.effects.get(Effect.SUBSTITUTE, None)
    if value is None:
        for k, v in list(pokemon.effects.items()):
            if getattr(k, "name", str(k)).lower() == "substitute":
                value = v
                break
    if isinstance(value, (int, float)) and value > 0:
        return min(float(value) / 25.0, 1.0)
    if value is True:
        return 0.5
    return 0.5

def _is_semi_invuln_or_charging(pokemon) -> float:
    if pokemon is None:
        return 0.0
    if any(e in pokemon.effects for e in _SEMI_INVULN_CHARGE_EFFECTS):
        return 1.0
    try:
        prep = getattr(pokemon, "_preparing_move", None) or getattr(pokemon, "preparing_move", None)
        if prep is not None:
            return 1.0
    except Exception:
        pass
    try:
        for eff in pokemon.effects.keys():
            n = getattr(eff, "name", str(eff))
            if n in _SEMI_INVULN_CHARGE_NAMES:
                return 1.0
            n2 = n.replace("_", "").lower()
            if any(x.replace("_", "").lower() == n2 for x in _SEMI_INVULN_CHARGE_NAMES):
                return 1.0
    except Exception:
        pass
    return 0.0

def _type_multi_hot(pokemon) -> np.ndarray:
    vec = np.zeros(len(_TYPE_LIST), dtype=np.float32)
    if pokemon is None:
        return vec
    for t in (pokemon.type_1, pokemon.type_2):
        if t is not None and t in _TYPE_INDEX:
            vec[_TYPE_INDEX[t]] = 1.0
    return vec

def _tera_type_vec(pokemon) -> np.ndarray:
    """One-hot тера-типа покемона."""
    vec = np.zeros(len(_TYPE_LIST), dtype=np.float32)
    if pokemon is None:
        return vec
    tera_type = None
    try:
        v = getattr(pokemon, "tera_type", None)
        if isinstance(v, PokemonType) and v in _TYPE_INDEX:
            tera_type = v
    except Exception:
        pass
    if tera_type is None:
        try:
            v = getattr(pokemon, "_terastallized_type", None)
            if isinstance(v, PokemonType) and v in _TYPE_INDEX:
                tera_type = v
        except Exception:
            pass
    if tera_type is None:
        try:
            details = getattr(pokemon, "_last_details", None) or getattr(pokemon, "_details", None) or getattr(pokemon, "details", None)
            if isinstance(details, str) and "tera:" in details.lower():
                m = re.search(r"tera:\s*([A-Za-z]+)", details, re.IGNORECASE)
                if m:
                    try:
                        cand = PokemonType.from_name(m.group(1))
                        if cand in _TYPE_INDEX:
                            tera_type = cand
                    except Exception:
                        pass
        except Exception:
            pass
    if tera_type is None:
        try:
            req = getattr(pokemon, "_last_request", None)
            if isinstance(req, dict):
                if "teraType" in req and req["teraType"]:
                    try:
                        cand = PokemonType.from_name(str(req["teraType"]))
                        if cand in _TYPE_INDEX:
                            tera_type = cand
                    except Exception:
                        pass
                d = req.get("details", "")
                if isinstance(d, str) and "tera:" in d.lower():
                    m = re.search(r"tera:\s*([A-Za-z]+)", d, re.IGNORECASE)
                    if m:
                        try:
                            cand = PokemonType.from_name(m.group(1))
                            if cand in _TYPE_INDEX:
                                tera_type = cand
                        except Exception:
                            pass
        except Exception:
            pass
    if tera_type is not None and tera_type in _TYPE_INDEX:
        vec[_TYPE_INDEX[tera_type]] = 1.0
    return vec

def _is_terastallized(pokemon) -> float:
    if pokemon is None:
        return 0.0
    return 1.0 if getattr(pokemon, "is_terastallized", False) else 0.0


def _can_tera_now(battle) -> float:
    can_tera = getattr(battle, "can_tera", None)
    return 1.0 if can_tera else 0.0

def _team_used_tera(team: dict) -> float:
    return 1.0 if any(getattr(mon, "is_terastallized", False) for mon in team.values()) else 0.0

# ---------------- Новые хелперы для флагов приёмов ----------------

def _move_boost_flags(move, kind: str = "own") -> np.ndarray:
    """5 флагов atk/def/spa/spd/spe: 1 если мув бустит свой (kind=own) или дропает чужой (kind=opp)."""
    vec = np.zeros(5, dtype=np.float32)
    if move is None:
        return vec
    try:
        boosts = None
        self_boost = None
        entry = getattr(move, "entry", None)
        # poke-env properties
        try:
            boosts = move.boosts  # target boosts
        except Exception:
            boosts = None
        try:
            self_boost = move.self_boost
        except Exception:
            self_boost = None
        # fallback to entry dict
        if entry is not None:
            if boosts is None:
                boosts = entry.get("boosts")
            if self_boost is None:
                # self может быть в entry["self"] или entry["selfBoost"]
                if "self" in entry and isinstance(entry["self"], dict):
                    self_boost = entry["self"].get("boosts")
                if self_boost is None and "selfBoost" in entry:
                    self_boost = entry["selfBoost"].get("boosts") if isinstance(entry["selfBoost"], dict) else entry["selfBoost"]
            # also check secondaries for boost flags
            sec_boosts = []
            for sec in getattr(move, "secondary", []) or []:
                if isinstance(sec, dict):
                    if "boosts" in sec:
                        # need to distinguish self vs opp — if sec has "self" key, it's self
                        if "self" in sec and isinstance(sec["self"], dict) and "boosts" in sec["self"]:
                            sec_boosts.append(("own", sec["self"]["boosts"]))
                        else:
                            sec_boosts.append(("opp", sec["boosts"]))
                    if "self" in sec and isinstance(sec["self"], dict) and "boosts" in sec["self"] and "boosts" not in sec:
                        # already handled
                        pass
            # merge secondary opp boosts into boosts for opp detection
            # for own we also check sec_boosts own
            if kind == "own":
                # self_boost + secondary self
                candidates = []
                if self_boost:
                    candidates.append(self_boost)
                for k, b in sec_boosts:
                    if k == "own":
                        candidates.append(b)
                for b in candidates:
                    if not isinstance(b, dict):
                        continue
                    for i, key in enumerate(_BOOST_KEYS):
                        if b.get(key, 0) > 0:
                            vec[i] = 1.0
            else:  # opp
                candidates = []
                if boosts:
                    candidates.append(boosts)
                for k, b in sec_boosts:
                    if k == "opp":
                        candidates.append(b)
                # also check entry secondaries directly if secondary property empty
                if not candidates and entry is not None:
                    for sec in entry.get("secondaries", []) or []:
                        if "boosts" in sec and isinstance(sec["boosts"], dict):
                            # assume opp if no self wrapper
                            if "self" not in sec:
                                candidates.append(sec["boosts"])
                for b in candidates:
                    if not isinstance(b, dict):
                        continue
                    for i, key in enumerate(_BOOST_KEYS):
                        if b.get(key, 0) < 0:
                            vec[i] = 1.0
        else:
            # no entry, use what we have
            if kind == "own" and self_boost:
                for i, key in enumerate(_BOOST_KEYS):
                    if self_boost.get(key, 0) > 0:
                        vec[i] = 1.0
            if kind == "opp" and boosts:
                for i, key in enumerate(_BOOST_KEYS):
                    if boosts.get(key, 0) < 0:
                        vec[i] = 1.0
    except Exception:
        pass
    return vec

def _move_hazard_clear_flags(move) -> np.ndarray:
    vec = np.zeros(2, dtype=np.float32)  # [own, opp]
    if move is None:
        return vec
    try:
        mid = getattr(move, "id", "") or getattr(move, "_id", "")
        mid = mid.lower().replace(" ", "").replace("-", "")
        if mid in _HAZARD_CLEAR_MAP:
            vec[0], vec[1] = _HAZARD_CLEAR_MAP[mid]
    except Exception:
        pass
    return vec

def _move_heal_pct(move) -> float:
    if move is None:
        return 0.0
    try:
        # poke-env Move.heal is 0..1
        h = getattr(move, "heal", 0.0)
        if isinstance(h, (int, float)) and h > 0:
            return float(np.clip(h, 0, 1))
        # also check drain
        d = getattr(move, "drain", 0.0)
        if isinstance(d, (int, float)) and d > 0:
            # drain 0.5 = 50% of damage, treat as heal flag
            return float(np.clip(d, 0, 1))
        # fallback to entry
        entry = getattr(move, "entry", {}) or {}
        if "heal" in entry:
            heal = entry["heal"]
            if isinstance(heal, (list, tuple)) and len(heal) == 2:
                return float(heal[0] / heal[1]) if heal[1] else 0.0
            if isinstance(heal, (int, float)):
                return float(heal)
        if "drain" in entry:
            drain = entry["drain"]
            if isinstance(drain, (list, tuple)) and len(drain) == 2:
                return float(drain[0] / drain[1]) if drain[1] else 0.0
    except Exception:
        pass
    return 0.0

def _move_status_prob(move) -> float:
    if move is None:
        return 0.0
    try:
        status = getattr(move, "status", None)
        if status is not None:
            return 1.0
        entry = getattr(move, "entry", {}) or {}
        if "status" in entry and entry["status"]:
            return 1.0
        max_prob = 0.0
        for sec in getattr(move, "secondary", []) or []:
            if not isinstance(sec, dict):
                continue
            if "status" in sec:
                chance = sec.get("chance", 100)
                try:
                    prob = float(chance) / 100.0 if chance > 1 else float(chance)
                except Exception:
                    prob = 1.0
                max_prob = max(max_prob, prob)
        if max_prob == 0.0 and entry:
            for sec in entry.get("secondaries", []) or []:
                if "status" in sec:
                    chance = sec.get("chance", 100)
                    try:
                        prob = float(chance) / 100.0 if chance > 1 else float(chance)
                    except Exception:
                        prob = 1.0
                    max_prob = max(max_prob, prob)
        return float(np.clip(max_prob, 0, 1))
    except Exception:
        return 0.0

def _move_priority(move) -> float:
    if move is None:
        return 0.0
    try:
        p = getattr(move, "priority", 0)
        # normalize -7..+5 -> -1..0.71, clip to -1..1
        return float(np.clip(p / 7.0, -1, 1))
    except Exception:
        return 0.0

def _move_stab_flag(move, pokemon) -> float:
    if move is None or pokemon is None:
        return 0.0
    try:
        mtype = getattr(move, "type", None)
        if mtype is None:
            return 0.0
        # tera stab not counted here (would need tera_type), just base types
        if mtype == getattr(pokemon, "type_1", None) or mtype == getattr(pokemon, "type_2", None):
            return 1.0
        # also check tera if terastallized
        tera = getattr(pokemon, "tera_type", None) or getattr(pokemon, "_terastallized_type", None)
        if tera is not None and mtype == tera:
            return 1.0
    except Exception:
        pass
    return 0.0

def _move_recoil_pct(move) -> float:
    if move is None:
        return 0.0
    try:
        r = getattr(move, "recoil", 0.0)
        if isinstance(r, (int,float)) and r>0:
            return float(np.clip(r,0,1))
        entry = getattr(move, "entry", {}) or {}
        if "recoil" in entry:
            rec = entry["recoil"]
            if isinstance(rec, (list,tuple)) and len(rec)==2 and rec[1]:
                return float(rec[0]/rec[1])
    except Exception:
        pass
    return 0.0

def _move_phaze_flag(move) -> float:
    if move is None:
        return 0.0
    try:
        if getattr(move, "force_switch", False):
            return 1.0
        mid = (getattr(move,"id","") or "").lower()
        if mid in _PHAZE_MOVES:
            return 1.0
        entry = getattr(move,"entry",{}) or {}
        if entry.get("forceSwitch"):
            return 1.0
    except Exception:
        pass
    return 0.0

def _base_stats_vec(mon, fusion_entry: dict | None = None) -> np.ndarray:
    """6 base stats normalized 0..1 (hp/atk/def/spa/spd/spe /255). Для fusion берём из чата если есть."""
    vec = np.zeros(6, dtype=np.float32)
    if mon is None:
        return vec
    try:
        stats = None
        if fusion_entry and "base_stats" in fusion_entry:
            stats = fusion_entry["base_stats"]
        else:
            stats = getattr(mon, "base_stats", None)
        if stats is None:
            return vec
        # stats may be dict with keys hp,atk,def,spa,spd,spe or at,df etc.
        # normalize standard poke_env keys
        keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        vals = []
        for k in keys:
            v = stats.get(k, stats.get(k.upper(), 0))
            if v is None:
                v = 0
            vals.append(float(v) / 255.0)
        vec = np.array(vals, dtype=np.float32)
    except Exception:
        pass
    return vec

def _weakness_score(mon, opp_active, type_chart) -> float:
    """0..1: max effectiveness of opp_active vs mon. 1.0 = x1, 2.0->0.5 normalized as (mult-1)/3 capped? We use 1 for弱, 0 for neutral/resist."""
    if mon is None or opp_active is None:
        return 0.0
    atk_types = [t for t in (opp_active.type_1, opp_active.type_2) if t is not None]
    if not atk_types:
        return 0.0
    max_mult = 1.0
    for atk in atk_types:
        try:
            mult = atk.damage_multiplier(mon.type_1, mon.type_2, type_chart=type_chart)
        except Exception:
            mult = 1.0
        max_mult = max(max_mult, mult)
    # map 1->0, 2->0.5, 4->1
    if max_mult >= 4:
        return 1.0
    if max_mult >= 2:
        return 0.5 + (max_mult - 2) * 0.25  # 2->0.5, 4->1
    if max_mult > 1:
        return (max_mult - 1) * 0.5
    return 0.0

def _bench_moves_vec(mon, opp_active, type_chart) -> np.ndarray:
    """5 dims per reserve: [n_revealed/4, avg_bp/100, max_eff/4, has_heal, max_status_prob]"""
    vec = np.zeros(5, dtype=np.float32)
    if mon is None or mon.fainted:
        return vec
    try:
        moves = list(getattr(mon, "moves", {}).values())
        if not moves:
            return vec
        n = len(moves)
        vec[0] = n / 4.0
        # avg base power
        bps = [getattr(m, "base_power", 0) or 0 for m in moves]
        # filter 0 (status moves)
        if bps:
            # avg over non-zero? use all
            vec[1] = float(np.mean([b for b in bps if b > 0]) / 100.0) if any(b > 0 for b in bps) else 0.0
        # max effectiveness vs opp_active if opp known
        if opp_active is not None:
            max_eff = 1.0
            for m in moves:
                try:
                    eff = m.type.damage_multiplier(opp_active.type_1, opp_active.type_2, type_chart=type_chart) if getattr(m, "type", None) else 1.0
                    max_eff = max(max_eff, eff)
                except Exception:
                    pass
            vec[2] = float(np.clip(max_eff / 4.0, 0, 1))
        # has_heal
        has_heal = any(_move_heal_pct(m) > 0 for m in moves)
        vec[3] = 1.0 if has_heal else 0.0
        # max status prob
        max_sp = max([_move_status_prob(m) for m in moves], default=0.0)
        vec[4] = float(max_sp)
    except Exception:
        pass
    return vec

def _trick_room_flag(battle) -> float:
    try:
        # poke_env: battle.fields may contain pseudo weather? check side conditions and battle attribute
        if getattr(battle, "trick_room", False):
            return 1.0
        # fallback: check in fields dict keys as string
        for k in list(getattr(battle, "fields", {}).keys()) + list(getattr(battle, "pseudo_weather", {}).keys() if hasattr(battle, "pseudo_weather") else []):
            n = getattr(k, "name", str(k)).lower()
            if "trick" in n:
                return 1.0
        # check side_conditions for trick room
        for k in battle.side_conditions.keys():
            if "trick" in getattr(k, "name", str(k)).lower():
                return 1.0
        for k in battle.opponent_side_conditions.keys():
            if "trick" in getattr(k, "name", str(k)).lower():
                return 1.0
    except Exception:
        pass
    return 0.0

def _tailwind_flags(side_conditions: dict) -> float:
    try:
        # SideCondition.TAILWIND if exists
        for k in side_conditions.keys():
            n = getattr(k,"name",str(k)).lower()
            if "tailwind" in n:
                return 1.0
    except Exception:
        pass
    return 0.0

def _screens_vec(side_conditions: dict) -> np.ndarray:
    return np.array([1.0 if sc in side_conditions else 0.0 for sc in _SCREEN_SIDE], dtype=np.float32)

def _bench_item_flag(mon) -> float:
    if mon is None or mon.fainted:
        return 0.0
    try:
        return 1.0 if getattr(mon, "item", None) else 0.0
    except Exception:
        return 0.0

_RESERVE_SLOT_SIZE = len(_TYPE_LIST) + 1 + len(_STATUSES) + 6 + 1 + 5 + 1  # типы + HP + статус + base_stats(6) + weakness(1) + bench_moves(5) + item_flag(1)


def _reserve_slot_vec(mon, opp_active=None, type_chart=None, fusion_entry: dict | None = None) -> np.ndarray:
    type_vec = _type_multi_hot(mon)
    hp = 0.0 if mon.fainted else mon.current_hp_fraction
    status_vec = _status_one_hot(mon.status)
    base_vec = _base_stats_vec(mon, fusion_entry)
    weak = np.array([_weakness_score(mon, opp_active, type_chart) if opp_active is not None and type_chart is not None else 0.0], dtype=np.float32)
    moves_vec = _bench_moves_vec(mon, opp_active, type_chart)
    item_flag = np.array([_bench_item_flag(mon)], dtype=np.float32)
    return np.concatenate([type_vec, [hp], status_vec, base_vec, weak, moves_vec, item_flag]).astype(np.float32)


def _bench_vec(team: dict, opp_active=None, type_chart=None, fusion_map: dict | None = None) -> np.ndarray:
    reserves = [mon for mon in team.values() if not mon.active]
    slots = []
    for i in range(MAX_RESERVES):
        if i < len(reserves):
            mon = reserves[i]
            # fusion entry per mon if available (key by species id)
            f_entry = None
            if fusion_map:
                # try species id
                try:
                    from poke_env.data.normalize import to_id_str
                    sid = to_id_str(getattr(mon, "species", "") or getattr(mon, "base_species", ""))
                    f_entry = fusion_map.get(sid)
                    if f_entry is None:
                        # try base_species
                        sid2 = to_id_str(getattr(mon, "base_species", ""))
                        f_entry = fusion_map.get(sid2)
                except Exception:
                    pass
            slots.append(_reserve_slot_vec(mon, opp_active, type_chart, f_entry))
        else:
            slots.append(np.zeros(_RESERVE_SLOT_SIZE, dtype=np.float32))
    return np.concatenate(slots)

def _vulnerability_frac(reserves: list, opponent_active, type_chart) -> float:
    """Доля живых резервных покемонов, получающих x2+ от текущих типов активного оппонента."""
    if opponent_active is None:
        return 0.0
    alive = [m for m in reserves if not m.fainted]
    if not alive:
        return 0.0
    atk_types = [t for t in (opponent_active.type_1, opponent_active.type_2) if t is not None]
    if not atk_types:
        return 0.0
    vulnerable = 0
    for mon in alive:
        max_mult = 1.0
        for atk in atk_types:
            try:
                mult = atk.damage_multiplier(mon.type_1, mon.type_2, type_chart=type_chart)
            except KeyError:
                mult = 1.0
            max_mult = max(max_mult, mult)
        if max_mult >= 2.0:
            vulnerable += 1
    return vulnerable / len(alive)

def _one_hot(value, options) -> np.ndarray:
    vec = np.zeros(len(options), dtype=np.float32)
    vec[options.index(value) if value in options else 0] = 1.0
    return vec


def _status_one_hot(status) -> np.ndarray:
    vec = np.zeros(len(_STATUSES), dtype=np.float32)
    idx = _STATUSES.index(status) if status in _STATUSES else 0
    vec[idx] = 1.0
    return vec


def _volatile_vec(pokemon) -> np.ndarray:
    if pokemon is None:
        return np.zeros(len(_VOLATILES), dtype=np.float32)
    return np.array([1.0 if v in pokemon.effects else 0.0 for v in _VOLATILES], dtype=np.float32)


def _boosts_vec(boosts: dict) -> np.ndarray:
    keys = ["atk", "def", "spa", "spd", "spe"]
    return np.array([boosts.get(k, 0) / 6.0 for k in keys], dtype=np.float32)


def _item_vec(pokemon) -> np.ndarray:
    if pokemon is None or not pokemon.item:
        return np.zeros(len(_KEY_ITEMS) + 1, dtype=np.float32)
    item_flags = [1.0 if pokemon.item == name else 0.0 for name in _KEY_ITEMS]
    return np.array([1.0] + item_flags, dtype=np.float32)


def _hazards(side_conditions: dict) -> np.ndarray:
    return np.array(
        [
            1.0 if SideCondition.STEALTH_ROCK in side_conditions else 0.0,
            side_conditions.get(SideCondition.SPIKES, 0) / 3.0,
            side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0,
            1.0 if SideCondition.STICKY_WEB in side_conditions else 0.0,
        ],
        dtype=np.float32,
    )


def _switch_summary(team: dict) -> np.ndarray:
    reserves = [mon for mon in team.values() if not mon.active]
    if not reserves:
        return np.zeros(2, dtype=np.float32)
    alive = [mon for mon in reserves if not mon.fainted]
    avg_hp = float(np.mean([mon.current_hp_fraction for mon in alive])) if alive else 0.0
    alive_frac = len(alive) / max(len(reserves), 1)
    return np.array([avg_hp, alive_frac], dtype=np.float32)


def _speed_range_estimate(base_spe: int, level: int) -> tuple[int, int]:
    lo = int(int((2 * base_spe) * level / 100 + 5) * 0.9)
    hi = int(int((2 * base_spe + 31 + 63) * level / 100 + 5) * 1.1)
    return lo, hi


def _get_speed_info(pokemon, fusion_entry: dict | None) -> tuple[float, float]:
    if fusion_entry and "speed_range" in fusion_entry:
        lo, hi = fusion_entry["speed_range"]
    else:
        base_spe = (
            fusion_entry["base_stats"]["spe"]
            if fusion_entry and "base_stats" in fusion_entry
            else pokemon.base_stats["spe"]
        )
        lo, hi = _speed_range_estimate(base_spe, pokemon.level)
    return lo / 300.0, hi / 300.0


def _move_wasted_flag(move, battle) -> float:
    opp = battle.opponent_active_pokemon
    move_id = move.id

    if move_id in _WEATHER_MOVES and _WEATHER_MOVES[move_id] in battle.weather:
        return 1.0
    if move_id in _TERRAIN_MOVES and _TERRAIN_MOVES[move_id] in battle.fields:
        return 1.0
    if move_id == "substitute" and Effect.SUBSTITUTE in (battle.active_pokemon.effects if battle.active_pokemon else {}):
        return 1.0
        
    if move_id in _HAZARD_MOVES:
        condition, max_layers = _HAZARD_MOVES[move_id]
        current = battle.opponent_side_conditions.get(condition, 0)
        current = 1 if current is True else (0 if current is False else current)
        return 1.0 if current >= max_layers else 0.0

    if move_id in _SCREEN_MOVES:
        return 1.0 if _SCREEN_MOVES[move_id] in battle.side_conditions else 0.0

    if move.status is not None and opp is not None and opp.status is not None:
        return 1.0

    if opp is not None and opp.ability is not None:
        immune_type = _IMMUNITY_ABILITIES.get(opp.ability)
        if immune_type is not None and move.type is not None and move.type.name.lower() == immune_type:
            return 1.0

    if opp is not None and opp.item == "airballoon" and move.type is not None:
        if move.type.name.lower() == "ground":
            return 1.0
    return 0.0


def embed_battle_with_fusion(battle, our_fusion, opp_fusion, our_protected_last_turn=0.0, opp_protected_last_turn=0.0, our_team_fusions: dict | None = None, opp_team_fusions: dict | None = None):
    from poke_env.data import GenData

    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    moves_wasted = np.zeros(4, dtype=np.float32)
    moves_accuracy = np.ones(4, dtype=np.float32)
    moves_pp_frac = np.ones(4, dtype=np.float32)
    moves_boost_own = np.zeros((4, 5), dtype=np.float32)
    moves_drop_opp = np.zeros((4, 5), dtype=np.float32)
    moves_hazard_clear = np.zeros((4, 2), dtype=np.float32)
    moves_heal = np.zeros(4, dtype=np.float32)
    moves_status_prob = np.zeros(4, dtype=np.float32)
    moves_priority = np.zeros(4, dtype=np.float32)
    moves_stab = np.zeros(4, dtype=np.float32)
    moves_recoil = np.zeros(4, dtype=np.float32)
    moves_phaze = np.zeros(4, dtype=np.float32)
    type_chart = GenData.from_gen(battle.gen).type_chart

    for i, move in enumerate(battle.available_moves):
        moves_base_power[i] = move.base_power / 100
        raw_acc = move.accuracy
        if raw_acc is True:
            moves_accuracy[i] = 1.0
        elif raw_acc is None:
            moves_accuracy[i] = 1.0
        else:
            moves_accuracy[i] = raw_acc / 100.0 if raw_acc > 1.0 else raw_acc
        moves_pp_frac[i] = move.current_pp / max(move.max_pp, 1)
        moves_wasted[i] = _move_wasted_flag(move, battle)
        if battle.opponent_active_pokemon is not None:
            try:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=type_chart,
                )
            except KeyError:
                moves_dmg_multiplier[i] = 1.0
        moves_boost_own[i] = _move_boost_flags(move, "own")
        moves_drop_opp[i] = _move_boost_flags(move, "opp")
        moves_hazard_clear[i] = _move_hazard_clear_flags(move)
        moves_heal[i] = _move_heal_pct(move)
        moves_status_prob[i] = _move_status_prob(move)
        moves_priority[i] = _move_priority(move)
        moves_stab[i] = _move_stab_flag(move, battle.active_pokemon)
        moves_recoil[i] = _move_recoil_pct(move)
        moves_phaze[i] = _move_phaze_flag(move)

    fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
    fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
    our_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
    opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0

    our_status = _status_one_hot(battle.active_pokemon.status if battle.active_pokemon else None)
    opp_status = _status_one_hot(battle.opponent_active_pokemon.status if battle.opponent_active_pokemon else None)
    our_hazards = _hazards(battle.side_conditions)
    opp_hazards = _hazards(battle.opponent_side_conditions)
    our_switches = _switch_summary(battle.team)
    opp_switches = _switch_summary(battle.opponent_team)
    our_boosts = _boosts_vec(battle.active_pokemon.boosts if battle.active_pokemon else {})
    opp_boosts = _boosts_vec(battle.opponent_active_pokemon.boosts if battle.opponent_active_pokemon else {})
    weather_vec = _one_hot(next(iter(battle.weather), None) if battle.weather else None, _WEATHERS)
    field_vec = _one_hot(next(iter(battle.fields), None) if battle.fields else None, _FIELDS)

    our_spe_lo, our_spe_hi = _get_speed_info(battle.active_pokemon, our_fusion) if battle.active_pokemon else (0.0, 0.0)
    opp_spe_lo, opp_spe_hi = (
        _get_speed_info(battle.opponent_active_pokemon, opp_fusion) if battle.opponent_active_pokemon else (0.0, 0.0)
    )

    our_volatiles = _volatile_vec(battle.active_pokemon)
    opp_volatiles = _volatile_vec(battle.opponent_active_pokemon)
    our_item = _item_vec(battle.active_pokemon)
    opp_item = _item_vec(battle.opponent_active_pokemon)
    speed_advantage = 1.0 if our_spe_lo > opp_spe_hi else (-1.0 if opp_spe_lo > our_spe_hi else 0.0)
    
    our_revealed = _revealed_moves_frac(battle.active_pokemon)
    opp_revealed = _revealed_moves_frac(battle.opponent_active_pokemon)

    our_semi_invuln = _is_semi_invuln_or_charging(battle.active_pokemon)
    opp_semi_invuln = _is_semi_invuln_or_charging(battle.opponent_active_pokemon)

    our_sub_damaged = _substitute_damaged(battle.active_pokemon)
    opp_sub_damaged = _substitute_damaged(battle.opponent_active_pokemon)

    our_restricted = _is_move_restricted(battle)
    our_reserves = [mon for mon in battle.team.values() if not mon.active]
    opp_reserves = [mon for mon in battle.opponent_team.values() if not mon.active]

    our_bench = _bench_vec(battle.team, battle.opponent_active_pokemon, type_chart, our_team_fusions)
    opp_bench = _bench_vec(battle.opponent_team, battle.active_pokemon, type_chart, opp_team_fusions)

    our_vulnerability = _vulnerability_frac(our_reserves, battle.opponent_active_pokemon, type_chart)
    opp_vulnerability = _vulnerability_frac(opp_reserves, battle.active_pokemon, type_chart)
    our_can_tera_now = _can_tera_now(battle)
    our_used_tera = _team_used_tera(battle.team)
    opp_used_tera = _team_used_tera(battle.opponent_team)
    our_tera_type = _tera_type_vec(battle.active_pokemon)
    trick_room = _trick_room_flag(battle)
    our_tailwind = _tailwind_flags(battle.side_conditions)
    opp_tailwind = _tailwind_flags(battle.opponent_side_conditions)
    our_screens = _screens_vec(battle.side_conditions)
    opp_screens = _screens_vec(battle.opponent_side_conditions)
    moves_boost_own_flat = moves_boost_own.flatten()
    moves_drop_opp_flat = moves_drop_opp.flatten()
    moves_hazard_clear_flat = moves_hazard_clear.flatten()
    obs = np.concatenate(
        [
            moves_base_power, moves_dmg_multiplier, moves_wasted, moves_accuracy, moves_pp_frac,
            moves_boost_own_flat, moves_drop_opp_flat, moves_hazard_clear_flat, moves_heal, moves_status_prob,
            moves_priority, moves_stab, moves_recoil, moves_phaze,
            [fainted_mon_team, fainted_mon_opponent, our_hp, opp_hp],
            our_status, opp_status, our_hazards, opp_hazards, our_switches, opp_switches,
            our_boosts, opp_boosts, weather_vec, field_vec,
            [trick_room, our_tailwind, opp_tailwind], our_screens, opp_screens,
            [speed_advantage],
            [our_revealed, opp_revealed],
            [our_semi_invuln, opp_semi_invuln],
            [our_sub_damaged, opp_sub_damaged],
            [our_restricted],
            our_volatiles, opp_volatiles,
            our_item, opp_item,
            our_bench, opp_bench,
            [our_vulnerability, opp_vulnerability],
            [our_can_tera_now, our_used_tera, opp_used_tera],
            our_tera_type,
            [our_protected_last_turn, opp_protected_last_turn]
        ],
        dtype=np.float32,
    )
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs
