import re
import numpy as np
from poke_env.battle import Field, SideCondition, Status, Weather
from poke_env.battle import Effect

try:
    from .type_utils import damage_multiplier_safe
except ImportError:  # запуск модуля вне пакета
    from type_utils import damage_multiplier_safe

try:
    from .fusion_types import effective_types, is_fusion_format
except ImportError:  # запуск модуля вне пакета
    from fusion_types import effective_types, is_fusion_format


def _eff_types(mon):
    """Типы мон с учётом фьюжна (см. fusion_types.py) — дексовые только как последний фолбэк."""
    t1, t2, _ = effective_types(mon)
    return t1, t2

try:
    from .damage import (
        DAMAGE_BLOCK_SIZE, EFFECT_FLAG_COUNT, FLAGS_BASE, MIRROR_BASE, OPP_MOVE_SLOTS,
        OUR_TEAM_SLOTS, TEAM_BASE, DamageContext, cached_best_move_damage, damage_frac,
        estimate_damage, opponent_effect_flags, prepare_mon, team_damage_matrix, team_slots,
        their_known_moves_damage,
    )
except ImportError:  # запуск модуля вне пакета
    from damage import (
        DAMAGE_BLOCK_SIZE, EFFECT_FLAG_COUNT, FLAGS_BASE, MIRROR_BASE, OPP_MOVE_SLOTS,
        OUR_TEAM_SLOTS, TEAM_BASE, DamageContext, cached_best_move_damage, damage_frac,
        estimate_damage, opponent_effect_flags, prepare_mon, team_damage_matrix, team_slots,
        their_known_moves_damage,
    )

# Размерность obs объявлена в agents/dims.py (без импортов) — см. комментарий там о цикле
# config -> action_space -> features. Никаких fallback-констант: расхождение ловят
# obs_layout()/embed_battle_with_fusion сразу, а не на этапе загрузки чекпоинта.
try:
    from .dims import N_FEATURES
except ImportError:  # запуск модуля вне пакета
    from dims import N_FEATURES       # type: ignore

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
_KEY_ABILITIES = [
    "intimidate","unaware","magicbounce","regenerator","protean","libero",
    "sheerforce","contrary","speedboost","levitate","flashfire","voltabsorb",
    "waterabsorb","stormdrain","sapsipper","prankster","guts","chlorophyll","swiftswim","sandrush"
]
_ABILITY_INDEX = {a:i for i,a in enumerate(_KEY_ABILITIES)}
_PHAZE_MOVES = {"roar","whirlwind","dragontail","circlethrow","yawn"}
# Полная оценка приёмов (см. AUDIT 33): 7 статов, включая accuracy/evasion — бусты по ним
# в старых 5-флаговых блоках не были видны вовсе (Hone Claws/Sand Attack и т.п.).
_BOOST_KEYS_EXT = _BOOST_KEYS + ["accuracy", "evasion"]
_BOOST_KEY_INDEX = {k: i for i, k in enumerate(_BOOST_KEYS_EXT)}
# Хазарды, которые приём СТАВИТ (sideCondition в данных Showdown): sr/spikes/tspikes/web.
_HAZARD_SET_KEYS = ["stealthrock", "spikes", "toxicspikes", "stickyweb"]
# Флаги флагов-взаимодействий со способностями: Bulletproof/Sharpness/Strong Jaw/Iron Fist/
# Mega Launcher/Wind Rider/Dancer (порошковые — «на кого действует Powder»).
_ABILITY_MOVE_FLAGS = ["bullet", "slicing", "bite", "punch", "pulse", "wind", "dance", "powder"]
# Приёмы, накладывающие статус НА СЕБЯ, у которых в данных нет ни self.status, ни secondary
# (Rest применяет сон через onHit, а не через поля данных).
_SELF_STATUS_MOVES = {"rest": 1.0}
# Приёмы с зарядкой (страховка, если в данных нет flags.charge — например, кастомные моды)
_CHARGE_MOVES = {
    "solarbeam", "solarblade", "fly", "dig", "dive", "bounce", "phantomforce", "shadowforce",
    "skyattack", "skullbash", "razorwind", "meteorbeam", "electroshot", "geomancy",
    "freezeshock", "iceburn", "iceball",
}


def _chance_frac(sec: dict) -> float:
    """chance из secondary-эффекта как доля 0..1 (по умолчанию 100%)."""
    try:
        c = sec.get("chance", 100)
        c = float(c)
        return float(np.clip(c / 100.0, 0.0, 1.0)) if c > 1 else float(np.clip(c, 0.0, 1.0))
    except Exception:
        return 1.0


def _move_boost_magnitude(move, kind: str = "own") -> np.ndarray:
    """Сколько стадий бустов/дебаффов даёт приём, по 7 статам (atk..spe + accuracy/evasion).

    kind="own"  — изменения НАШИХ статов (в т.ч. отрицательные: Close Combat/Overheat),
    kind="opp"  — изменения статов СОПЕРНИКА (обычно отрицательные: Memento/Growl).

    Почему магнитуды, а не флаги: флаг говорил только «какой-то буст в этот стат», а модель
    не могла отличить Swords Dance (+2 atk) от Howl (+1 atk) и не видела, что Close Combat
    СНИЖАЕТ свои def/spd. Secondary-эффекты взвешиваются их шансом (ожидаемое изменение).
    """
    vec = np.zeros(len(_BOOST_KEYS_EXT), dtype=np.float32)
    if move is None:
        return vec
    try:
        entry = getattr(move, "entry", {}) or {}
        # Цели «нашей» стороны: self (сам), allies/allySide/allyTeam (Howl, Tailwind, экраны).
        # Раньше проверялся только "self", и Howl (+1 atk союзникам) не давал признака вовсе.
        target_own_side = str(entry.get("target", "")).lower() in (
            "self", "allies", "allyside", "allyteam")

        def _add(boosts, chance=1.0):
            if not isinstance(boosts, dict):
                return
            for key, val in boosts.items():
                idx = _BOOST_KEY_INDEX.get(str(key))
                if idx is None:
                    continue
                try:
                    vec[idx] += float(val) * float(chance)
                except Exception:
                    continue

        # прямые изменения статов цели: для self-targeting это НАШИ статы, иначе — соперника
        boosts = getattr(move, "boosts", None)
        if boosts is None:
            boosts = entry.get("boosts")
        if target_own_side:
            if kind == "own":
                _add(boosts)
        else:
            if kind == "opp":
                _add(boosts)
        # self.boosts — изменения НАШИХ статов самим приёмом (Close Combat, Overheat)
        if kind == "own":
            self_boost = getattr(move, "self_boost", None)
            if self_boost is None:
                self_entry = entry.get("self")
                if isinstance(self_entry, dict):
                    self_boost = self_entry.get("boosts")
                if self_boost is None and isinstance(entry.get("selfBoost"), dict):
                    self_boost = entry["selfBoost"].get("boosts")
            _add(self_boost)
        # secondary-эффекты (взвешены шансом)
        for sec in getattr(move, "secondary", []) or []:
            if not isinstance(sec, dict):
                continue
            ch = _chance_frac(sec)
            if kind == "own":
                own_sec = sec.get("self")
                if isinstance(own_sec, dict):
                    _add(own_sec.get("boosts"), ch)
            else:
                _add(sec.get("boosts"), ch)
        if kind == "opp" and not isinstance(move.secondary, list):
            for sec in entry.get("secondaries", []) or []:
                if isinstance(sec, dict) and "self" not in sec:
                    _add(sec.get("boosts"), _chance_frac(sec))
    except Exception:
        pass
    return vec


def _move_status_self_prob(move) -> float:
    """Шанс наложить статус НА СЕБЯ (Rest — 100% сна; rare self-secondary)."""
    if move is None:
        return 0.0
    try:
        mid = str(getattr(move, "id", "") or "").lower().replace(" ", "").replace("-", "")
        if mid in _SELF_STATUS_MOVES:
            return float(_SELF_STATUS_MOVES[mid])
        entry = getattr(move, "entry", {}) or {}
        self_entry = entry.get("self")
        if isinstance(self_entry, dict) and self_entry.get("status"):
            return 1.0
        best = 0.0
        for sec in getattr(move, "secondary", []) or []:
            if not isinstance(sec, dict):
                continue
            own_sec = sec.get("self")
            if isinstance(own_sec, dict) and own_sec.get("status"):
                best = max(best, _chance_frac(sec))
        if best == 0.0:
            for sec in entry.get("secondaries", []) or []:
                own_sec = sec.get("self") if isinstance(sec, dict) else None
                if isinstance(own_sec, dict) and own_sec.get("status"):
                    best = max(best, _chance_frac(sec))
        return float(np.clip(best, 0.0, 1.0))
    except Exception:
        return 0.0


def _move_charge_turns(move) -> float:
    """Сколько ходов приём заряжается (Solar Beam/Fly/Dig/Geomancy -> 1)."""
    if move is None:
        return 0.0
    try:
        entry = getattr(move, "entry", {}) or {}
        flags = entry.get("flags") or {}
        if flags.get("charge"):
            return 1.0
        if entry.get("charge"):
            return 1.0
        mid = str(getattr(move, "id", "") or "").lower().replace(" ", "").replace("-", "")
        if mid in _CHARGE_MOVES:
            return 1.0
    except Exception:
        pass
    return 0.0


def _move_recharge_flag(move) -> float:
    """Приём требует хода на перезарядку (Hyper Beam и родня)."""
    if move is None:
        return 0.0
    try:
        entry = getattr(move, "entry", {}) or {}
        flags = entry.get("flags") or {}
        if flags.get("recharge"):
            return 1.0
        self_entry = entry.get("self")
        if isinstance(self_entry, dict) and self_entry.get("volatileStatus") == "mustrecharge":
            return 1.0
    except Exception:
        pass
    return 0.0


def _move_revive_frac(move) -> float:
    """Возрождение союзного покемона: доля HP, с которой он возвращается (Revival Blessing)."""
    if move is None:
        return 0.0
    try:
        entry = getattr(move, "entry", {}) or {}
        if str(entry.get("slotCondition", "")).lower() == "revivalblessing":
            heal = entry.get("heal")
            if isinstance(heal, (list, tuple)) and len(heal) == 2 and heal[1]:
                return float(np.clip(heal[0] / heal[1], 0, 1))
            return 0.5
        mid = str(getattr(move, "id", "") or "").lower().replace(" ", "").replace("-", "")
        if mid == "revivalblessing":
            return 0.5
        if entry.get("revive"):
            try:
                return float(np.clip(float(entry["revive"]), 0, 1))
            except Exception:
                return 0.5
    except Exception:
        pass
    return 0.0


def _move_faint_user_flag(move) -> float:
    """Приём роняет СВОЕГО покемона (Explosion/Self-Destruct/Memento/Healing Wish)."""
    if move is None:
        return 0.0
    try:
        entry = getattr(move, "entry", {}) or {}
        sd = entry.get("selfdestruct")
        if sd in ("always", "ifHit", True) or sd == 1:
            return 1.0
        flags = entry.get("flags") or {}
        if flags.get("selfdestruct"):
            return 1.0
    except Exception:
        pass
    return 0.0


def _move_hazard_set_flags(move) -> np.ndarray:
    """Какие хазарды приём СТАВИТ: [stealthrock, spikes, toxicspikes, stickyweb]."""
    vec = np.zeros(len(_HAZARD_SET_KEYS), dtype=np.float32)
    if move is None:
        return vec
    try:
        entry = getattr(move, "entry", {}) or {}
        cond = str(entry.get("sideCondition", "") or "").lower()
        if not cond:
            mid = str(getattr(move, "id", "") or "").lower().replace(" ", "").replace("-", "")
            cond = mid if mid in _HAZARD_SET_KEYS else ""
        if cond in _HAZARD_SET_KEYS:
            vec[_HAZARD_SET_KEYS.index(cond)] = 1.0
    except Exception:
        pass
    return vec


def _move_ability_flags(move) -> np.ndarray:
    """Флаги, через которые приём взаимодействует со способностями (Bulletproof, Sharpness,
    Strong Jaw, Iron Fist, Mega Launcher, Wind Rider, Dancer, Powder)."""
    vec = np.zeros(len(_ABILITY_MOVE_FLAGS), dtype=np.float32)
    if move is None:
        return vec
    try:
        entry = getattr(move, "entry", {}) or {}
        flags = entry.get("flags") or {}
        for i, name in enumerate(_ABILITY_MOVE_FLAGS):
            if flags.get(name):
                vec[i] = 1.0
    except Exception:
        pass
    return vec


def _move_drain_pct(move) -> float:
    """Доля УРОНА, возвращаемая в HP (Giga Drain и родня); отдельно от прямого heal."""
    if move is None:
        return 0.0
    try:
        d = getattr(move, "drain", 0.0)
        if isinstance(d, (int, float)) and d > 0:
            return float(np.clip(d, 0, 1))
        entry = getattr(move, "entry", {}) or {}
        d = entry.get("drain")
        if isinstance(d, (list, tuple)) and len(d) == 2 and d[1]:
            return float(np.clip(d[0] / d[1], 0, 1))
        if isinstance(d, (int, float)) and d > 0:
            return float(np.clip(d, 0, 1))
    except Exception:
        pass
    return 0.0
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

# --- блок «типы противника против нашей команды» (см. _type_matchup_block) -------------
TYPE_MATCHUP_TEAM_SLOTS = 6      # слот 0 — активный, 1..5 — резервы в каноническом порядке
TYPE_MATCHUP_TYPE_SLOTS = 2      # у покемона максимум два типа
TYPE_MATCHUP_ROW_SIZE = 2 + TYPE_MATCHUP_TEAM_SLOTS     # [флаг типа, скаляр типа, 6 множителей]
TYPE_MATCHUP_BLOCK_SIZE = (
    len(_TYPE_LIST)                                          # типы нашего активного
    + TYPE_MATCHUP_TYPE_SLOTS * TYPE_MATCHUP_ROW_SIZE * TYPE_MATCHUP_TEAM_SLOTS
    + TYPE_MATCHUP_TEAM_SLOTS                                # флаги «тип подтверждён сервером»
)
# абсолютное смещение блока: он идёт в хвост obs, сразу за блоком урона. 715 — это
# MIN_PREFIX_OBS_DIM из training.py (импортировать нельзя: training импортирует features)
TYPE_MATCHUP_BASE = 715 + DAMAGE_BLOCK_SIZE


def _type_scalar(t) -> float:
    """Скаляр типа 0..1 (0 — типа нет), та же шкала, что у `_move_type_scalar` для наших приёмов."""
    if t is None or t not in _TYPE_INDEX:
        return 0.0
    return float(_TYPE_INDEX[t]) / max(1, len(_TYPE_LIST) - 1)


def _type_multi_hot(pokemon) -> np.ndarray:
    vec = np.zeros(len(_TYPE_LIST), dtype=np.float32)
    if pokemon is None:
        return vec
    for t in _eff_types(pokemon):
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
        pt1, pt2 = _eff_types(pokemon)
        if mtype == pt1 or mtype == pt2:
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

def _ability_vec(pokemon) -> np.ndarray:
    vec = np.zeros(len(_KEY_ABILITIES), dtype=np.float32)
    if pokemon is None:
        return vec
    try:
        ab = getattr(pokemon, "ability", None)
        if ab:
            ab = str(ab).lower().replace(" ", "").replace("-", "")
            if ab in _ABILITY_INDEX:
                vec[_ABILITY_INDEX[ab]] = 1.0
    except Exception:
        pass
    return vec

def _move_type_scalar(move) -> float:
    if move is None:
        return 0.0
    try:
        mtype = getattr(move, "type", None)
        if mtype is not None and mtype in _TYPE_INDEX:
            return float(_TYPE_INDEX[mtype] / max(1, len(_TYPE_LIST)-1))
        entry = getattr(move, "entry", {}) or {}
        tname = entry.get("type", "")
        if tname:
            try:
                cand = PokemonType.from_name(tname)
                if cand in _TYPE_INDEX:
                    return float(_TYPE_INDEX[cand] / max(1, len(_TYPE_LIST)-1))
            except Exception:
                pass
    except Exception:
        pass
    return 0.0

def _move_category_vec(move) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)  # PHYS, SPECIAL, STATUS
    if move is None:
        return vec
    try:
        cat = getattr(move, "category", None)
        if cat is not None:
            n = getattr(cat, "name", str(cat)).upper()
            if n == "PHYSICAL":
                vec[0]=1.0
            elif n == "SPECIAL":
                vec[1]=1.0
            elif n == "STATUS":
                vec[2]=1.0
            return vec
        entry = getattr(move, "entry", {}) or {}
        c = entry.get("category", "").upper()
        if c == "PHYSICAL":
            vec[0]=1.0
        elif c == "SPECIAL":
            vec[1]=1.0
        elif c == "STATUS":
            vec[2]=1.0
    except Exception:
        pass
    return vec

def _boosts_vec(boosts: dict) -> np.ndarray:
    keys = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    return np.array([boosts.get(k, 0) / 6.0 for k in keys], dtype=np.float32)

def _move_contact_flag(move) -> float:
    if move is None:
        return 0.0
    try:
        flags = getattr(move, "flags", set()) or set()
        if "contact" in flags:
            return 1.0
        entry = getattr(move, "entry", {}) or {}
        if "contact" in entry.get("flags", {}):
            return 1.0
        # fallback: check entry flags dict
        if isinstance(entry.get("flags"), set) and "contact" in entry["flags"]:
            return 1.0
        if isinstance(entry.get("flags"), dict) and entry["flags"].get("contact"):
            return 1.0
    except Exception:
        pass
    return 0.0

def _move_sound_flag(move) -> float:
    if move is None:
        return 0.0
    try:
        flags = getattr(move, "flags", set()) or set()
        if "sound" in flags:
            return 1.0
        entry = getattr(move, "entry", {}) or {}
        if "sound" in str(entry.get("flags", "")):
            return 1.0
        # check flags set
        eff = getattr(move, "flags", set())
        # flags may contain "sound"
        if any("sound" == f.lower() for f in eff):
            return 1.0
    except Exception:
        pass
    return 0.0

def _move_multihit_flag(move) -> float:
    if move is None:
        return 0.0
    try:
        n = getattr(move, "n_hit", (1,1))
        if isinstance(n, (tuple, list)) and len(n)==2:
            if n[0] != 1 or n[1] != 1:
                return 1.0
        entry = getattr(move, "entry", {}) or {}
        if "multihit" in entry:
            return 1.0
        if getattr(move, "expected_hits", 1) > 1.1:
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

def _actual_stats_vec(mon, fusion_entry: dict | None = None) -> np.ndarray:
    """6 реальных статов с учётом уровня / IV/EV. Нормируем /600 (макс ~714 HP, ~500 осталь)."""
    vec = np.zeros(6, dtype=np.float32)
    if mon is None:
        return vec
    try:
        base = None
        if fusion_entry and "base_stats" in fusion_entry:
            base = fusion_entry["base_stats"]
        else:
            base = getattr(mon, "base_stats", None)
        if base is None:
            return vec
        level = getattr(mon, "level", 100) or 100
        # random battles: IV 31, EV 85 *6 =510 total ~85 each, nature neutral 1.0
        IV = 31
        EV = 85
        def calc(b, is_hp=False):
            b = float(b)
            if is_hp:
                return int(((2*b + IV + EV/4)*level/100) + level + 10)
            else:
                return int(((2*b + IV + EV/4)*level/100 + 5) * 1.0)
        keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        vals = []
        for i, k in enumerate(keys):
            bv = base.get(k, base.get(k.upper(), 80)) or 80
            is_hp = (i==0)
            real = calc(bv, is_hp)
            # нормируем: HP /714 (max), остальные /500
            norm = real / 714.0 if is_hp else real / 500.0
            vals.append(float(np.clip(norm, 0, 1)))
        vec = np.array(vals, dtype=np.float32)
    except Exception:
        pass
    return vec

def _fusion_entry_for(mon, fusion_map: dict | None) -> dict | None:
    """Запись фьюжна для мон из карты СВОЕЙ стороны (см. fusion_parser.get_team_fusion_map).

    Ключ карты — вид, под которым запись положил парсер: это имя из `|switch|`/`|typechange`,
    а мод пишет там "+Тело" (то есть ключ = вид-«тело»). У покемона же из `details` виден
    только вид-«голова» (`Pokemon.species`), поэтому просто `map[mon.species]` не находил
    запись — статы скамейки молча падали на дексовые. Ищем по очереди: точный ключ по голове,
    ключ по телу из '+'-имени, составной "голова_тело" (не путает двух мон с одним телом).
    """
    if not fusion_map or mon is None:
        return None
    try:
        from poke_env.data.normalize import to_id_str
    except Exception:
        return None

    def _sid(v):
        try:
            return to_id_str(v) or None
        except Exception:
            return None

    head = _sid(getattr(mon, "species", "") or getattr(mon, "base_species", ""))
    entry = fusion_map.get(head) if head else None
    if entry is not None:
        return entry
    _head, body = fusion_types_pair(mon)   # fusion_pair возвращает (голова, тело)
    if body:
        if head:
            entry = fusion_map.get(f"{head}_{body}")
            if entry is not None:
                return entry
        entry = fusion_map.get(body)
        if entry is not None:
            return entry
    base = _sid(getattr(mon, "base_species", ""))
    if base and base != head:
        return fusion_map.get(base)
    return None


def fusion_types_pair(mon):
    """(голова, тело) фьюжна у мон — тонкая обёртка, чтобы не тянуть импорт в кучу мест."""
    try:
        from .fusion_types import fusion_pair
    except ImportError:
        from fusion_types import fusion_pair
    try:
        return fusion_pair(mon)
    except Exception:
        return None, None


def _weakness_score(mon, opp_active, type_chart) -> float:
    """0..1: max effectiveness of opp_active vs mon. 1.0 = x1, 2.0->0.5 normalized as (mult-1)/3 capped? We use 1 for弱, 0 for neutral/resist."""
    if mon is None or opp_active is None:
        return 0.0
    atk_types = [t for t in _eff_types(opp_active) if t is not None]
    if not atk_types:
        return 0.0
    mt1, mt2 = _eff_types(mon)
    max_mult = 1.0
    for atk in atk_types:
        # безопасный расчёт: неизвестный второй тип не маскирует иммунитет
        mult = damage_multiplier_safe(atk, mt1, mt2, type_chart)
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
                ot1, ot2 = _eff_types(opp_active)
                eff = damage_multiplier_safe(m.type, ot1, ot2, type_chart) if getattr(m, "type", None) else 1.0
                max_eff = max(max_eff, eff)
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
    # надёжно: poke_env Field.TRICK_ROOM или pseudo_weather TRICK_ROOM
    try:
        # 1) прямой атрибут
        if getattr(battle, "trick_room", False):
            return 1.0
        # 2) fields / pseudo_weather по enum
        for attr in ("fields", "pseudo_weather", "pseudoWeather"):
            d = getattr(battle, attr, None)
            if isinstance(d, dict):
                for k in d.keys():
                    n = getattr(k, "name", str(k)).lower().replace(" ", "").replace("_", "")
                    if n == "trickroom":
                        return 1.0
        # 3) fallback строковый поиск (оставляем как страховку)
        for src in (getattr(battle, "fields", {}), getattr(battle, "pseudo_weather", {}) if hasattr(battle, "pseudo_weather") else {}):
            try:
                for k in src.keys():
                    if "trick" in getattr(k, "name", str(k)).lower():
                        return 1.0
            except Exception:
                pass
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
    actual_vec = _actual_stats_vec(mon, fusion_entry)
    weak = np.array([_weakness_score(mon, opp_active, type_chart) if opp_active is not None and type_chart is not None else 0.0], dtype=np.float32)
    moves_vec = _bench_moves_vec(mon, opp_active, type_chart)
    item_flag = np.array([_bench_item_flag(mon)], dtype=np.float32)
    return np.concatenate([type_vec, [hp], status_vec, actual_vec, weak, moves_vec, item_flag]).astype(np.float32)


def canonical_reserves(team) -> list:
    """Резервные покемоны в КАНОНИЧЕСКОМ порядке (сортировка по species, тай-брейк — имя).

    Порядок `dict` в poke-env — это порядок выхода/превью, то есть произволен и меняется
    между боями: одинаковые состояния давали разные obs (8 dims на перестановку двух
    резервов в тесте, диапазон индексов 299..565 — это ровно bench-блок). В damage.py тот
    же принцип уже применён к слотам матриц (`team_slots`), здесь он оставлен недоисправленным.
    Имя (identify/ник) уникально внутри стороны, поэтому для двух покемонов одного вида
    порядок всё равно детерминирован.
    """
    reserves = [mon for mon in (team or {}).values()
                if mon is not None and not getattr(mon, "active", False)]
    return sorted(reserves, key=lambda m: (str(getattr(m, "species", "") or ""),
                                           str(getattr(m, "name", "") or "")))


def _bench_vec(team: dict, opp_active=None, type_chart=None, fusion_map: dict | None = None) -> np.ndarray:
    reserves = canonical_reserves(team)
    slots = []
    for i in range(MAX_RESERVES):
        if i < len(reserves):
            mon = reserves[i]
            # запись фьюжна: ключ карты — вид-"тело", у мон виден вид-"голова" (см. _fusion_entry_for)
            f_entry = _fusion_entry_for(mon, fusion_map)
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
    atk_types = [t for t in _eff_types(opponent_active) if t is not None]
    if not atk_types:
        return 0.0
    vulnerable = 0
    for mon in alive:
        max_mult = 1.0
        for atk in atk_types:
            # было `except KeyError: mult = 1.0` — теперь компонентный расчёт
            mt1, mt2 = _eff_types(mon)
            mult = damage_multiplier_safe(atk, mt1, mt2, type_chart)
            max_mult = max(max_mult, mult)
        if max_mult >= 2.0:
            vulnerable += 1
    return vulnerable / len(alive)

def _matchup_team_slots(active, team) -> list:
    """Слоты команды для матрицы типов: 0 — активный, 1..5 — резервы в каноническом порядке.

    Порядок совпадает с bench-блоком (там резервы без активного), поэтому «наш слот 0» — это
    ровно тот покемон, чьи признаки лежат в начале obs, а «наш слот k>0» — тот, что стоит
    на (k-1)-м месте bench-блока.
    """
    slots: list = [active]
    for mon in canonical_reserves(team):
        if mon is None or mon is active:
            continue
        if len(slots) >= TYPE_MATCHUP_TEAM_SLOTS:
            break
        slots.append(mon)
    while len(slots) < TYPE_MATCHUP_TEAM_SLOTS:
        slots.append(None)
    return slots[:TYPE_MATCHUP_TEAM_SLOTS]


def _type_matchup_block(battle, type_chart=None) -> np.ndarray:
    """Эффективность КАЖДОГО типа КАЖДОГО покемона соперника против комбинации типов КАЖДОГО
    нашего покемона — чтобы модель видела угрозу супер-эффективного удара ещё до того, как
    противник покажет приём.

    Раскладка (TYPE_MATCHUP_BLOCK_SIZE = 121, абсолютно с TYPE_MATCHUP_BASE = 870):

      [0:19)   multi-hot типов нашего активного (фьюжн-осознанно, см. _eff_types)
      [19:115) 12 строк по 8 (r = opp_slot * 2 + type_slot):
                 [0]     флаг: у этого покемона соперника есть тип в этом слоте
                 [1]     скаляр типа (0..1, как _move_type_scalar у наших приёмов)
                 [2:8]   множитель урона этого типа против наших слотов 0..5
      [115:121) 6 флагов по слотам соперника: тип подтверждён сервером (typechange/тера или
                обычный формат, где декс и есть правда), а не выведен формулой фьюжна

    Слоты соперника: 0 — активный, 1..5 — резервы в каноническом порядке, как в bench-блоке.
    Для ещё не виденных покемонов соперника неизвестного вида строка нулевая (флаг 0).
    """
    out = np.zeros(TYPE_MATCHUP_BLOCK_SIZE, dtype=np.float32)
    n_types = len(_TYPE_LIST)
    rows_base = n_types
    conf_base = n_types + TYPE_MATCHUP_TYPE_SLOTS * TYPE_MATCHUP_ROW_SIZE * TYPE_MATCHUP_TEAM_SLOTS

    our_slots = _matchup_team_slots(battle.active_pokemon, battle.team)
    opp_slots = _matchup_team_slots(battle.opponent_active_pokemon, battle.opponent_team)

    out[:n_types] = _type_multi_hot(battle.active_pokemon)
    our_types = [_eff_types(mon) if mon is not None else (None, None) for mon in our_slots]

    fusion_fmt = is_fusion_format(battle)
    for i, mon in enumerate(opp_slots):
        if mon is None:
            continue
        t1, t2, src = effective_types(mon, battle=battle)
        # сервер прислал тип (typechange/тера) либо формат не фьюжн — тогда декс и есть правда
        out[conf_base + i] = 1.0 if (str(src).startswith("server") or not fusion_fmt) else 0.0
        for j, atk in enumerate((t1, t2)):
            if atk is None:
                continue
            row = rows_base + (i * TYPE_MATCHUP_TYPE_SLOTS + j) * TYPE_MATCHUP_ROW_SIZE
            out[row] = 1.0
            out[row + 1] = _type_scalar(atk)
            for k, (mt1, mt2) in enumerate(our_types):
                if mt1 is None and mt2 is None:
                    continue
                out[row + 2 + k] = damage_multiplier_safe(atk, mt1, mt2, type_chart=type_chart)
    return out


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
    reserves = canonical_reserves(team)
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
    if move_id == "substitute":
        # уже есть кукла
        if Effect.SUBSTITUTE in (battle.active_pokemon.effects if battle.active_pokemon else {}):
            return 1.0
        # hp <=25% — не поставится (25% стоит кукла), wasted
        try:
            if battle.active_pokemon is not None and getattr(battle.active_pokemon, "current_hp_fraction", 1.0) <= 0.26:
                return 1.0
        except Exception:
            pass
    # хил при полном HP — wasted
    try:
        entry = getattr(move, "entry", {}) or {}
        is_heal = False
        if entry.get("heal", None) is not None:
            is_heal = True
        elif move_id in ("recover","roost","softboiled","morningsun","moonlight","synthesis","healorder","slackoff","milkdrink","swallow","rest","shoreup","strengthsap","wish","healingwish","lunardance","purify","lifedew","junglehealing"):
            is_heal = True
        is_drain = entry.get("drain", None) is not None
        if is_heal and not is_drain and battle.active_pokemon is not None:
            if getattr(battle.active_pokemon, "current_hp_fraction", 0) >= 0.98:
                return 1.0
    except Exception:
        pass
    # буст уже на +6 / -6 — wasted
    try:
        boosts = getattr(move, "boosts", None)
        if boosts is None:
            boosts = (getattr(move, "entry", {}) or {}).get("boosts", None)
        if boosts:
            # определяем цель буста: положительные -> на себя, отрицательные -> на оппа
            has_pos = any(v>0 for v in boosts.values())
            has_neg = any(v<0 for v in boosts.values())
            if has_pos and battle.active_pokemon is not None:
                cur = getattr(battle.active_pokemon, "boosts", {}) or {}
                for stat, delta in boosts.items():
                    if stat not in ("atk","def","spa","spd","spe","accuracy","evasion"):
                        continue
                    if delta > 0 and cur.get(stat, 0) >= 6:
                        return 1.0
            if has_neg and opp is not None:
                cur = getattr(opp, "boosts", {}) or {}
                for stat, delta in boosts.items():
                    if stat not in ("atk","def","spa","spd","spe","accuracy","evasion"):
                        continue
                    if delta < 0 and cur.get(stat, 0) <= -6:
                        return 1.0
    except Exception:
        pass
        
    if move_id in _HAZARD_MOVES:
        condition, max_layers = _HAZARD_MOVES[move_id]
        current = battle.opponent_side_conditions.get(condition, 0)
        current = 1 if current is True else (0 if current is False else current)
        return 1.0 if current >= max_layers else 0.0

    if move_id in _SCREEN_MOVES:
        return 1.0 if _SCREEN_MOVES[move_id] in battle.side_conditions else 0.0
    # Defog/Rapid Spin когда нет хазардов — wasted (нет смысла)
    if move_id in ("rapidspin","mortalspin","tidyup"):
        # эти чистят только свои хазарды
        if not battle.side_conditions:
            # проверяем есть ли хоть один хазард
            has_haz = any(c in battle.side_conditions for c in (SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB))
            if not has_haz:
                return 1.0
    if move_id in ("defog","courtchange"):
        has_own = any(c in battle.side_conditions for c in (SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB))
        has_opp = any(c in battle.opponent_side_conditions for c in (SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB))
        if not has_own and not has_opp:
            return 1.0

    if move.status is not None and opp is not None and opp.status is not None:
        return 1.0
    # статус на иммунный тип (яд на сталь, ожог на огонь и т.д.) — тоже wasted
    if move.status is not None and opp is not None:
        try:
            # move.status может быть Status enum или строка, приводим к строке типа "brn"/"psn"
            s = str(getattr(move.status, "name", str(move.status))).lower()
            # нормализуем "burn" -> "brn" и т.д.
            s_map = {"burn":"brn","paralyze":"par","paralysis":"par","poison":"psn","toxic":"tox","sleep":"slp","freeze":"frz","brn":"brn","par":"par","psn":"psn","tox":"tox","slp":"slp","frz":"frz"}
            s_key = s_map.get(s, s[:3])
            immune_types = {"brn": ["fire"], "par": ["electric","ground"], "psn": ["poison","steel"], "tox": ["poison","steel"], "slp": [], "frz": []}.get(s_key, [])
            if immune_types:
                # проверяем типы оппа
                for t in _eff_types(opp):
                    if t is not None and getattr(t, "name", str(t)).lower() in immune_types:
                        return 1.0
        except Exception:
            pass

    if opp is not None and opp.ability is not None:
        immune_type = _IMMUNITY_ABILITIES.get(opp.ability)
        if immune_type is not None and move.type is not None and move.type.name.lower() == immune_type:
            return 1.0

    if opp is not None and opp.item == "airballoon" and move.type is not None:
        if move.type.name.lower() == "ground":
            return 1.0
    # типовая иммуннасть (0 урона) — тоже wasted для дамажных приёмов (Earthquake vs Flying и т.д.)
    try:
        if opp is not None and move.type is not None:
            bp = getattr(move, "base_power", 0) or 0
            if bp == 0:
                entry = getattr(move, "entry", {}) or {}
                bp = entry.get("basePower", 0) or entry.get("base_power", 0) or 0
            if bp and bp >= 10:
                # FIX: раньше тут был каскад try/except с fallback mult=1.0 — при неизвестном
                # втором типе (??? / STELLAR) это прятало иммунитет. Теперь считаем покомпонентно.
                ot1, ot2 = _eff_types(opp)
                mult = damage_multiplier_safe(getattr(move, "type", None), ot1, ot2)
                if mult == 0:
                    return 1.0
    except Exception:
        pass
    return 0.0


def move_slots_for_action(battle) -> list:
    """Приёмы в том порядке, которому соответствуют действия 6..9 (и 22..25) poke-env.

    Эталон — `poke_env.environment.singles_env.SinglesEnv` (action_to_order / order_to_action /
    get_action_mask):

        known_moves = list(battle.active_pokemon.moves.values())[:4]
        mvs = battle.available_moves if (len(available_moves) == 1
                                         and available_moves[0].id not in known_ids) else known_moves

    `battle.available_moves` — НЕ тот список, который нумеруют действия: он короче и без «дыр»
    (сервер присылает только невыключенные приёмы: PP=0, Disable, Taunt, Choice-lock, Heal Block,
    Torment, ...). Маска и action_to_order при этом всегда нумеруют СЛОТЫ known_moves, поэтому
    `enumerate(battle.available_moves)` сдвигал признаки: при выключенном приёме №2 слот 1
    описывал приём №3, а слот 2 — приём №4, тогда как легальные действия 8/9 указывали на
    приёмы №3/№4, т.е. сеть видела характеристики не тех вариантов, что реально выбирала.

    Возвращаем ровно список слотов действий (длиной не больше 4); пустые слоты остаются
    дефолтными у вызывающего.
    """
    pokemon = getattr(battle, "active_pokemon", None)
    if pokemon is None:
        known_moves = []
    else:
        known_moves = list((getattr(pokemon, "moves", {}) or {}).values())[:4]
    avail = list(getattr(battle, "available_moves", None) or [])
    if len(avail) == 1 and avail[0].id not in [m.id for m in known_moves]:
        # редкий случай poke-env: доступен ровно один приём, которого нет среди известных
        return avail
    return known_moves


def move_slot_index(battle, move_id: str):
    """Индекс слота (0..3) для id приёма в раскладке действий, либо None."""
    mid = str(move_id or "").lower()
    for i, move in enumerate(move_slots_for_action(battle)):
        if str(getattr(move, "id", "") or "").lower() == mid:
            return i
    return None


def available_move_ids(battle) -> set:
    """id приёмов, которые СЕЙЧАС разрешены сервером (для флага wasted недоступных слотов)."""
    return {str(getattr(m, "id", "") or "").lower()
            for m in (getattr(battle, "available_moves", None) or [])}


# =====================================================================================
# Раскладка obs и «кандидатные» срезы (режим embed, см. AUDIT 32)
# =====================================================================================
# Второй режим работы политики оценивает КАНДИДАТОВ по их собственным признакам, а не по
# номеру слота. Для этого obs нужно уметь разрезать на:
#   * контекст (всё, что не про конкретного кандидата),
#   * векторы 4 приёмов (по 30 признаков на приём из головы + оценка урона),
#   * векторы 5 резервов (bench-слот 40 признаков + 12 множителей угрозы по типам соперника).
# Таблица ниже — единственный источник истины о раскладке. Сборка obs (embed_battle_with_fusion)
# сверяет с ней имена и ширины блоков и падает при расхождении, поэтому срезы не могут
# тихо разъехаться при правке признаков (как это уже случалось с N_FEATURES).
# kinds: per_move (ширина делится на 4 слота), bench (5 слотов по BENCH_SLOT_DIM),
#        damage (особый: [0:12] — по 3 значения на приём), type_matchup (особый), context.
BENCH_SLOT_DIM = 40                    # _RESERVE_SLOT_SIZE, см. _reserve_slot_vec
# TYPE_MATCHUP_ROW_SIZE / TYPE_MATCHUP_TYPE_SLOTS / TYPE_MATCHUP_TEAM_SLOTS определены выше

OBS_BLOCKS = (
    ("moves_base_power", 4, "per_move"),
    ("moves_dmg_multiplier", 4, "per_move"),
    ("moves_wasted", 4, "per_move"),
    ("moves_accuracy", 4, "per_move"),
    ("moves_pp_frac", 4, "per_move"),
    ("moves_boost_own", 20, "per_move"),
    ("moves_drop_opp", 20, "per_move"),
    ("moves_hazard_clear", 8, "per_move"),
    ("moves_heal", 4, "per_move"),
    ("moves_status_prob", 4, "per_move"),
    ("moves_priority", 4, "per_move"),
    ("moves_stab", 4, "per_move"),
    ("moves_recoil", 4, "per_move"),
    ("moves_phaze", 4, "per_move"),
    ("moves_contact", 4, "per_move"),
    ("moves_sound", 4, "per_move"),
    ("moves_multihit", 4, "per_move"),
    ("moves_type", 4, "per_move"),
    ("moves_category", 12, "per_move"),
    ("faint_hp", 4, "context"),
    ("our_status", 7, "context"),
    ("opp_status", 7, "context"),
    ("our_hazards", 4, "context"),
    ("opp_hazards", 4, "context"),
    ("our_switches", 2, "context"),
    ("opp_switches", 2, "context"),
    ("our_boosts", 7, "context"),
    ("opp_boosts", 7, "context"),
    ("our_actual_stats", 6, "context"),
    ("opp_actual_stats", 6, "context"),
    ("our_ability", 20, "context"),
    ("opp_ability", 20, "context"),
    ("weather", 5, "context"),
    ("field", 5, "context"),
    ("trick_room_tailwind", 3, "context"),
    ("our_screens", 3, "context"),
    ("opp_screens", 3, "context"),
    ("speed_advantage", 1, "context"),
    ("revealed", 2, "context"),
    ("semi_invuln", 2, "context"),
    ("sub_damaged", 2, "context"),
    ("restricted", 1, "context"),
    ("our_volatiles", 11, "context"),
    ("opp_volatiles", 11, "context"),
    ("our_item", 11, "context"),
    ("opp_item", 11, "context"),
    ("our_bench", 5 * BENCH_SLOT_DIM, "bench"),
    ("opp_bench", 5 * BENCH_SLOT_DIM, "context"),
    ("vulnerability", 2, "context"),
    ("tera_flags", 3, "context"),
    ("is_tera", 2, "context"),
    ("our_tera_type", 19, "context"),
    ("protect", 2, "context"),
    ("damage", DAMAGE_BLOCK_SIZE, "damage"),
    ("type_matchup", TYPE_MATCHUP_BLOCK_SIZE, "type_matchup"),
    # --- полная оценка приёмов (AUDIT 33): только ХВОСТ, чтобы старые колонки не сдвигались ---
    ("moves_boost_own_stages", 4 * 7, "per_move"),        # магнитуды по 7 статам (+/-, с шансом)
    ("moves_drop_opp_stages", 4 * 7, "per_move"),         # изменения статов соперника
    ("moves_status_self", 4, "per_move"),              # шанс статуса НА СЕБЯ (Rest)
    ("moves_charge_turns", 4, "per_move"),             # сколько ходов заряжается
    ("moves_recharge", 4, "per_move"),                 # тратит следующий ход (Hyper Beam)
    ("moves_revive", 4, "per_move"),                   # возрождение союзника (доля HP)
    ("moves_faint_user", 4, "per_move"),               # роняет своего (Explosion/Memento)
    ("moves_drain", 4, "per_move"),                    # доля урона в HP (Giga Drain)
    ("moves_hazard_set", 4 * 4, "per_move"),           # sr/spikes/tspikes/web
    ("moves_ability_flags", 4 * len(_ABILITY_MOVE_FLAGS), "per_move"),  # bullet/punch/slicing/...
)

# сколько признаков кандидата уходит в его embedding
# сумма per_move-блоков / 4 — считается из OBS_BLOCKS, чтобы не разъезжаться с раскладкой
MOVE_FEATURES_PER_SLOT = int(sum(w for _n, w, k in OBS_BLOCKS if k == "per_move") / 4)
# Хвостовые блоки «полной оценки приёмов» (AUDIT 33). Отдельная константа нужна тестам
# миграции чекпоинтов: они сравнивают рост obs_dim с суммой блоков.
MOVE_EFFECT_TAIL_BLOCKS = (
    "moves_boost_own_stages", "moves_drop_opp_stages", "moves_status_self", "moves_charge_turns",
    "moves_recharge", "moves_revive", "moves_faint_user", "moves_drain", "moves_hazard_set",
    "moves_ability_flags",
)
MOVE_EFFECT_BLOCK_SIZE = int(sum(w for n, w, _k in OBS_BLOCKS if n in MOVE_EFFECT_TAIL_BLOCKS))
assert MOVE_EFFECT_BLOCK_SIZE == 128, MOVE_EFFECT_BLOCK_SIZE
MOVE_DAMAGE_TRIPLE = 3                 # damage[3*i:3*i+3]: min_frac, max_frac, guaranteed_ko
MOVE_CAND_FLAGS = 2                    # can_tera, is_tera
MOVE_FEATURES_DIM = MOVE_FEATURES_PER_SLOT + MOVE_DAMAGE_TRIPLE                    # 33 без флагов
MOVE_CAND_DIM = MOVE_FEATURES_DIM + MOVE_CAND_FLAGS                                # 35 с флагами
SWITCH_THREAT_DIM = TYPE_MATCHUP_TYPE_SLOTS * TYPE_MATCHUP_TEAM_SLOTS  # 12 строк: 2 слота типов x 6 слотов соперника
SWITCH_CAND_DIM = BENCH_SLOT_DIM + SWITCH_THREAT_DIM                               # 52


def obs_layout() -> dict:
    """{имя: (start, width, kind)} + итог; считается из OBS_BLOCKS один раз."""
    global _OBS_LAYOUT_CACHE
    if _OBS_LAYOUT_CACHE is None:
        out = {}
        off = 0
        for name, width, kind in OBS_BLOCKS:
            out[name] = (off, int(width), kind)
            off += int(width)
        out["_total"] = (0, off, "total")
        _OBS_LAYOUT_CACHE = out
    return _OBS_LAYOUT_CACHE


_OBS_LAYOUT_CACHE = None


def verify_obs_layout(named_blocks) -> None:
    """Сверяет фактическую сборку obs с OBS_BLOCKS (имена и ширины, по порядку).

    Без этой сверки срезы кандидатов (и весь режим embed) молча поехали бы при правке
    признаков — ровно та же болезнь, что была с ручным подсчётом N_FEATURES.
    """
    expected = [(name, int(width)) for name, width, _kind in OBS_BLOCKS]
    got = [(name, int(np.asarray(arr).shape[-1]) if np.asarray(arr).ndim else 1)
           for name, arr in named_blocks]
    if got != expected:
        diff = next((f"{i}: {g} != {e}" for i, (g, e) in enumerate(zip(got, expected)) if g != e),
                    f"длина {len(got)} != {len(expected)}")
        raise AssertionError(
            f"раскладка obs разошлась с OBS_BLOCKS ({diff}). Обнови OBS_BLOCKS в features.py "
            f"вместе с изменением набора признаков — на таблицу опирается режим embed."
        )


def _index_maps() -> dict:
    """Индексные карты срезов: (4,33) приёмы, (5,52) резервы, (599,) контекст, can_tera.

    Модель (режим embed) берёт признаки кандидатов НЕ резанием по слотам, а этими картами:
    `x[:, MOVE_FEATURE_IDX]` -> (B, 4, 33). Карты считаются из OBS_BLOCKS, поэтому совпадают
    со сборкой obs по построению (см. verify_obs_layout).
    """
    global _INDEX_MAPS_CACHE
    if _INDEX_MAPS_CACHE is not None:
        return _INDEX_MAPS_CACHE
    layout = obs_layout()
    if layout["_total"][1] != N_FEATURES:
        raise AssertionError(f"OBS_BLOCKS дают {layout['_total'][1]} признаков, а N_FEATURES={N_FEATURES}")

    # карта приёмов: (4, 33) — строка = слот, столбцы = признаки этого приёма по порядку блоков
    # (собираем ПО СЛОТАМ, а не плоским списком: иначе reshape перепутал бы слоты с признаками)
    per_move_blocks = [(st, w) for st, w, kind in
                       (layout[name] for name, _w, _k in OBS_BLOCKS) if kind == "per_move"]
    d_start, _dw, _dk = layout["damage"]
    move_idx = np.zeros((4, MOVE_FEATURES_DIM), dtype=np.int64)
    col = 0
    for start, width in per_move_blocks:
        per = width // 4
        for slot in range(4):
            move_idx[slot, col:col + per] = np.arange(start + slot * per, start + (slot + 1) * per)
        col += per
    for slot in range(4):
        move_idx[slot, col:col + MOVE_DAMAGE_TRIPLE] = np.arange(
            d_start + 3 * slot, d_start + 3 * slot + MOVE_DAMAGE_TRIPLE)

    t_start, t_width, _ = layout["type_matchup"]
    n_types = 19
    b_start, _bw, _bk = layout["our_bench"]
    switch_core = np.zeros((5, BENCH_SLOT_DIM), dtype=np.int64)
    for slot in range(5):
        switch_core[slot] = np.arange(b_start + slot * BENCH_SLOT_DIM,
                                      b_start + (slot + 1) * BENCH_SLOT_DIM)
    threat_cols = []
    for row in range(TYPE_MATCHUP_TYPE_SLOTS * TYPE_MATCHUP_TEAM_SLOTS):
        threat_cols.append(t_start + n_types + row * TYPE_MATCHUP_ROW_SIZE + 2)   # колонка актива

    used = np.zeros(N_FEATURES, dtype=bool)
    used[move_idx.reshape(-1)] = True
    used[switch_core.reshape(-1)] = True
    for our_slot in range(1, 6):
        for row in range(TYPE_MATCHUP_TYPE_SLOTS * TYPE_MATCHUP_TEAM_SLOTS):
            used[t_start + n_types + row * TYPE_MATCHUP_ROW_SIZE + 2 + our_slot] = True
    ctx_idx = np.flatnonzero(~used).astype(np.int64)

    switch_idx = np.zeros((5, BENCH_SLOT_DIM + SWITCH_THREAT_DIM), dtype=np.int64)
    switch_idx[:, :BENCH_SLOT_DIM] = switch_core
    for slot in range(5):
        for row, base_col in enumerate(threat_cols):
            switch_idx[slot, BENCH_SLOT_DIM + row] = base_col + (slot + 1)

    _INDEX_MAPS_CACHE = {
        "move": move_idx,                                   # (4, 33)
        "switch": switch_idx,                               # (5, 52)
        "context": ctx_idx,                                 # (599,)
        "can_tera": int(layout["tera_flags"][0]),           # скаляр-индекс
        "is_tera": int(layout["is_tera"][0]),
    }
    return _INDEX_MAPS_CACHE


_INDEX_MAPS_CACHE = None


def move_feature_index() -> np.ndarray:
    """(4, 33) индексы признаков приёмов (без флагов кандидата)."""
    return _index_maps()["move"]


def switch_feature_index() -> np.ndarray:
    """(5, 52) индексы признаков резервов."""
    return _index_maps()["switch"]


def context_feature_index() -> np.ndarray:
    """(599,) индексы контекста (всё, что не про конкретного кандидата)."""
    return _index_maps()["context"]


def context_dim() -> int:
    """Длина контекстного вектора (все признаки, не относящиеся к конкретному кандидату)."""
    return int(context_feature_index().shape[0])


def can_tera_index() -> int:
    """Индекс признака our_can_tera_now (нужен как флаг кандидата)."""
    return _index_maps()["can_tera"]


def context_vector(obs) -> np.ndarray:
    """Всё, что не относится к конкретному кандидату (для оценки совместимости)."""
    obs = np.asarray(obs)
    return np.ascontiguousarray(obs[..., context_feature_index()], dtype=np.float32)


def move_candidate_matrix(obs, with_flags: bool = True) -> np.ndarray:
    """(4, 33|35): признаки каждого приёма в СВОЁМ слоте (как действия 6..9).

    with_flags=True добавляет 2 колонки-флага кандидата: can_tera (доступна ли тера) и
    is_tera (0 для обычного приёма) — их подставляет модель, чтобы обычные и теровые
    кандидаты различались при одинаковых признаках приёма.
    """
    obs = np.asarray(obs)
    core = obs[..., move_feature_index()]
    if not with_flags:
        return np.ascontiguousarray(core, dtype=np.float32)
    out = np.zeros(core.shape[:-1] + (MOVE_CAND_DIM,), dtype=np.float32)
    out[..., :core.shape[-1]] = core
    out[..., -2] = obs[..., can_tera_index()]
    out[..., -1] = 0.0
    return out


def tera_candidate_matrix(obs) -> np.ndarray:
    """(4, 35): те же приёмы, но как теровые кандидаты (is_tera=1)."""
    out = move_candidate_matrix(obs)
    out[..., -1] = 1.0
    return out


def switch_candidate_matrix(obs) -> np.ndarray:
    """(5, SWITCH_CAND_DIM): признаки каждого резерва в КАНОНИЧЕСКОМ порядке bench-блока.

    Именно этот порядок использует режим embed для действий-свитчей 0..4, поэтому признаки
    кандидата и действие, которым его выбирают, — одно и то же (в режиме indices действия
    нумеруются порядком team, который к признакам отношения не имеет).
    """
    obs = np.asarray(obs)
    return np.ascontiguousarray(obs[..., switch_feature_index()], dtype=np.float32)


def candidate_slices_ok(obs) -> bool:
    """Самопроверка: срезы не пересекаются и вместе с контекстом покрывают obs ровно один раз."""
    n = int(np.asarray(obs).shape[-1])
    used = np.zeros(n, dtype=np.int32)
    for idx in (move_feature_index(), switch_feature_index()):
        used[idx.reshape(-1)] += 1
    used[context_feature_index()] += 1
    return bool((used == 1).all())


def _damage_block(battle, our_fusion, opp_fusion, our_team_fusions=None, opp_team_fusions=None) -> np.ndarray:
    """Признаки потенциального урона: по приёмам активного, входящий удар и матрицы 6x6.

    Порядок (см. damage.DAMAGE_BLOCK_SIZE):
      [0:12]  4 приёма активного против активного противника: min_frac, max_frac, guaranteed_ko
      [12:15] лучший приём противника по нашему активному: min_frac, max_frac, guaranteed_ko
      [15:51] матрица «наш i -> их j»: min_frac урона по текущему HP цели
      [51:87] матрица «их j -> наш i»: min_frac урона по текущему HP нашей цели
      [87:99] известные приёмы противника по нашему активному: min_frac, max_frac, possible_KO
              (possible_KO — по максимальному роллу: «может убить», а не «убьёт наверняка»)
      [99:123] те же приёмы противника x наши 6 слотов: min_frac
      [123:155] флаги особых эффектов у противника (см. damage.EFFECT_FLAGS)

    Доли нормированы 0..1 (cap 2x HP). Нет данных/фейнт/неизвестно -> 0 (фейнт проверяется
    явно: у фейнта HP = 0, и деление урона на 1 HP насыщало бы признак до максимума).
    """
    out = np.zeros(DAMAGE_BLOCK_SIZE, dtype=np.float32)
    try:
        tag = str(getattr(battle, "battle_tag", "") or "")
        ctx_ours = DamageContext.ours(battle)
        ctx_theirs = DamageContext.theirs(battle)
        our_active = prepare_mon(getattr(battle, "active_pokemon", None), our_fusion)
        opp_active = prepare_mon(getattr(battle, "opponent_active_pokemon", None), opp_fusion)

        # --- 1) по приёмам нашего активного против их активного ---
        # слоты приёмов — как у действий 6..9 (known_moves), а не порядок available_moves:
        # иначе урон в слоте i описывал не тот приём, на который указывает действие 6+i
        moves = move_slots_for_action(battle)
        if our_active is not None and opp_active is not None:
            hp_now, hp_max = opp_active["hp_now"], opp_active["hp_max"]
            for i, move in enumerate(moves):
                r = estimate_damage(our_active, opp_active, move, ctx_ours)
                if not r:
                    continue
                out[3 * i + 0] = damage_frac(r[0], hp_now, hp_max)
                out[3 * i + 1] = damage_frac(r[1], hp_now, hp_max)
                out[3 * i + 2] = 1.0 if (hp_now is not None and r[0] >= hp_now) else 0.0

        # --- 2) лучший известный приём противника по нашему активному ---
        if opp_active is not None and our_active is not None:
            dmin, dmax, _ = cached_best_move_damage(opp_active, our_active, ctx_theirs, tag)
            hp_now, hp_max = our_active["hp_now"], our_active["hp_max"]
            out[12] = damage_frac(dmin, hp_now, hp_max)
            out[13] = damage_frac(dmax, hp_now, hp_max)
            out[14] = 1.0 if (hp_now is not None and dmin >= hp_now) else 0.0

        # --- 3-4) матрицы 6x6 (канонический порядок слотов по species) ---
        our_slots = team_slots(getattr(battle, "team", None))
        opp_slots = team_slots(getattr(battle, "opponent_team", None))
        our_team_fusions = our_team_fusions or {}
        opp_team_fusions = opp_team_fusions or {}

        def _fusion_for(mon, own_map):
            """Фьюжн-запись ТОЛЬКО со своей стороны.

            Раньше при отсутствии вида в своей карте подставлялась чужая
            (`our_team_fusions.get(sid) or opp_team_fusions.get(sid)`): если у обоих игроков
            совпадала species, наш покемон получал статы фьюжна противника (и наоборот).
            Фьюжн-партнёры у одинаковых species могут быть разными, поэтому чужая карта не
            используется вовсе — нет записи, значит статы берутся из декса (`prepare_mon`
            с None), как и для любого неизвестного случая.

            Ключ карты — вид-"тело" (имя из `|switch|` с "+"), а у мон виден вид-"голова":
            разбирается в `_fusion_entry_for`.
            """
            return _fusion_entry_for(mon, own_map)

        our_prepared = []
        for mon in our_slots:
            f = _fusion_for(mon, our_team_fusions)
            if f is None and mon is getattr(battle, "active_pokemon", None):
                f = our_fusion
            our_prepared.append(prepare_mon(mon, f))
        opp_prepared = []
        for mon in opp_slots:
            f = _fusion_for(mon, opp_team_fusions)
            if f is None and mon is getattr(battle, "opponent_active_pokemon", None):
                f = opp_fusion
            opp_prepared.append(prepare_mon(mon, f))

        our_slots = our_slots[:6]
        opp_slots = opp_slots[:6]
        our_prepared = our_prepared[:6]
        opp_prepared = opp_prepared[:6]

        # фейнт у цели -> 0: у фейнта HP = 0, а damage_frac делил бы урон на 1 HP и насыщал
        # признак до максимума (в obs уже есть отдельные флаги фейнта)
        m_our_to_opp = team_damage_matrix(our_prepared, opp_prepared, ctx_ours, tag)
        for i, row in enumerate(m_our_to_opp[:6]):
            for j, dmg in enumerate(row[:6]):
                dfn = opp_prepared[j] if j < len(opp_prepared) else None
                if dfn is None or getattr(dfn.get("mon"), "fainted", False):
                    continue
                out[15 + i * 6 + j] = damage_frac(dmg, dfn["hp_now"], dfn["hp_max"])

        m_opp_to_our = team_damage_matrix(opp_prepared, our_prepared, ctx_theirs, tag)
        for j, row in enumerate(m_opp_to_our[:6]):
            for i, dmg in enumerate(row[:6]):
                dfn = our_prepared[i] if i < len(our_prepared) else None
                if dfn is None or getattr(dfn.get("mon"), "fainted", False):
                    continue
                out[51 + j * 6 + i] = damage_frac(dmg, dfn["hp_now"], dfn["hp_max"])

        # --- 5) зеркальный урон: известные приёмы противника по НАМ ---
        # базы (MIRROR_BASE/TEAM_BASE/FLAGS_BASE) считаются в damage.py из размеров срезов,
        # магических чисел здесь быть не должно: сдвиг сломает уже обученные модели молча
        active_block, team_block = their_known_moves_damage(
            opp_active, our_active, our_prepared, ctx_theirs, n_slots=OPP_MOVE_SLOTS)
        out[MIRROR_BASE:TEAM_BASE] = active_block[:OPP_MOVE_SLOTS * 3]
        out[TEAM_BASE:FLAGS_BASE] = team_block[:OPP_MOVE_SLOTS * OUR_TEAM_SLOTS]

        # --- 6) флаги особых эффектов у противника (по всей раскрытой команде) ---
        opp_mons = list((getattr(battle, "opponent_team", None) or {}).values())
        if not opp_mons and isinstance(opp_active, dict):
            opp_mons = [opp_active["mon"]]
        out[FLAGS_BASE:FLAGS_BASE + EFFECT_FLAG_COUNT] = opponent_effect_flags(opp_mons)
    except Exception:
        # молчаливое проглатывание уже один раз стоило нам нулевой матрицы — логируем один раз
        global _DAMAGE_BLOCK_ERROR_LOGGED
        if not _DAMAGE_BLOCK_ERROR_LOGGED:
            _DAMAGE_BLOCK_ERROR_LOGGED = True
            import traceback
            print("[damage] ошибка построения блока признаков урона:")
            traceback.print_exc()
    return out


_DAMAGE_BLOCK_ERROR_LOGGED = False


def embed_battle_with_fusion(battle, our_fusion, opp_fusion, our_protected_last_turn=0.0, opp_protected_last_turn=0.0, our_team_fusions: dict | None = None, opp_team_fusions: dict | None = None, debug: bool = False):
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
    moves_contact = np.zeros(4, dtype=np.float32)
    moves_sound = np.zeros(4, dtype=np.float32)
    moves_multihit = np.zeros(4, dtype=np.float32)
    moves_type = np.zeros(4, dtype=np.float32)
    moves_category = np.zeros((4, 3), dtype=np.float32)
    # --- полная оценка эффектов приёма (AUDIT 33) ---
    moves_boost_own_stages = np.zeros((4, 7), dtype=np.float32)      # +7: accuracy/evasion
    moves_drop_opp_stages = np.zeros((4, 7), dtype=np.float32)
    moves_status_self = np.zeros(4, dtype=np.float32)
    moves_charge_turns = np.zeros(4, dtype=np.float32)
    moves_recharge = np.zeros(4, dtype=np.float32)
    moves_revive = np.zeros(4, dtype=np.float32)
    moves_faint_user = np.zeros(4, dtype=np.float32)
    moves_drain = np.zeros(4, dtype=np.float32)
    moves_hazard_set = np.zeros((4, len(_HAZARD_SET_KEYS)), dtype=np.float32)
    moves_ability_flags = np.zeros((4, len(_ABILITY_MOVE_FLAGS)), dtype=np.float32)
    type_chart = GenData.from_gen(battle.gen).type_chart

    # Слоты приёмов нумеруются как действия poke-env (known_moves[:4]), а не как
    # battle.available_moves: если часть приёмов выключена (PP=0/Disable/Taunt/Choice-lock),
    # available_moves короче и без «дыр», и признаки в слотах разъезжались с действиями
    # 6..9/22..25 (см. move_slots_for_action).
    _slots = move_slots_for_action(battle)
    _avail_ids = available_move_ids(battle)
    for i, move in enumerate(_slots):
        moves_base_power[i] = move.base_power / 100
        raw_acc = move.accuracy
        if raw_acc is True:
            moves_accuracy[i] = 1.0
        elif raw_acc is None:
            moves_accuracy[i] = 1.0
        else:
            moves_accuracy[i] = raw_acc / 100.0 if raw_acc > 1.0 else raw_acc
        moves_pp_frac[i] = move.current_pp / max(move.max_pp, 1)
        if str(getattr(move, "id", "") or "").lower() in _avail_ids:
            moves_wasted[i] = _move_wasted_flag(move, battle)
        else:
            # приём есть в мувсете, но сервер его сейчас не разрешает (PP=0, Disable, Taunt,
            # Choice-lock, Heal Block, Torment) — для действия это ровно «бесполезен сейчас»
            moves_wasted[i] = 1.0
        if battle.opponent_active_pokemon is not None:
            # FIX: раньше при KeyError писался нейтрал 1.0, из-за чего модель не видела
            # иммунитет (Electric vs Ground-фьюжн со вторым типом "???"). Теперь 0.0 сохраняется.
            _oat1, _oat2 = _eff_types(battle.opponent_active_pokemon)
            moves_dmg_multiplier[i] = damage_multiplier_safe(
                move.type,
                _oat1,
                _oat2,
                type_chart=type_chart,
            )
        moves_boost_own[i] = _move_boost_flags(move, "own")
        moves_drop_opp[i] = _move_boost_flags(move, "opp")
        moves_hazard_clear[i] = _move_hazard_clear_flags(move)
        moves_heal[i] = _move_heal_pct(move)
        moves_status_prob[i] = _move_status_prob(move)
        moves_priority[i] = _move_priority(move)
        moves_stab[i] = _move_stab_flag(move, battle.active_pokemon)
        moves_recoil[i] = _move_recoil_pct(move)
        moves_phaze[i] = _move_phaze_flag(move)
        moves_contact[i] = _move_contact_flag(move)
        moves_sound[i] = _move_sound_flag(move)
        moves_multihit[i] = _move_multihit_flag(move)
        moves_type[i] = _move_type_scalar(move)
        moves_category[i] = _move_category_vec(move)
        moves_boost_own_stages[i] = _move_boost_magnitude(move, "own")
        moves_drop_opp_stages[i] = _move_boost_magnitude(move, "opp")
        moves_status_self[i] = _move_status_self_prob(move)
        moves_charge_turns[i] = _move_charge_turns(move)
        moves_recharge[i] = _move_recharge_flag(move)
        moves_revive[i] = _move_revive_frac(move)
        moves_faint_user[i] = _move_faint_user_flag(move)
        moves_drain[i] = _move_drain_pct(move)
        moves_hazard_set[i] = _move_hazard_set_flags(move)
        moves_ability_flags[i] = _move_ability_flags(move)

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
    our_reserves = canonical_reserves(battle.team)
    opp_reserves = canonical_reserves(battle.opponent_team)

    our_bench = _bench_vec(battle.team, battle.opponent_active_pokemon, type_chart, our_team_fusions)
    opp_bench = _bench_vec(battle.opponent_team, battle.active_pokemon, type_chart, opp_team_fusions)

    our_vulnerability = _vulnerability_frac(our_reserves, battle.opponent_active_pokemon, type_chart)
    opp_vulnerability = _vulnerability_frac(opp_reserves, battle.active_pokemon, type_chart)
    our_can_tera_now = _can_tera_now(battle)
    our_used_tera = _team_used_tera(battle.team)
    opp_used_tera = _team_used_tera(battle.opponent_team)
    our_is_tera = _is_terastallized(battle.active_pokemon)
    opp_is_tera = _is_terastallized(battle.opponent_active_pokemon)
    our_tera_type = _tera_type_vec(battle.active_pokemon)
    trick_room = _trick_room_flag(battle)
    our_tailwind = _tailwind_flags(battle.side_conditions)
    opp_tailwind = _tailwind_flags(battle.opponent_side_conditions)
    our_screens = _screens_vec(battle.side_conditions)
    opp_screens = _screens_vec(battle.opponent_side_conditions)
    our_actual_stats = _actual_stats_vec(battle.active_pokemon, our_fusion)
    opp_actual_stats = _actual_stats_vec(battle.opponent_active_pokemon, opp_fusion)
    our_ability = _ability_vec(battle.active_pokemon)
    opp_ability = _ability_vec(battle.opponent_active_pokemon)
    damage_feats = _damage_block(battle, our_fusion, opp_fusion, our_team_fusions, opp_team_fusions)
    type_matchup_feats = _type_matchup_block(battle, type_chart)
    moves_boost_own_flat = moves_boost_own.flatten()
    moves_drop_opp_flat = moves_drop_opp.flatten()
    moves_hazard_clear_flat = moves_hazard_clear.flatten()
    moves_category_flat = moves_category.flatten()
    moves_boost_own_stages_flat = moves_boost_own_stages.flatten()
    moves_drop_opp_stages_flat = moves_drop_opp_stages.flatten()
    moves_hazard_set_flat = moves_hazard_set.flatten()
    moves_ability_flags_flat = moves_ability_flags.flatten()
    # Имена блоков нужны не для красоты: verify_obs_layout сверяет сборку с OBS_BLOCKS, на
    # которую опираются срезы кандидатов (режим embed). Молчаливый сдвиг блоков ловится здесь.
    named_blocks = [
        ("moves_base_power", moves_base_power),
        ("moves_dmg_multiplier", moves_dmg_multiplier),
        ("moves_wasted", moves_wasted),
        ("moves_accuracy", moves_accuracy),
        ("moves_pp_frac", moves_pp_frac),
        ("moves_boost_own", moves_boost_own_flat),
        ("moves_drop_opp", moves_drop_opp_flat),
        ("moves_hazard_clear", moves_hazard_clear_flat),
        ("moves_heal", moves_heal),
        ("moves_status_prob", moves_status_prob),
        ("moves_priority", moves_priority),
        ("moves_stab", moves_stab),
        ("moves_recoil", moves_recoil),
        ("moves_phaze", moves_phaze),
        ("moves_contact", moves_contact),
        ("moves_sound", moves_sound),
        ("moves_multihit", moves_multihit),
        ("moves_type", moves_type),
        ("moves_category", moves_category_flat),
        ("faint_hp", [fainted_mon_team, fainted_mon_opponent, our_hp, opp_hp]),
        ("our_status", our_status),
        ("opp_status", opp_status),
        ("our_hazards", our_hazards),
        ("opp_hazards", opp_hazards),
        ("our_switches", our_switches),
        ("opp_switches", opp_switches),
        ("our_boosts", our_boosts),
        ("opp_boosts", opp_boosts),
        ("our_actual_stats", our_actual_stats),
        ("opp_actual_stats", opp_actual_stats),
        ("our_ability", our_ability),
        ("opp_ability", opp_ability),
        ("weather", weather_vec),
        ("field", field_vec),
        ("trick_room_tailwind", [trick_room, our_tailwind, opp_tailwind]),
        ("our_screens", our_screens),
        ("opp_screens", opp_screens),
        ("speed_advantage", [speed_advantage]),
        ("revealed", [our_revealed, opp_revealed]),
        ("semi_invuln", [our_semi_invuln, opp_semi_invuln]),
        ("sub_damaged", [our_sub_damaged, opp_sub_damaged]),
        ("restricted", [our_restricted]),
        ("our_volatiles", our_volatiles),
        ("opp_volatiles", opp_volatiles),
        ("our_item", our_item),
        ("opp_item", opp_item),
        ("our_bench", our_bench),
        ("opp_bench", opp_bench),
        ("vulnerability", [our_vulnerability, opp_vulnerability]),
        ("tera_flags", [our_can_tera_now, our_used_tera, opp_used_tera]),
        ("is_tera", [our_is_tera, opp_is_tera]),
        ("our_tera_type", our_tera_type),
        ("protect", [our_protected_last_turn, opp_protected_last_turn]),
        ("damage", damage_feats),
        ("type_matchup", type_matchup_feats),
        ("moves_boost_own_stages", moves_boost_own_stages_flat),
        ("moves_drop_opp_stages", moves_drop_opp_stages_flat),
        ("moves_status_self", moves_status_self),
        ("moves_charge_turns", moves_charge_turns),
        ("moves_recharge", moves_recharge),
        ("moves_revive", moves_revive),
        ("moves_faint_user", moves_faint_user),
        ("moves_drain", moves_drain),
        ("moves_hazard_set", moves_hazard_set_flat),
        ("moves_ability_flags", moves_ability_flags_flat),
    ]
    verify_obs_layout(named_blocks)
    obs = np.concatenate(
        [arr for _name, arr in named_blocks], dtype=np.float32,
    )
    if False:  # старая сборка оставлена ниже закомментированной для истории правок
      obs = np.concatenate(
        [
            moves_base_power, moves_dmg_multiplier, moves_wasted, moves_accuracy, moves_pp_frac,
            moves_boost_own_flat, moves_drop_opp_flat, moves_hazard_clear_flat, moves_heal, moves_status_prob,
            moves_priority, moves_stab, moves_recoil, moves_phaze, moves_contact, moves_sound, moves_multihit,
            moves_type, moves_category_flat,
            [fainted_mon_team, fainted_mon_opponent, our_hp, opp_hp],
            our_status, opp_status, our_hazards, opp_hazards, our_switches, opp_switches,
            our_boosts, opp_boosts, our_actual_stats, opp_actual_stats, our_ability, opp_ability, weather_vec, field_vec,
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
            [our_is_tera, opp_is_tera],
            our_tera_type,
            [our_protected_last_turn, opp_protected_last_turn],
            damage_feats,
            type_matchup_feats,
        ],
        dtype=np.float32,
      )
    if not np.isfinite(obs).all():
        if debug:
            print(f"WARN embed_battle_with_fusion non-finite: {np.where(~np.isfinite(obs))[0][:10]}")
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    # защита от рассинхрона N_FEATURES (было 3 инцидента ручного подсчёта)
    from .config import N_FEATURES
    if damage_feats.shape[0] != DAMAGE_BLOCK_SIZE:  # страховка от рассинхрона блока
        raise AssertionError(f"damage-блок {damage_feats.shape[0]} != DAMAGE_BLOCK_SIZE {DAMAGE_BLOCK_SIZE}")
    if obs.shape[0] != N_FEATURES:
        raise AssertionError(f"embed_battle_with_fusion вернула {obs.shape[0]}, а N_FEATURES={N_FEATURES}. "
                         f"Обнови config.py (715 + {DAMAGE_BLOCK_SIZE} урона + "
                         f"{TYPE_MATCHUP_BLOCK_SIZE} типов соперника = {TYPE_MATCHUP_BASE + TYPE_MATCHUP_BLOCK_SIZE}).")
    return obs
