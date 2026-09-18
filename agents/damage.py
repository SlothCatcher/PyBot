"""Оценка потенциального урона приёма (для признаков obs).

Считаем «как в игре», но с оговорками, которые важны для RL-признаков:

* **минимум** (главная величина, как просил пользователь) — нижняя граница урона:
  минимальный ролл 0.85, без крита, без учёта неизвестных способностей/предметов;
* **максимум** — тот же расчёт с роллом 1.0 (верхняя граница для известных модификаторов).

Формула (поколение 5+, уровень 100):

    base = floor(floor(2*level/5 + 2) * power * A / D / 50) + 2
    dmg  = base * STAB * эффективность * погода * статус * экраны * способности * предметы
    min  = floor(dmg * 0.85), max = floor(dmg)

Где A/D — реальные статы (не базовые): считаются из base stats с учётом уровня и
фиксированных для random battle IV 31 / EV 85. Для фьюжнов base stats берутся из
html-таблицы чата (`fusion_entry["base_stats"]`), иначе из декса.

Способности/предметы учитываются, только если они известны (revealed) — иначе множитель 1.0.
Это осознанный компромисс: признак не должен «знать» скрытую информацию. Иммунитеты
(Levitate, Flash Fire, Volt Absorb, водопоглощение, Air Balloon) учитываются явно — они дают
0 урона, и именно эту информацию модель раньше не видела.

Всё, что не влияет на оценку (снаряды/связки вроде Zoom Lens), игнорируем.
"""
from __future__ import annotations

from typing import Any, Optional

from .type_utils import damage_multiplier_safe

# --- статы ---------------------------------------------------------------------------

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_IV = 31
_EV = 85  # random battles: 85 в каждый стат


def base_stats_of(mon, fusion_entry: Optional[dict] = None) -> Optional[dict]:
    """Base stats покемона: из html-таблицы фьюжна, иначе из декса."""
    if mon is None:
        return None
    if fusion_entry and isinstance(fusion_entry.get("base_stats"), dict):
        return fusion_entry["base_stats"]
    return getattr(mon, "base_stats", None)


def real_stat(base: float, level: int = 100, is_hp: bool = False, nature: float = 1.0) -> int:
    """Реальный стат из базового (IV 31, EV 85, нейтральная природа по умолчанию)."""
    b = float(base or 0)
    if is_hp:
        if b == 1:  # Shedinja
            return 1
        return int((2 * b + _IV + _EV / 4) * level / 100 + level + 10)
    return int(((2 * b + _IV + _EV / 4) * level / 100 + 5) * nature)


def mon_stats(mon, fusion_entry: Optional[dict] = None) -> Optional[dict]:
    """{hp, atk, def, spa, spd, spe} реальными числами (из base stats фьюжна/декса)."""
    base = base_stats_of(mon, fusion_entry)
    if not base:
        return None
    level = int(getattr(mon, "level", 100) or 100)
    out = {}
    for k in _STAT_KEYS:
        v = base.get(k, base.get(k.upper()))
        if v is None:
            v = 100
        out[k] = real_stat(v, level, is_hp=(k == "hp"))
    return out


def _hp_now(mon) -> Optional[int]:
    """Текущее HP в абсолютных единицах (None, если неизвестно)."""
    if mon is None:
        return None
    try:
        cur = getattr(mon, "current_hp", None)
        if cur is not None and cur > 0:
            return int(cur)
    except Exception:
        pass
    return None


def current_hp_abs(mon, stats: Optional[dict], fusion_entry: Optional[dict] = None) -> Optional[int]:
    """Текущее HP: абсолютное, если известно, иначе доля * max_hp."""
    if mon is None:
        return None
    cur = _hp_now(mon)
    if cur is not None:
        return cur
    if not stats:
        return None
    frac = getattr(mon, "current_hp_fraction", None)
    if frac is None:
        return stats["hp"]
    return max(1, int(round(float(frac) * stats["hp"])))


# --- бусты, погода, экраны -----------------------------------------------------------

def boost_mult(stage: int) -> float:
    """Классический множитель стадии: (2+n)/2 или 2/(2-n)."""
    s = max(-6, min(6, int(stage or 0)))
    return (2 + s) / 2 if s >= 0 else 2 / (2 - s)


_WEATHER_BOOST = {
    ("SUNNYDAY", "FIRE"): 1.5, ("SUNNYDAY", "WATER"): 0.5,
    ("RAINDANCE", "WATER"): 1.5, ("RAINDANCE", "FIRE"): 0.5,
}
_TERRAIN_BOOST = {
    ("ELECTRIC_TERRAIN", "ELECTRIC"): 1.3,
    ("GRASSY_TERRAIN", "GRASS"): 1.3,
    ("PSYCHIC_TERRAIN", "PSYCHIC"): 1.3,
    ("MISTY_TERRAIN", "FAIRY"): 1.3,
}

_ABILITY_IMMUNITY = {
    "levitate": "GROUND", "flashfire": "FIRE", "voltabsorb": "ELECTRIC",
    "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC", "waterabsorb": "WATER",
    "stormdrain": "WATER", "sapsipper": "GRASS", "dryskin": "WATER",
    "windrider": None, "earth eater": "GROUND", "eartheater": "GROUND",
    "wellbakedbody": "FIRE", "sapsipper": "GRASS",
}

# атакующие способности: (множитель, фильтр по типу приёма или None = любой)
_ATK_ABILITY_MULT = {
    "adaptability": None,      # STAB 2.0 вместо 1.5 (обрабатываем отдельно)
    "technician": None,        # bp <= 60 -> 1.5 (обрабатываем отдельно)
    "steelworker": "STEEL",
    "transistor": "ELECTRIC",
    "dragonsmaw": "DRAGON",
    "waterbubble": "WATER",
    "toughclaws": "CONTACT",
    "strongjaw": "BITE",
    "ironfist": "PUNCH",
    "punkrock": "SOUND",
    "reckless": None,          # на отдачу/прыжок — приблизительно не учитываем
    "sheerforce": None,
}
_ATK_ABILITY_FLAT = {  # способности, удваивающие стат
    "hugepower": "atk", "purepower": "atk",
}
# защитные способности: множитель урона
_DEF_ABILITY_MULT = {
    "multiscale": 0.5, "shadowshield": 0.5,       # на полном HP
    "filter": 0.75, "solidrock": 0.75, "prismarmor": 0.75,
    "thickfat": 0.5, "heatproof": 0.5,            # против огня
    "icescales": 0.5,                             # против спец. приёмов
    "fluffy": 0.5,                                # против контактных
    "punkrock": 0.5,                              # против звуковых
    "waterbubble": 0.5,                           # против огня
    "dryskin": 1.25,                              # огонь бьёт сильнее
}

_ITEM_ATK_MULT = {"lifeorb": 1.3, "choiceband": 1.5, "choicespecs": 1.5, "expertbelt": 1.2}
_ITEM_DEF_MULT = {"assaultvest": 1.5, "eviolite": 1.5}


def _id(x) -> str:
    return str(x or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _screens_of(side_conditions) -> tuple:
    """(physical_screen_up, special_screen_up) по сайд-условиям стороны защиты."""
    try:
        from poke_env.battle import SideCondition
        sc = side_conditions or {}
        refl = SideCondition.REFLECT in sc
        ls = SideCondition.LIGHT_SCREEN in sc
        veil = SideCondition.AURORA_VEIL in sc
        return (bool(refl or veil), bool(ls or veil))
    except Exception:
        return (False, False)


class DamageContext:
    """Поле + ЭКРАНЫ СТОРОНЫ ЗАЩИТЫ для конкретного направления атаки.

    Экраны зависят от направления: когда бьём мы — режут экраны противника; когда бьют нас —
    наши. Поэтому контекст создаётся под направление (`attacker_is_ours=True/False`).
    """

    def __init__(self, battle=None, attacker_is_ours: bool = True,
                 weather: Optional[str] = None, terrain: Optional[str] = None,
                 defender_screens=(False, False)):
        if battle is not None:
            try:
                w = next(iter(battle.weather), None) if getattr(battle, "weather", None) else None
                weather = str(getattr(w, "name", w) or "").upper() or None
            except Exception:
                weather = None
            try:
                f = next(iter(battle.fields), None) if getattr(battle, "fields", None) else None
                terrain = str(getattr(f, "name", f) or "").upper() or None
            except Exception:
                terrain = None
            own = _screens_of(getattr(battle, "side_conditions", None))
            opp = _screens_of(getattr(battle, "opponent_side_conditions", None))
            defender_screens = opp if attacker_is_ours else own
        self.weather = weather
        self.terrain = terrain
        self.defender_screens = defender_screens

    @classmethod
    def ours(cls, battle):
        return cls(battle, attacker_is_ours=True)

    @classmethod
    def theirs(cls, battle):
        return cls(battle, attacker_is_ours=False)


def prepare_mon(mon, fusion_entry: Optional[dict] = None) -> Optional[dict]:
    """Всё, что нужно для расчёта урона по этому покемону (считается один раз)."""
    if mon is None:
        return None
    stats = mon_stats(mon, fusion_entry)
    if stats is None:
        return None
    ability = _id(getattr(mon, "ability", None))
    item = _id(getattr(mon, "item", None))
    status = str(getattr(getattr(mon, "status", None), "name", getattr(mon, "status", "")) or "").upper()
    # атакующие способности, удваивающие стат
    if ability in _ATK_ABILITY_FLAT:
        stats = dict(stats)
        stats[_ATK_ABILITY_FLAT[ability]] = int(stats[_ATK_ABILITY_FLAT[ability]] * 2)
    # предметы, повышающие стат защиты
    if item in _ITEM_DEF_MULT:
        stats = dict(stats)
        m = _ITEM_DEF_MULT[item]
        stats["def"] = int(stats["def"] * m)
        stats["spd"] = int(stats["spd"] * m)
    # фактический максимум HP из боя важнее расчётного из base stats (иначе Multiscale и доли
    # урона считались бы от завышенного знаменателя)
    hp_max = None
    try:
        mh = getattr(mon, "max_hp", None)
        if mh and int(mh) > 0:
            hp_max = int(mh)
    except Exception:
        hp_max = None
    if not hp_max:
        hp_max = stats["hp"]
    return {
        "mon": mon,
        "stats": stats,
        "boosts": dict(getattr(mon, "boosts", {}) or {}),
        "ability": ability,
        "item": item,
        "status": status,
        "hp_now": current_hp_abs(mon, stats, fusion_entry),
        "hp_max": hp_max,
        "moves": [m for m in (getattr(mon, "moves", None) or {}).values() if m is not None],
    }


def _move_power(move) -> int:
    bp = getattr(move, "base_power", 0) or 0
    if not bp:
        entry = getattr(move, "entry", {}) or {}
        bp = entry.get("basePower", 0) or entry.get("base_power", 0) or 0
    return int(bp or 0)


def _move_category(move) -> str:
    cat = getattr(move, "category", None)
    name = str(getattr(cat, "name", cat) or "").upper()
    if name in ("PHYSICAL", "SPECIAL"):
        return name
    try:
        entry = getattr(move, "entry", {}) or {}
        return str(entry.get("category", "") or "").upper()
    except Exception:
        return ""


def _move_flags(move) -> dict:
    """Флаги приёма: contact/sound/bite/punch (для способностей)."""
    flags = {}
    try:
        entry = getattr(move, "entry", {}) or {}
        f = entry.get("flags", {}) or {}
        if isinstance(f, dict):
            flags.update({k.lower(): True for k in f.keys()})
    except Exception:
        pass
    mid = _id(getattr(move, "id", ""))
    if "sound" not in flags and _move_is_sound(mid):
        flags["sound"] = True
    if "bite" not in flags and mid in _BITE_MOVES:
        flags["bite"] = True
    if "punch" not in flags and ("punch" in mid or mid in _PUNCH_MOVES):
        flags["punch"] = True
    if "contact" not in flags and _move_is_contact(mid):
        flags["contact"] = True
    return flags


# Кэш статических данных приёма: id -> (power, is_phys, type_name, flags).
# Обход 36 пар x 4 приёма на каждом шаге делает строковые операции дорогими, поэтому
# всё неизменное считаем один раз.
_MOVE_STATIC: dict = {}


def move_static(move) -> dict:
    key = _id(getattr(move, "id", ""))
    st = _MOVE_STATIC.get(key) if key else None
    if st is None:
        cat = _move_category(move)
        mtype = getattr(move, "type", None)
        st = {
            "power": _move_power(move),
            "category": cat,
            "is_phys": cat == "PHYSICAL",
            "type_name": str(getattr(mtype, "name", mtype) or "").upper(),
            "flags": _move_flags(move),
        }
        if len(_MOVE_STATIC) > 4000:
            _MOVE_STATIC.clear()
        if key:
            _MOVE_STATIC[key] = st
    return st


_BITE_MOVES = {"bite", "crunch", "firefang", "icefang", "thunderfang", "poisonfang", "psychicfangs", "jawlock", "fishiousrend"}
_PUNCH_MOVES = {"bulletpunch", "machpunch", "drainpunch", "icepunch", "firepunch", "thunderpunch", "focuspunch", "hammerarm", "skyuppercut", "closecombat"}
_SOUND_MOVES = {"boomburst", "hypervoice", "bugbuzz", "snarl", "overdrive", "torchsong", "uproar", "round", "echoedvoice", "relicsong", "sparklingaria", "clangingscales", "clangoroussoul", "metalsound", "screech", "sing", "perishsong", "grasswhistle", "snore", "confide", "nobleroar", "partingshot"}
_NO_CONTACT_IDS = {"earthquake", "earthpower", "surf", "hydropump", "flamethrower", "icebeam", "thunderbolt", "psychic", "shadowball", "sludgebomb", "energyball", "focusblast", "dazzlinggleam", "moonblast", "flashcannon", "aurasphere", "darkpulse", "dragonpulse", "fierydance", "scald", "waterspout", "eruption", "hurricane", "airslash", "heatwave", "blizzard", "thunder", "rockslide", "stoneedge", "rockblast", "bulletseed", "pinmissile", "icywind"}


def _move_is_sound(mid: str) -> bool:
    return mid in _SOUND_MOVES


def _move_is_contact(mid: str) -> bool:
    return mid not in _NO_CONTACT_IDS


def estimate_damage(atk: Optional[dict], dfn: Optional[dict], move, ctx: Optional[DamageContext] = None,
                    defender_fusion: Optional[dict] = None) -> Optional[tuple]:
    """(dmg_min, dmg_max) в абсолютных HP, либо None если приём не дамажный/нет данных.

    atk/dfn — результаты prepare_mon. Эффективность типов и иммунитеты учитываются,
    включая способности-иммунитеты и Air Balloon защиты.
    """
    if atk is None or dfn is None or move is None:
        return None
    st = move_static(move)
    power = st["power"]
    if power <= 0:
        return None
    category = st["category"]
    if category not in ("PHYSICAL", "SPECIAL"):
        return None
    ctx = ctx or DamageContext()
    mtype = getattr(move, "type", None)
    if mtype is None:
        return None
    atk_name = st["type_name"]

    # эффективность по типам защиты (с учётом неизвестных типов — покомпонентно)
    dmon = dfn["mon"]
    eff = damage_multiplier_safe(mtype, getattr(dmon, "type_1", None), getattr(dmon, "type_2", None))
    if eff <= 0:
        return (0.0, 0.0)

    # способности-иммунитеты защиты + воздушный шар
    ab = dfn["ability"]
    if ab in _ABILITY_IMMUNITY:
        imm_type = _ABILITY_IMMUNITY[ab]
        if imm_type and atk_name == imm_type:
            return (0.0, 0.0)
    if dfn["item"] == "airballoon" and atk_name == "GROUND":
        return (0.0, 0.0)
    # способность защиты из «двойного типа»: Dry Skin — иммунитет к воде
    if ab == "dryskin" and atk_name == "WATER":
        return (0.0, 0.0)
    if ab in ("waterabsorb", "stormdrain") and atk_name == "WATER":
        return (0.0, 0.0)
    if ab == "flashfire" and atk_name == "FIRE":
        return (0.0, 0.0)
    if ab == "sapsipper" and atk_name == "GRASS":
        return (0.0, 0.0)
    if ab == "levitate" and atk_name == "GROUND":
        return (0.0, 0.0)

    # A и D со стадиями
    is_phys = st["is_phys"]
    atk_key = "atk" if is_phys else "spa"
    def_key = "def" if is_phys else "spd"
    atk_stage = int(atk["boosts"].get(atk_key, 0) or 0)
    def_stage = int(dfn["boosts"].get(def_key, 0) or 0)
    A = atk["stats"][atk_key] * boost_mult(atk_stage)
    D = dfn["stats"][def_key] * boost_mult(def_stage)

    level = int(getattr(atk["mon"], "level", 100) or 100)
    base = int((int(2 * level / 5) + 2) * power * A / D / 50) + 2

    mult = float(eff)
    # STAB
    amon = atk["mon"]
    stab_types = {str(getattr(getattr(amon, "type_1", None), "name", "") or "").upper(),
                  str(getattr(getattr(amon, "type_2", None), "name", "") or "").upper()}
    tera = getattr(amon, "tera_type", None) or getattr(amon, "_terastallized_type", None)
    tera_name = str(getattr(tera, "name", tera) or "").upper()
    if atk_name in stab_types or (tera_name and atk_name == tera_name):
        mult *= 2.0 if atk["ability"] == "adaptability" else 1.5
    # техник
    if atk["ability"] == "technician" and power <= 60:
        mult *= 1.5
    # способности-множители атакующего
    ab_mult = _ATK_ABILITY_MULT.get(atk["ability"])
    if ab_mult and isinstance(ab_mult, str):
        flags = st["flags"]
        if ab_mult == atk_name or ab_mult in ("CONTACT", "SOUND", "BITE", "PUNCH") and _flag_match(ab_mult, flags):
            mult *= {"toughclaws": 1.3, "strongjaw": 1.5, "ironfist": 1.2, "punkrock": 1.3,
                     "steelworker": 1.5, "transistor": 1.3, "dragonsmaw": 1.5, "waterbubble": 2.0}.get(atk["ability"], 1.0)
    # погода/террейн
    if ctx.weather and (ctx.weather, atk_name) in _WEATHER_BOOST:
        mult *= _WEATHER_BOOST[(ctx.weather, atk_name)]
    if ctx.terrain and (ctx.terrain, atk_name) in _TERRAIN_BOOST:
        mult *= _TERRAIN_BOOST[(ctx.terrain, atk_name)]
    # предметы атакующего
    item = atk["item"]
    if item in _ITEM_ATK_MULT:
        if item == "choiceband" and not is_phys:
            pass
        elif item == "choicespecs" and is_phys:
            pass
        elif item == "expertbelt" and eff <= 1.0:
            pass
        else:
            mult *= _ITEM_ATK_MULT[item]
    # статус атакующего (ожог режет физический урон)
    if is_phys and atk["status"] in ("BRN", "BURN"):
        mult *= 1.5 if atk["ability"] == "guts" else 0.5
    if atk["ability"] == "guts" and atk["status"] not in ("", "FNT", "NONE"):
        mult *= 1.5
    # экраны защиты
    phys_screen, spec_screen = ctx.defender_screens
    if is_phys and phys_screen:
        mult *= 0.5
    if (not is_phys) and spec_screen:
        mult *= 0.5
    # способности защиты
    dfn_ab = dfn["ability"]
    if dfn_ab in _DEF_ABILITY_MULT:
        m = _DEF_ABILITY_MULT[dfn_ab]
        applies = False
        if dfn_ab in ("multiscale", "shadowshield"):
            applies = dfn["hp_now"] is not None and dfn["hp_max"] and dfn["hp_now"] >= dfn["hp_max"]
        elif dfn_ab in ("filter", "solidrock", "prismarmor"):
            applies = eff > 1.0
        elif dfn_ab in ("thickfat", "heatproof", "waterbubble"):
            applies = atk_name == "FIRE"
        elif dfn_ab == "dryskin":
            applies = atk_name == "FIRE"
        elif dfn_ab == "icescales":
            applies = not is_phys
        elif dfn_ab in ("fluffy", "punkrock"):
            applies = _flag_match("CONTACT" if dfn_ab == "fluffy" else "SOUND", st["flags"])
        if applies:
            mult *= m

    dmg = base * mult
    if dmg <= 0:
        return (0.0, 0.0)
    return (float(int(dmg * 0.85)), float(int(dmg)))


_flags_cache: dict = {}


def flags_cache(move) -> dict:
    key = _id(getattr(move, "id", "")) or id(move)
    f = _flags_cache.get(key)
    if f is None:
        f = _move_flags(move)
        _flags_cache[key] = f
    return f


def _flag_match(want: str, flags: dict) -> bool:
    if want == "CONTACT":
        return bool(flags.get("contact", True))
    return bool(flags.get(want.lower()))


def best_move_damage(atk: Optional[dict], dfn: Optional[dict], ctx: Optional[DamageContext] = None):
    """Лучший по минимальному урону приём: (dmg_min, dmg_max, move_id)."""
    if atk is None or dfn is None:
        return (0.0, 0.0, "")
    best = (0.0, 0.0, "")
    for move in atk["moves"]:
        r = estimate_damage(atk, dfn, move, ctx)
        if not r:
            continue
        if r[0] > best[0] or (r[0] == best[0] and r[1] > best[1]):
            best = (r[0], r[1], _id(getattr(move, "id", "")))
    return best


_PAIR_CACHE: dict = {}


def _mon_sig(prep: dict, attacker: bool) -> tuple:
    """Сигнатура покемона для кэша пар: меняется только то, что влияет на урон."""
    mon = prep["mon"]
    boosts = tuple(sorted((str(k), int(v or 0)) for k, v in (prep["boosts"] or {}).items()))
    types = (str(getattr(getattr(mon, "type_1", None), "name", "")),
             str(getattr(getattr(mon, "type_2", None), "name", "")))
    sig = (id(mon), boosts, prep["ability"], prep["item"], len(prep["moves"] or []))
    if attacker:
        sig = sig + (prep["status"], types)
    else:
        full_hp = bool(prep["hp_now"] is not None and prep["hp_max"] and prep["hp_now"] >= prep["hp_max"])
        sig = sig + (types, full_hp)
    return sig


def cached_best_move_damage(atk: Optional[dict], dfn: Optional[dict],
                            ctx: Optional[DamageContext] = None, battle_tag: str = ""):
    """best_move_damage с кэшем по паре: (атакующий, защитник, поле, экраны).

    Кэшируется АБСОЛЮТНЫЙ урон (не доля от HP), поэтому результат переиспользуется между
    шагами, пока не изменились бусты/способность/предмет/состав приёмов/погода/экраны.
    """
    if atk is None or dfn is None:
        return (0.0, 0.0, "")
    ctx = ctx or DamageContext()
    key = (_mon_sig(atk, True), _mon_sig(dfn, False), ctx.weather, ctx.terrain, ctx.defender_screens)
    hit = _PAIR_CACHE.get(key)
    if hit is None:
        hit = best_move_damage(atk, dfn, ctx)
        if len(_PAIR_CACHE) > 40_000:
            _PAIR_CACHE.clear()
        _PAIR_CACHE[key] = hit
    return hit


def damage_frac(dmg: float, hp_now: Optional[int], hp_max: Optional[int], cap: float = 2.0) -> float:
    """Доля урона: от текущего HP (если известно), иначе от максимума. Нормируем 0..1."""
    denom = hp_now if (hp_now and hp_now > 0) else hp_max
    if not denom or denom <= 0:
        return 0.0
    return float(min(max(dmg, 0.0) / denom, cap) / cap)


# --- размер блока признаков урона в obs ------------------------------------------------
#  4 доступных приёма x (min_frac, max_frac, guaranteed_ko)   = 12
#  лучший приём оппонента по нам: (min_frac, max_frac, ko)     =  3
#  матрица «наш покемон -> покемон противника» (6x6, min_frac) = 36
#  матрица «покемон противника -> наш покемон» (6x6, min_frac) = 36
DAMAGE_MOVES = 4
DAMAGE_BLOCK_SIZE = DAMAGE_MOVES * 3 + 3 + 36 + 36


def team_slots(team) -> list:
    """Канонический порядок слотов команды: сортировка по species id.

    Dict-порядок в poke-env нестабилен (это давало permutation variance в bench-признаках),
    поэтому для матриц фиксируем порядок по имени вида.
    """
    mons = [m for m in (team or {}).values() if m is not None]
    return sorted(mons, key=lambda m: str(getattr(m, "species", "") or ""))


def team_damage_matrix(attackers_prepared: list, defenders_prepared: list,
                       ctx: DamageContext, battle_tag: str = "") -> list:
    """[[min_damage_abs]] матрица лучших приёмов (по минимальному урону).

    Возвращает список строк по атакующим; значения — абсолютный урон (0.0 для фainted/нет данных).
    Абсолютный урон кэшируемо-стабилен, долю от HP считает вызывающий.
    """
    out = []
    for atk in attackers_prepared:
        row = []
        for dfn in defenders_prepared:
            if atk is None or dfn is None or getattr(atk["mon"], "fainted", False):
                row.append(0.0)
                continue
            dmin, _, _ = cached_best_move_damage(atk, dfn, ctx, battle_tag)
            row.append(float(dmin))
        out.append(row)
    return out
