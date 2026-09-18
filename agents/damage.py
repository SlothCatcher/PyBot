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
    moves = [m for m in (getattr(mon, "moves", None) or {}).values() if m is not None]
    return {
        "mon": mon,
        "stats": stats,
        "boosts": dict(getattr(mon, "boosts", {}) or {}),
        "ability": ability,
        "item": item,
        "status": status,
        "hp_now": current_hp_abs(mon, stats, fusion_entry),
        "hp_max": hp_max,
        "moves": moves,
        # части сигнатуры кэша считаем один раз здесь: _mon_sig вызывается ~200 раз за шаг
        # (2 матрицы 6x6 + зеркало), и сортировка статов/приёмов на каждом вызове съедала
        # больше времени, чем сам расчёт урона
        "stats_sig": tuple(sorted((str(k), int(v or 0)) for k, v in stats.items())),
        "moves_sig": tuple(sorted(_id(getattr(m, "id", "")) for m in moves)),
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


OPP_MOVE_SLOTS = 4        # сколько известных приёмов противника описываем
OUR_TEAM_SLOTS = 6        # слотов в наших матрицах

# Флаги «у противника есть приём с особым эффектом». Порядок фиксирован — он же порядок
# признаков; менять состав можно только в конец (иначе сдвинутся индексы для уже обученных
# моделей). Структурные поля Move (side_condition/volatile_status/force_switch/...) дают почти
# всё, там где движок признак явно не хранит — небольшие id-наборы.
EFFECT_FLAGS = (
    # входные опасности (hazards)
    "hazard_stealthrock", "hazard_spikes", "hazard_toxicspikes", "hazard_stickyweb", "hazard_any",
    # статусы
    "status_burn", "status_para", "status_poison", "status_sleep", "status_freeze",
    "status_confuse", "status_any",
    # контроль и защита
    "phazing", "protect", "screens", "substitute", "taunt_or_disable",
    # восстановление
    "healing_move", "drain_move", "leechseed",
    # поле
    "weather_setter", "terrain_setter", "self_switch",
    # прямая угроза
    "priority_attack", "setup_booster", "target_debuff", "trick_item", "knockoff",
    # счётчики (нормированы на 4 приёма)
    "count_status_moves", "count_setup_moves", "count_damaging_moves", "count_hazard_moves",
)
EFFECT_FLAG_COUNT = len(EFFECT_FLAGS)

# side_condition -> id флага. Атаки, которые ставят хазарды, side_condition не хранят
# (Stone Axe кладёт Stealth Rock, Ceaseless Edge — Spikes), поэтому для них явная карта.
SIDE_CONDITION_HAZARDS = {"STEALTH_ROCK": "stealthrock", "SPIKES": "spikes",
                          "TOXIC_SPIKES": "toxicspikes", "STICKY_WEB": "stickyweb"}
HAZARD_ATTACK_IDS = {"stoneaxe": "stealthrock", "ceaselessedge": "spikes"}
HAZARD_MOVE_IDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
HEAL_MOVE_IDS = {"recover", "roost", "slackoff", "softboiled", "milkdrink", "shoreup", "synthesis",
                 "moonlight", "morningsun", "wish", "rest", "lifedew", "junglehealing", "healorder",
                 "strengthsap", "floralhealing", "painsplit", "lunarblessing", "aquaring", "ingrain"}
TRICK_MOVE_IDS = {"trick", "switcheroo", "thief", "covet", "bestow"}
KNOCKOFF_MOVE_IDS = {"knockoff", "corrosivegas"}
TAUNT_DISABLE_MOVE_IDS = {"taunt", "encore", "torment", "disable", "imprison", "healblock"}
DEBUFF_MOVE_IDS = {"partingshot", "memento", "charm", "growl", "leer", "tailwhip", "screech",
                   "metalsound", "faketears", "captivate", "venomdrench", "nobleroar", "tickle",
                   "babydolleyes", "stringshot", "cottonspore", "scaryface"}


def _safe_attr(obj, name, default=None):
    """getattr, переживающий исключения в свойствах.

    `Move.status`/`Move.weather` внутри делают `Status[...]`/`Weather[...]`, то есть на незнакомой
    строке (кастомный мод) кидают KeyError. Раньше это уронило бы весь блок признаков урона —
    проверяем на всех 954 приёмах гена 9, но подстраховка нужна для модов.
    """
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return default if value is None else value


def _move_target(move) -> str:
    t = _safe_attr(move, "target")
    return str(getattr(t, "name", t) or "").upper()


def _positive_boosts(boosts) -> bool:
    return any(isinstance(v, (int, float)) and float(v) > 0 for v in (boosts or {}).values())


def _move_self_buffs(move) -> bool:
    """Бустит ли приём СЕБЯ: self_boost или secondary[].self.boosts (Flame Charge, Power-Up Punch).

    Только положительные: Overheat/Close Combat/Draco Meteor держат в self_boost ШТРАФ к своим
    статам, и считать их «setup» нельзя.
    """
    if _positive_boosts(_safe_attr(move, "self_boost", {}) or {}):
        return True
    for sec in (_safe_attr(move, "secondary", []) or []):
        if isinstance(sec, dict):
            self_part = sec.get("self")
            if isinstance(self_part, dict) and _positive_boosts(self_part.get("boosts")):
                return True
    return False


def _move_statuses(move) -> set:
    """Статусы/волатильные эффекты приёма: собственный status + secondary-эффекты."""
    out = set()
    st = _safe_attr(move, "status")
    if st is not None:
        out.add(str(getattr(st, "name", st)).upper())
    vs = _safe_attr(move, "volatile_status")
    if vs is not None:
        out.add(str(getattr(vs, "name", vs)).upper())
    for sec in (_safe_attr(move, "secondary", []) or []):
        if isinstance(sec, dict):
            for key in ("status", "volatileStatus", "volatile_status"):
                v = sec.get(key)
                if v:
                    out.add(str(v).upper())
    return out


def known_opponent_moves(mon) -> list:
    """Раскрытые приёмы покемона в детерминированном порядке (сортировка по id)."""
    moves = [m for m in (getattr(mon, "moves", None) or {}).values() if m is not None]
    return sorted(moves, key=lambda m: _id(getattr(m, "id", "")))


def _all_known_moves(mons) -> list:
    seen, out = set(), []
    for mon in (mons or []):
        for mv in known_opponent_moves(mon):
            mid = _id(getattr(mv, "id", ""))
            if mid not in seen:
                seen.add(mid)
                out.append(mv)
    return out


def opponent_effect_flags(opp_mons) -> list:
    """Флаги особых эффектов по ВСЕМ известным приёмам команды противника.

    Берём union по команде, а не только по активному: сеттер хазардов/статусов часто сидит
    на скамейке, и знать об этом заранее полезно.

    Каждый приём обрабатывается отдельно и под try: один странный приём (кастомный мод,
    незнакомый статус) не должен обнулять весь блок признаков — ошибка логируется один раз
    (`_note_effect_flag_error`, набор `_EFFECT_FLAG_ERRORS`).
    """
    flags = dict.fromkeys(EFFECT_FLAGS, 0.0)
    moves = _all_known_moves(opp_mons)
    if not moves:
        return [0.0] * EFFECT_FLAG_COUNT

    hazard_ids = set()
    screen_ids, weather_ids, terrain_ids = set(), set(), set()
    status_moves = setup_moves = damaging_moves = 0
    for mv in moves:
        mid = _id(getattr(mv, "id", ""))
        try:
            side = _safe_attr(mv, "side_condition")
            side_name = str(getattr(side, "name", side) or "").upper()
            statuses = _move_statuses(mv)
            cat = _move_category(mv)   # .name, иначе str(category) даёт "PHYSICAL (MOVE CATEGORY) OBJECT"
            boosts = _safe_attr(mv, "boosts", {}) or {}
            is_status_move = cat == "STATUS"
            is_damaging = cat in ("PHYSICAL", "SPECIAL")

            hazard = SIDE_CONDITION_HAZARDS.get(side_name)
            if hazard is None and (mid in HAZARD_ATTACK_IDS or mid in HAZARD_MOVE_IDS):
                hazard = HAZARD_ATTACK_IDS.get(mid, mid)
            if hazard:
                hazard_ids.add(hazard)
            if side_name in ("REFLECT", "LIGHT_SCREEN", "AURORA_VEIL"):
                screen_ids.add(side_name)
            if _safe_attr(mv, "weather") is not None:
                weather_ids.add(mid)
            if _safe_attr(mv, "terrain") is not None:
                terrain_ids.add(mid)

            if "BRN" in statuses:
                flags["status_burn"] = 1.0
            if "PAR" in statuses:
                flags["status_para"] = 1.0
            if statuses & {"PSN", "TOX"}:
                flags["status_poison"] = 1.0
            if "SLP" in statuses:
                flags["status_sleep"] = 1.0
            if "FRZ" in statuses:
                flags["status_freeze"] = 1.0
            if "CONFUSION" in statuses or mid == "confuseray":
                flags["status_confuse"] = 1.0

            if _safe_attr(mv, "force_switch", False):
                flags["phazing"] = 1.0
            if _safe_attr(mv, "is_protect_move", False) or _safe_attr(mv, "stalling_move", False):
                flags["protect"] = 1.0
            if _safe_attr(mv, "self_switch", False):
                flags["self_switch"] = 1.0
            if "SUBSTITUTE" in statuses or mid == "substitute":
                flags["substitute"] = 1.0
            if "TAUNT" in statuses or mid in TAUNT_DISABLE_MOVE_IDS:
                flags["taunt_or_disable"] = 1.0

            if float(_safe_attr(mv, "heal", 0) or 0) > 0 or mid in HEAL_MOVE_IDS:
                flags["healing_move"] = 1.0
            if float(_safe_attr(mv, "drain", 0) or 0) > 0:
                flags["drain_move"] = 1.0
            if "LEECH_SEED" in statuses or mid == "leechseed":
                flags["leechseed"] = 1.0

            if float(_safe_attr(mv, "priority", 0) or 0) > 0 and is_damaging:
                flags["priority_attack"] = 1.0
            # setup: буст СЕБЯ (в т.ч. через secondary) либо статусный приём с бустом,
            # который целится в себя (Swords Dance / Calm Mind, но не Swagger/Charm)
            target = _move_target(mv)
            self_targeted = target in ("", "SELF")
            if _move_self_buffs(mv) or (is_status_move and self_targeted and _positive_boosts(boosts)):
                flags["setup_booster"] = 1.0
                setup_moves += 1
            if (boosts and any(float(v) < 0 for v in boosts.values()
                               if isinstance(v, (int, float)))) or mid in DEBUFF_MOVE_IDS:
                flags["target_debuff"] = 1.0
            if mid in TRICK_MOVE_IDS:
                flags["trick_item"] = 1.0
            if mid in KNOCKOFF_MOVE_IDS:
                flags["knockoff"] = 1.0

            if is_status_move:
                status_moves += 1
            if is_damaging:
                damaging_moves += 1
        except Exception as exc:  # noqa: BLE001 — один битый приём не должен ронять блок
            _note_effect_flag_error(mid, exc)
            continue

    flags["hazard_stealthrock"] = 1.0 if "stealthrock" in hazard_ids else 0.0
    flags["hazard_spikes"] = 1.0 if "spikes" in hazard_ids else 0.0
    flags["hazard_toxicspikes"] = 1.0 if "toxicspikes" in hazard_ids else 0.0
    flags["hazard_stickyweb"] = 1.0 if "stickyweb" in hazard_ids else 0.0
    flags["hazard_any"] = 1.0 if hazard_ids else 0.0
    flags["screens"] = 1.0 if screen_ids else 0.0
    flags["weather_setter"] = 1.0 if weather_ids else 0.0
    flags["terrain_setter"] = 1.0 if terrain_ids else 0.0
    flags["status_any"] = 1.0 if any(
        flags[k] for k in ("status_burn", "status_para", "status_poison", "status_sleep",
                           "status_freeze", "status_confuse")) else 0.0
    flags["count_status_moves"] = min(status_moves / 4.0, 1.0)
    flags["count_setup_moves"] = min(setup_moves / 4.0, 1.0)
    flags["count_damaging_moves"] = min(damaging_moves / 4.0, 1.0)
    flags["count_hazard_moves"] = min(len(hazard_ids) / 4.0, 1.0)
    return [float(flags[k]) for k in EFFECT_FLAGS]


_EFFECT_FLAG_ERRORS: set = set()


def _note_effect_flag_error(move_id: str, exc: Exception) -> None:
    """Один раз на процесс сообщаем о битом приёме (молча глотать нельзя — так теряются данные)."""
    if move_id in _EFFECT_FLAG_ERRORS:
        return
    _EFFECT_FLAG_ERRORS.add(move_id)
    print(f"[damage] флаги эффектов: приём {move_id!r} пропущен ({type(exc).__name__}: {exc})")


def cached_move_damage(atk: Optional[dict], dfn: Optional[dict], move, ctx: Optional[DamageContext] = None,
                       atk_sig: Optional[tuple] = None, dfn_sig: Optional[tuple] = None):
    """estimate_damage с кэшем по (атакующий, защитник, приём, поле).

    Кэшируется и None (статусный приём/нет данных): иначе такие приёмы пересчитывались бы
    каждый шаг.
    """
    if atk is None or dfn is None or move is None:
        return None
    ctx = ctx or DamageContext()
    key = (atk_sig or _mon_sig(atk, True), dfn_sig or _mon_sig(dfn, False),
           _id(getattr(move, "id", "")), ctx.weather, ctx.terrain, ctx.defender_screens)
    if key in _MOVE_DMG_CACHE:
        return _MOVE_DMG_CACHE[key]
    val = estimate_damage(atk, dfn, move, ctx)
    if len(_MOVE_DMG_CACHE) > 60_000:
        _MOVE_DMG_CACHE.clear()
    _MOVE_DMG_CACHE[key] = val
    return val


_MOVE_DMG_CACHE: dict = {}

_PAIR_CACHE: dict = {}


def _mon_sig(prep: dict, attacker: bool) -> tuple:
    """Сигнатура покемона для кэшей урона: всё, от чего зависит estimate_damage.

    Раньше в сигнатуре был только `id(mon)` и ЧИСЛО известных приёмов. Этого мало:
    Python переиспользует id убитых объектов, поэтому запись из прошлого боя могла попасть
    в кэш нового (в бою живут одни и те же species с теми же статами — коллизия реальна).
    Теперь в сигнатуре species/статы/уровень/терра и сами id приёмов, а `id(mon)` оставлен
    лишь как быстрый дискриминатор внутри боя.
    """
    mon = prep["mon"]
    boosts = tuple(sorted((str(k), int(v or 0)) for k, v in (prep["boosts"] or {}).items()))
    types = (str(getattr(getattr(mon, "type_1", None), "name", "")),
             str(getattr(getattr(mon, "type_2", None), "name", "")))
    stats = prep.get("stats_sig")
    if stats is None:  # dict не из prepare_mon (тесты/внешние вызовы) — считаем сами
        stats = tuple(sorted((str(k), int(v or 0)) for k, v in (prep.get("stats") or {}).items()))
    moves = prep.get("moves_sig")
    if moves is None:
        moves = tuple(sorted(_id(getattr(m, "id", "")) for m in (prep.get("moves") or [])))
    tera = getattr(mon, "tera_type", None) or getattr(mon, "_terastallized_type", None)
    sig = (id(mon), str(getattr(mon, "species", "") or ""), types, stats,
           int(getattr(mon, "level", 100) or 100), str(getattr(tera, "name", tera) or ""),
           boosts, prep["ability"], prep["item"], moves)
    if attacker:
        sig = sig + (prep["status"],)
    else:
        full_hp = bool(prep["hp_now"] is not None and prep["hp_max"] and prep["hp_now"] >= prep["hp_max"])
        sig = sig + (full_hp,)
    return sig


def cached_best_move_damage(atk: Optional[dict], dfn: Optional[dict],
                            ctx: Optional[DamageContext] = None, battle_tag: str = "",
                            atk_sig: Optional[tuple] = None, dfn_sig: Optional[tuple] = None):
    """best_move_damage с кэшем по паре: (атакующий, защитник, поле, экраны).

    atk_sig/dfn_sig — заранее посчитанные `_mon_sig` (в матрицах 6x6 это экономит ~140
    построений сигнатуры на шаг).

    Кэшируется АБСОЛЮТНЫЙ урон (не доля от HP), поэтому результат переиспользуется между
    шагами, пока не изменились бусты/способность/предмет/состав приёмов/погода/экраны.
    """
    if atk is None or dfn is None:
        return (0.0, 0.0, "")
    ctx = ctx or DamageContext()
    key = (atk_sig or _mon_sig(atk, True), dfn_sig or _mon_sig(dfn, False),
           ctx.weather, ctx.terrain, ctx.defender_screens)
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
MIRROR_BASE = DAMAGE_MOVES * 3 + 3 + 36 + 36        # начало зеркального блока (было 87)
TEAM_BASE = MIRROR_BASE + OPP_MOVE_SLOTS * 3        # начало «их приёмы x наши слоты»
FLAGS_BASE = TEAM_BASE + OPP_MOVE_SLOTS * OUR_TEAM_SLOTS   # начало флагов эффектов
DAMAGE_BLOCK_SIZE = FLAGS_BASE + EFFECT_FLAG_COUNT       # сейчас 87/99/123/155


def team_slots(team) -> list:
    """Канонический порядок слотов команды: сортировка по species id.

    Dict-порядок в poke-env нестабилен (это давало permutation variance в bench-признаках),
    поэтому для матриц фиксируем порядок по имени вида.
    """
    mons = [m for m in (team or {}).values() if m is not None]
    return sorted(mons, key=lambda m: str(getattr(m, "species", "") or ""))


def their_known_moves_damage(opp_active: Optional[dict], our_active: Optional[dict],
                             our_prepared: Optional[list] = None,
                             ctx: Optional[DamageContext] = None,
                             n_slots: int = OPP_MOVE_SLOTS) -> tuple:
    """Урон от известных приёмов противника по НАМ (зеркало к нашему блоку урона).

    Возвращает (active_block, team_block):
      * active_block — n_slots x (min_frac, max_frac, possible_KO) по нашему активному;
      * team_block  — n_slots x OUR_TEAM_SLOTS min_frac по нашим слотам, сетка фиксированная
        (порядок слотов тот же, что в матрицах блока: team_slots, сортировка по species).

    Порядок приёмов — по убыванию минимального урона в наш активный, затем по id: результат
    не зависит от порядка раскрытия приёмов.

    possible_KO = 1, когда максимальный ролл добивает активного. Верхние срезы блока считают
    наоборот ГАРАНТИРОВАННЫЙ KO (по минимальному роллу): там мы ищем надёжный удар, здесь —
    опасность для нас, и «может убить» важнее, чем «убьёт наверняка».

    Фейнт → 0: у фейнта HP = 0, и без явной проверки `damage_frac` делил бы урон на 1 HP и
    насыщал признак до максимума (в obs есть отдельные флаги фейнта, урон по мёртвому не нужен).
    Если фейнт наш активный (решение о замене) — обнуляется только его срез, строки по скамейке
    остаются полезными.
    """
    ctx = ctx or DamageContext()
    our_prepared = list(our_prepared or [])
    out_active = [0.0] * (n_slots * 3)
    out_team = [0.0] * (n_slots * OUR_TEAM_SLOTS)
    if opp_active is None or our_active is None:
        return out_active, out_team

    opp_mon = opp_active["mon"] if isinstance(opp_active, dict) else opp_active
    moves = known_opponent_moves(opp_mon)
    if not moves:
        return out_active, out_team

    atk_sig = _mon_sig(opp_active, True)
    act_sig = _mon_sig(our_active, False)
    scored = []
    for mv in moves:
        dmg = cached_move_damage(opp_active, our_active, mv, ctx, atk_sig, act_sig)
        scored.append((float(dmg[0]) if dmg else 0.0, _id(getattr(mv, "id", "")), mv, dmg))
    scored.sort(key=lambda x: (-x[0], x[1]))

    hp_now = our_active.get("hp_now")
    hp_max = our_active.get("hp_max")
    active_alive = not getattr(our_active.get("mon"), "fainted", False)
    for slot, (_, _, mv, dmg) in enumerate(scored[:n_slots]):
        if dmg is not None and active_alive:
            out_active[slot * 3 + 0] = damage_frac(dmg[0], hp_now, hp_max)
            out_active[slot * 3 + 1] = damage_frac(dmg[1], hp_now, hp_max)
            out_active[slot * 3 + 2] = 1.0 if (hp_now is not None and dmg[1] >= hp_now) else 0.0
        for j, dfn in enumerate(our_prepared[:OUR_TEAM_SLOTS]):
            if dfn is None or getattr(dfn.get("mon"), "fainted", False):
                continue
            d = cached_move_damage(opp_active, dfn, mv, ctx, atk_sig, _mon_sig(dfn, False))
            if d is None:
                continue
            out_team[slot * OUR_TEAM_SLOTS + j] = damage_frac(d[0], dfn["hp_now"], dfn["hp_max"])
    return out_active, out_team


    opp_mon = opp_active["mon"] if isinstance(opp_active, dict) else opp_active
    moves = known_opponent_moves(opp_mon)
    if not moves:
        return out_active, out_team

    scored = []
    for mv in moves:
        dmg = cached_move_damage(opp_active, our_active, mv, ctx)
        scored.append((float(dmg[0]) if dmg else 0.0, _id(getattr(mv, "id", "")), mv, dmg))
    scored.sort(key=lambda x: (-x[0], x[1]))

    hp_now = our_active.get("hp_now")
    hp_max = our_active.get("hp_max")
    for slot, (_, _, mv, dmg) in enumerate(scored[:n_slots]):
        if dmg is None:
            out_active[slot * 3 + 0] = 0.0
            out_active[slot * 3 + 1] = 0.0
            out_active[slot * 3 + 2] = 0.0
        else:
            out_active[slot * 3 + 0] = damage_frac(dmg[0], hp_now, hp_max)
            out_active[slot * 3 + 1] = damage_frac(dmg[1], hp_now, hp_max)
            out_active[slot * 3 + 2] = 1.0 if (dmg[1] is not None and hp_now is not None
                                               and dmg[1] >= hp_now) else 0.0
        for j, dfn in enumerate(our_prepared):
            if dfn is None:
                continue
            d = dmg if (j == 0 and dfn is our_active) else cached_move_damage(opp_active, dfn, mv, ctx)
            if d is None:
                continue
            out_team[slot * len(our_prepared) + j] = damage_frac(d[0], dfn["hp_now"], dfn["hp_max"])
    return out_active, out_team


def team_damage_matrix(attackers_prepared: list, defenders_prepared: list,
                       ctx: DamageContext, battle_tag: str = "") -> list:
    """[[min_damage_abs]] матрица лучших приёмов (по минимальному урону).

    Возвращает список строк по атакующим; значения — абсолютный урон (0.0 для фainted/нет данных).
    Абсолютный урон кэшируемо-стабилен, долю от HP считает вызывающий.
    """
    atk_sigs = [None if a is None else _mon_sig(a, True) for a in attackers_prepared]
    dfn_sigs = [None if d is None else _mon_sig(d, False) for d in defenders_prepared]
    out = []
    for atk, atk_sig in zip(attackers_prepared, atk_sigs):
        row = []
        for dfn, dfn_sig in zip(defenders_prepared, dfn_sigs):
            if atk is None or dfn is None or getattr(atk["mon"], "fainted", False):
                row.append(0.0)
                continue
            dmin, _, _ = cached_best_move_damage(atk, dfn, ctx, battle_tag, atk_sig, dfn_sig)
            row.append(float(dmin))
        out.append(row)
    return out
