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
    # poke_env хранит Effect.SUBSTITUTE в pokemon.effects; значение может быть bool/int
    if Effect.SUBSTITUTE not in pokemon.effects:
        # fallback: некоторые версии могут хранить как строку 'substitute' в effects
        try:
            if not any(getattr(k, "name", str(k)).lower() == "substitute" for k in pokemon.effects.keys()):
                return 0.0
        except Exception:
            return 0.0
    value = pokemon.effects.get(Effect.SUBSTITUTE, None)
    # если ключ был строковым, пробуем достать иначе
    if value is None:
        for k, v in list(pokemon.effects.items()):
            if getattr(k, "name", str(k)).lower() == "substitute":
                value = v
                break
    if isinstance(value, (int, float)) and value > 0:
        return min(float(value) / 25.0, 1.0)
    if value is True:
        return 0.5
    # кукла стоит, но точное состояние неизвестно — нейтральное значение вместо 0/1
    # Effect.SUBSTITUTE in effects но value == 0/None -> всё равно кукла есть
    return 0.5

def _is_semi_invuln_or_charging(pokemon) -> float:
    if pokemon is None:
        return 0.0
    # основной путь: через Effect enum (из тех что существуют: PHANTOM_FORCE/SHADOW_FORCE/SKY_DROP)
    if any(e in pokemon.effects for e in _SEMI_INVULN_CHARGE_EFFECTS):
        return 1.0
    # fallback1: preparing (двухходовые: Solar Beam, Fly на зарядке и т.п.) — poke_env кладёт в _preparing_move
    try:
        prep = getattr(pokemon, "_preparing_move", None) or getattr(pokemon, "preparing_move", None)
        if prep is not None:
            return 1.0
    except Exception:
        pass
    # fallback2: проверка имени эффекта как строки (если будущая версия poke_env добавит FLY/DIG и т.п.)
    try:
        for eff in pokemon.effects.keys():
            n = getattr(eff, "name", str(eff))
            if n in _SEMI_INVULN_CHARGE_NAMES:
                return 1.0
            # также вариант lower без подчёркиваний
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
    """One-hot тера-типа покемона. Чиним DEAD: poke_env 0.8.x не заполняет tera_type из request,
    поэтому пробуем несколько источников (teambuilder, _terastallized_type, _last_details/request)."""
    vec = np.zeros(len(_TYPE_LIST), dtype=np.float32)
    if pokemon is None:
        return vec
    tera_type = None
    # 1) основной путь — pokemon.tera_type ( == _terastallized_type, заполняется для teambuilder и после terastallize)
    try:
        v = getattr(pokemon, "tera_type", None)
        if isinstance(v, PokemonType) and v in _TYPE_INDEX:
            tera_type = v
    except Exception:
        pass
    # 2) прямое поле _terastallized_type (на случай если property переопределят)
    if tera_type is None:
        try:
            v = getattr(pokemon, "_terastallized_type", None)
            if isinstance(v, PokemonType) and v in _TYPE_INDEX:
                tera_type = v
        except Exception:
            pass
    # 3) _last_details строка с 'tera:' (Showdown details: 'Pikachu, L83, tera:Flying' или 'super:tera:Flying')
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
    # 4) _last_request dict с полем 'teraType' (gen9 request) или 'details' с tera
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
                # некоторые серверы кладут tera в details внутри request
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
    """Доступна ли терастализация прямо сейчас как опция хода (только для своей стороны — poke-env не палит доступность у оппонента)."""
    can_tera = getattr(battle, "can_tera", None)
    return 1.0 if can_tera else 0.0

def _team_used_tera(team: dict) -> float:
    """Использовал ли кто-либо в команде терастал за весь бой (тера остаётся на моне даже после свитча)."""
    return 1.0 if any(getattr(mon, "is_terastallized", False) for mon in team.values()) else 0.0

_RESERVE_SLOT_SIZE = len(_TYPE_LIST) + 1 + len(_STATUSES)  # типы + HP + статус


def _reserve_slot_vec(mon) -> np.ndarray:
    type_vec = _type_multi_hot(mon)
    hp = 0.0 if mon.fainted else mon.current_hp_fraction
    status_vec = _status_one_hot(mon.status)
    return np.concatenate([type_vec, [hp], status_vec]).astype(np.float32)


def _bench_vec(team: dict) -> np.ndarray:
    reserves = [mon for mon in team.values() if not mon.active]
    slots = []
    for i in range(MAX_RESERVES):
        if i < len(reserves):
            slots.append(_reserve_slot_vec(reserves[i]))
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


def embed_battle_with_fusion(battle, our_fusion, opp_fusion, our_protected_last_turn=0.0, opp_protected_last_turn=0.0):
    from poke_env.data import GenData

    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    moves_wasted = np.zeros(4, dtype=np.float32)
    moves_accuracy = np.ones(4, dtype=np.float32)
    moves_pp_frac = np.ones(4, dtype=np.float32)
    type_chart = GenData.from_gen(battle.gen).type_chart

    for i, move in enumerate(battle.available_moves):
        moves_base_power[i] = move.base_power / 100
        raw_acc = move.accuracy
        if raw_acc is True:
            moves_accuracy[i] = 1.0
        elif raw_acc is None:
            moves_accuracy[i] = 1.0  # на случай отсутствия данных
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

    our_bench = _bench_vec(battle.team)
    opp_bench = _bench_vec(battle.opponent_team)

    our_vulnerability = _vulnerability_frac(our_reserves, battle.opponent_active_pokemon, type_chart)
    opp_vulnerability = _vulnerability_frac(opp_reserves, battle.active_pokemon, type_chart)
    # --- новое: терастал ---
    our_can_tera_now = _can_tera_now(battle)
    our_used_tera = _team_used_tera(battle.team)
    opp_used_tera = _team_used_tera(battle.opponent_team)
    our_tera_type = _tera_type_vec(battle.active_pokemon)
    obs = np.concatenate(
        [
            moves_base_power, moves_dmg_multiplier, moves_wasted, moves_accuracy, moves_pp_frac,
            [fainted_mon_team, fainted_mon_opponent, our_hp, opp_hp],
            our_status, opp_status, our_hazards, opp_hazards, our_switches, opp_switches,
            our_boosts, opp_boosts, weather_vec, field_vec,
            [speed_advantage],
            [our_revealed, opp_revealed],
            [our_semi_invuln, opp_semi_invuln],
            [our_sub_damaged, opp_sub_damaged],
            [our_restricted],
            our_volatiles, opp_volatiles,
            our_item, opp_item,
            our_bench, opp_bench,                          # <-- новое, по 5×_RESERVE_SLOT_SIZE на сторону
            [our_vulnerability, opp_vulnerability],         # <-- новое, 2 скаляра
            [our_can_tera_now, our_used_tera, opp_used_tera],   # <-- новое, 3 скаляра
            our_tera_type,                                       # <-- новое, len(_TYPE_LIST)
            [our_protected_last_turn, opp_protected_last_turn]
        ],
        dtype=np.float32,
    )
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs