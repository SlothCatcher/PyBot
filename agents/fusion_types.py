"""Фьюжн-типы формата `*fusionmon*`: считаем тип покемона ровно как мод.

ПРОБЛЕМА
--------
Showdown **не сообщает** нестандартный тип фьюжна заранее: тип приходит только когда
покемон выходит на поле — `|-start|<ident>|typechange|<t1>/<t2>|[silent]`. poke-env кладёт
эти типы в `Pokemon._temporary_types` и **очищает их при свитче** (`Pokemon.switch_out`).
Итог:

* пока мон на поле — `type_1/type_2` верные (тип фьюжна);
* ушёл в скамейку — снова дексовые типы «головы» (у фьюжна они другие!);
* ещё не выходил — тип фьюжна вообще неизвестен.

Поэтому врал весь урон, который считается по покемонам ВНЕ поля: матрица 6x6 «их j -> наш i»
(`damage.DAMAGE_BLOCK` [51:87]), блок «их известные приёмы x наши 6 слотов» ([99:123]),
`_weakness_score` / `_vulnerability_frac`, мультитипы в obs, STAB-флаг приёма и т.д.

РЕШЕНИЕ
-------
Считаем тип фьюжна сами — по формуле мода (она детерминирована и использует только то,
что у нас есть локально):

    fuseTypes(types1, types2):
        types = [types1[0], types2[1] ? types2[1] : types2[0]].filter(Boolean)
        if (types[0] === types[1]) types.pop()          # одинаковые — схлопываем в один
        if (types.length === 0) types = [types1[0] || "Normal"]

    parseName(pokemon):
        name    = pokemon.name.substring(1, 20)         # мод вешает на партнёра префикс "+"
        species = CutDexMap.get(name)                   # вид по подстроке в dex-id
        Arceus  (num 493): item.onPlate  + ability multitype -> "Arceus-<Plate>"
        Silvally(num 773): item.onMemory + ability rkssystem -> "Silvally-<Memory>"

где `types1` — типы «головы» (основной вид), `types2` — типы партнёра, распарсенного из
имени. У нас «голова» — это `Pokemon.species` (poke-env берёт её из `details`), а партнёр —
`Pokemon.name` (poke-env хранит отображаемое имя из `|switch|`/`|request|`, а мод пишет там
"+Тело"). Обе половины знает и наша команда целиком (реквест), и любой уже виденный мон
противника, поэтому тип считается и для скамейки.

ПРИОРИТЕТ ИСТОЧНИКОВ (важно: считаем только там, где сервер молчит)
------------------------------------------------------------------
1. террасталлизация  -> типы от poke-env (в бою тип реально такой);
2. `_temporary_types` (мон на поле, есть `typechange`) -> типы от poke-env (правда сервера);
3. формат содержит "fusionmon" и известна пара (голова, партнёр) -> формула мода (этот модуль);
4. иначе -> дексовые типы, как раньше (+ счётчик `fusion_type_unknown`).

Формат берём из `config.BATTLE_FORMAT` (скрипты прогонов подменяют его в рантайме) или из
`PYBOT_BATTLE_FORMAT`; если передан `battle` — из его тега (`battle-gen9fusionmonsrandombattle-…`).
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

# Платы и диски-«памяти»: в dex-записи фьюжна Arceus/Silvally не знают, какой тип взять —
# его задаёт предмет (как в моде). Списки — только типы, реально существующих предметов.
_PLATE_TYPES = {
    "dracoplate": "Dragon", "dreadplate": "Dark", "earthplate": "Ground", "fistplate": "Fighting",
    "flameplate": "Fire", "icicleplate": "Ice", "insectplate": "Bug", "ironplate": "Steel",
    "meadowplate": "Grass", "mindplate": "Psychic", "pixieplate": "Fairy", "skyplate": "Flying",
    "splashplate": "Water", "spookyplate": "Ghost", "stoneplate": "Rock", "toxicplate": "Poison",
    "zapplate": "Electric",
}
_MEMORY_TYPES = {
    "bugmemory": "Bug", "darkmemory": "Dark", "dragonmemory": "Dragon", "electricmemory": "Electric",
    "fairymemory": "Fairy", "fightingmemory": "Fighting", "firememory": "Fire", "flyinmemory": "Flying",
    "flyingmemory": "Flying", "ghostmemory": "Ghost", "grassmemory": "Grass", "groundmemory": "Ground",
    "icememory": "Ice", "poisonmemory": "Poison", "psychicmemory": "Psychic", "rockmemory": "Rock",
    "steelmemory": "Steel", "watermemory": "Water",
}

_ARCEUS_NUM = 493
_SILVALLY_NUM = 773

# кэш dex по поколению: pokedex — тяжёлый объект, берём из GenData один раз
_POKEDEX_CACHE: dict[int, dict] = {}


_TO_ID = None          # ссылка на poke_env.data.normalize.to_id_str (ленивая, один раз)
_NOTE_FN = None        # ссылка на type_utils.note_fusion_event


def _resolve_helpers() -> None:
    """Один раз достаём poke-env/свои хелперы: in-function import в горячем пути дорогой."""
    global _TO_ID, _NOTE_FN
    if _TO_ID is None:
        try:
            from poke_env.data.normalize import to_id_str as _t

            _TO_ID = _t
        except Exception:
            _TO_ID = lambda name: "".join(ch for ch in str(name or "").lower() if ch.isalnum())  # noqa: E731
    if _NOTE_FN is None:
        try:
            from .type_utils import note_fusion_event as _n
        except ImportError:
            try:
                from type_utils import note_fusion_event as _n  # type: ignore
            except Exception:
                _n = lambda kind, key: None  # noqa: E731
        _NOTE_FN = _n


def _to_id_str(name: Any) -> str:
    """to_id_str из poke-env с локальным фолбэком (модуль можно импортировать без poke-env)."""
    if _TO_ID is None:
        _resolve_helpers()
    return str(_TO_ID(str(name or "")) or "")


def pokedex(gen: int = 9) -> dict:
    """dex поколения (кэш): нужен для типов «тела» и поиска вида по имени."""
    key = int(gen or 9)
    cached = _POKEDEX_CACHE.get(key)
    if cached is None:
        try:
            from poke_env.data import GenData

            cached = GenData.from_gen(key).pokedex
        except Exception:
            cached = {}
        _POKEDEX_CACHE[key] = cached
    return cached


# --- определение формата -------------------------------------------------------------

def current_format() -> str:
    """Формат текущего прогона: env-переменная важнее конфига (в конфиге значение по умолчанию)."""
    env = os.environ.get("PYBOT_BATTLE_FORMAT", "").strip()
    if env:
        return env
    try:
        from . import config

        return str(getattr(config, "BATTLE_FORMAT", "") or "")
    except Exception:
        return ""


def is_fusion_format(battle: Any = None, fmt: Optional[str] = None) -> bool:
    """True, если формат — фьюжны (`id` содержит подстроку "fusionmon").

    Тег боя (`battle-gen9fusionmonsrandombattle-…`) точнее конфига: если он есть — берём его,
    иначе config/env. Сравнение по подстроке, как просил пользователь ("fusionmon" покрывает
    и `gen9fusionmonsrandombattle`, и `gen9fusionmonou`).
    """
    if fmt:
        return "fusionmon" in str(fmt).lower()
    tag = str(getattr(battle, "battle_tag", "") or "")
    if tag:
        return "fusionmon" in tag.lower()
    return "fusionmon" in current_format().lower()


# --- разбор имени и вида -------------------------------------------------------------

def partner_name(display_name: Any) -> Optional[str]:
    """Имя партнёра из отображаемого имени: "+Тело" -> "Тело" (иначе None).

    Ровно как `parseName` в моде: `pokemon.name.substring(1, 20)`. Префикс "+" — единственный
    признак того, что имя — это фьюжн-партнёр, поэтому без него (обычный ник) фьюжн не считаем.
    """
    name = str(display_name or "").strip()
    if len(name) < 2 or not name.startswith("+"):
        return None
    body = name[1:20].strip()
    return body or None


_SPECIES_CACHE: dict[tuple, Tuple[Optional[str], str]] = {}


def resolve_species(name: Any, gen: int = 9) -> Tuple[Optional[str], str]:
    """Вид по имени: (species_id, как_нашли). Точное совпадение, иначе подстрока (CutDexMap)."""
    sid = _to_id_str(name)
    cached = _SPECIES_CACHE.get((int(gen or 9), sid))
    if cached is not None:
        return cached
    out = _resolve_species_uncached(sid, gen)
    if len(_SPECIES_CACHE) > 4096:
        _SPECIES_CACHE.clear()
    _SPECIES_CACHE[(int(gen or 9), sid)] = out
    return out


def _resolve_species_uncached(sid: str, gen: int = 9) -> Tuple[Optional[str], str]:
    """Точное совпадение, иначе первый вид, чей id содержит строку (CutDexMap мода)."""
    if not sid:
        return None, "пустое имя"
    dex = pokedex(gen)
    if not dex:
        return None, "dex недоступен"
    if sid in dex:
        return sid, "точно"
    # CutDexMap: первый вид, чей id содержит строку (у нас порядок ключей poke-env, у мода — dex;
    # точное совпадение выше покрывает все реальные имена, подстрока — фолбэк как у мода)
    for key in dex:
        if sid in key:
            return key, f"по подстроке '{sid}'"
    return None, f"вид '{sid}' не найден"


_DEX_TYPES_CACHE: dict[tuple, Tuple[str, ...]] = {}


def dex_types(species_id: Optional[str], gen: int = 9) -> Tuple[str, ...]:
    """Типы вида из dex: ("Grass", "Flying") / ("Rock",). Результат кэшируется."""
    if not species_id:
        return ()
    sid = _to_id_str(species_id)
    key = (int(gen or 9), sid)
    hit = _DEX_TYPES_CACHE.get(key)
    if hit is None:
        entry = pokedex(gen).get(sid) or {}
        hit = tuple(str(t) for t in (entry.get("types") or ()) if t)
        _DEX_TYPES_CACHE[key] = hit
    return hit


def partner_species_of(mon: Any, gen: int = 9) -> Tuple[Optional[str], str]:
    """Вид-партнёр фьюжна у покемона: (species_id, как_нашли) либо (None, причина).

    Учитывает спец-случаи мода: Arceus с платой и Multitype, Silvally с диском и RKS System.
    """
    raw = partner_name(getattr(mon, "name", None))
    if raw is None:
        return None, "у мон нет '+'-имени (не фьюжн)"
    sid, how = resolve_species(raw, gen)
    if sid is None:
        return None, how
    dex = pokedex(gen)
    num = (dex.get(sid) or {}).get("num")
    item = _to_id_str(getattr(mon, "item", None))
    ability = _to_id_str(getattr(mon, "ability", None))
    if num == _ARCEUS_NUM:
        plate = _PLATE_TYPES.get(item)
        return (f"arceus{_to_id_str(plate)}" if plate and ability == "multitype" else "arceus"), how
    if num == _SILVALLY_NUM:
        memory = _MEMORY_TYPES.get(item)
        return (f"silvally{_to_id_str(memory)}" if memory and ability == "rkssystem" else "silvally"), how
    return sid, how


# --- формула мода --------------------------------------------------------------------

def fuse_type_names(head_types, partner_types) -> Tuple[str, ...]:
    """`fuseTypes(types1, types2)` из мода, дословно.

    [первый тип головы, второй тип партнёра (или первый, если у партнёра один тип)],
    схлопываем одинаковые, если ничего не осталось — [первый тип головы || "Normal"].
    """
    head = [str(t) for t in (head_types or ()) if t]
    partner = [str(t) for t in (partner_types or ()) if t]
    first = head[0] if head else None
    second = partner[1] if len(partner) > 1 else (partner[0] if partner else None)
    types = [t for t in (first, second) if t]
    if len(types) == 2 and types[0] == types[1]:
        types.pop()
    if not types:
        types = [first or "Normal"]
    return tuple(types)


_POKEMON_TYPE = None


def _pokemon_types(names: Tuple[str, ...], gen: int = 9) -> Tuple[Any, Any]:
    """Имена типов -> объекты PokemonType (как отдаёт poke-env)."""
    global _POKEMON_TYPE
    try:
        if _POKEMON_TYPE is None:
            from poke_env.battle.pokemon_type import PokemonType

            _POKEMON_TYPE = PokemonType
        parsed = [_POKEMON_TYPE.from_name(n) for n in names]
    except Exception:
        return None, None
    if not parsed:
        return None, None
    return parsed[0], (parsed[1] if len(parsed) > 1 else None)


def _type_names(types) -> Tuple[str, ...]:
    names = []
    for t in types:
        if t is None:
            continue
        name = str(getattr(t, "name", t) or "")
        if name:
            names.append(name)
    return tuple(names)


def _note(kind: str, key: str) -> None:
    if _NOTE_FN is None:
        _resolve_helpers()
    _NOTE_FN(kind, key)


_FUSION_TYPES_CACHE: dict[tuple, Tuple[Any, Any, str]] = {}


def effective_types(mon: Any, battle: Any = None, gen: int = 9,
                    fmt: Optional[str] = None) -> Tuple[Any, Any, str]:
    """Типы покемона с учётом фьюжнов: (type_1, type_2, источник).

    Источник — человекочитаемая строка ("server:tera", "server:typechange", "fusion:…", "dex"),
    она же уходит в диагностику, чтобы в логе было видно, откуда взялся тип.
    """
    if mon is None:
        return None, None, "нет покемона"
    t1, t2 = getattr(mon, "type_1", None), getattr(mon, "type_2", None)

    # 1-2) сервер уже сказал тип: тера или typechange на поле — это правда, не пересчитываем
    if getattr(mon, "_terastallized", False) and t1 is not None:
        return t1, t2, "server:tera"
    if getattr(mon, "_temporary_types", None):
        return t1, t2, "server:typechange"

    if not is_fusion_format(battle, fmt=fmt):
        return t1, t2, "dex"

    # 3) считаем тип фьюжна сами (скамейка/мон без typechange). Всё, от чего зависит расчёт
    # (вид, имя-«тело», предмет и способность для Arceus/Silvally), статично -> кэшируем по ним;
    # состояние (тера/typechange) проверено выше и в кэш не попадает
    # в ключ входят и фолбэк-типы самого poke-env: расчёт от них не зависит, только когда
    # партнёр распознан, — а для моков/неизвестных мон (species без партнёра) они и есть результат
    key = (int(gen or 9), _to_id_str(getattr(mon, "species", None)),
           str(getattr(mon, "name", "") or ""), _to_id_str(getattr(mon, "item", None)),
           _to_id_str(getattr(mon, "ability", None)), _type_names((t1, t2)))
    hit = _FUSION_TYPES_CACHE.get(key)
    if hit is not None:
        return hit
    out = _compute_fusion_types(mon, t1, t2, gen=gen)
    if len(_FUSION_TYPES_CACHE) > 4096:
        _FUSION_TYPES_CACHE.clear()
    _FUSION_TYPES_CACHE[key] = out
    return out


def _compute_fusion_types(mon: Any, t1: Any, t2: Any, gen: int = 9) -> Tuple[Any, Any, str]:
    """Собственно расчёт (см. effective_types): голова из details, тело из '+'-имени."""
    # Голова — вид из details: берём его dex-типы (тот же результат, что у poke-env, но в
    # «человеческом» регистре сервера для логов), фолбэк — типы от poke-env
    head_names = dex_types(_to_id_str(getattr(mon, "species", None)), gen=gen) or _type_names((t1, t2))
    if not head_names or any(n in ("THREE_QUESTION_MARKS", "STELLAR") for n in head_names):
        return t1, t2, "dex"
    partner, how = partner_species_of(mon, gen=gen)
    if partner is None:
        _note("fusion_type_unknown",
              f"{getattr(mon, 'species', '?')} ({how}) — тип взят из декса")
        return t1, t2, "dex (партнёр не распознан)"
    names = fuse_type_names(head_names, dex_types(partner, gen=gen))
    if not names:
        return t1, t2, "dex"
    fu1, fu2 = _pokemon_types(names, gen=gen)
    if fu1 is None:
        return t1, t2, "dex"
    _note("fusion_type_used",
          f"{_to_id_str(getattr(mon, 'species', '?'))}+{partner} -> "
          f"{'/'.join(names)} (сервер тип ещё не сообщал)")
    return fu1, fu2, f"fusion:{_to_id_str(getattr(mon, 'species', '?'))}+{partner}"


def effective_type_names(mon: Any, battle: Any = None, gen: int = 9,
                         fmt: Optional[str] = None) -> Tuple[str, ...]:
    """То же, что `effective_types`, но строками имён типов (для логов/тестов)."""
    t1, t2, _ = effective_types(mon, battle=battle, gen=gen, fmt=fmt)
    return _type_names((t1, t2))


def fusion_pair(mon: Any, gen: int = 9) -> Tuple[Optional[str], Optional[str]]:
    """(голова, партнёр) у мон: голова — вид из poke-env, партнёр — из '+'-имени."""
    head = _to_id_str(getattr(mon, "species", None)) or None
    partner, _ = partner_species_of(mon, gen=gen)
    return head, partner
