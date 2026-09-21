"""Раскладка действий: два режима работы — `indices` (poke-env) и `embed` (по кандидатам).

**indices** (по умолчанию, обратная совместимость). Ровно то, что делает poke-env `SinglesEnv`:
0..5 — свитч, индекс в `battle.team.values()`; 6..9 — приём (слот known_moves); 22..25 — тот же
приём с терой. Порядок `battle.team` — это порядок, в котором команда пришла в бой, и он
произволен от боя к бою.

**embed** (второй режим). Смысл действий 0..5 меняется на «j-й резерв в КАНОНИЧЕСКОМ порядке»
(тот же порядок, в котором резервы лежат в bench-блоке признаков, см.
`features.canonical_reserves`). Движки/тера-действия не меняются. Зачем: политика режима embed
оценивает кандидатов по их признакам, поэтому признаки кандидата-свитча и действие, которым его
выбирают, обязаны означать ОДНО И ТО ЖЕ. В режиме indices это не так: bench-блок отсортирован
канонически, а действия нумеруются порядком team, поэтому «свитч в слот 2» в одном бою —
Garchomp, в другом — Ferrothorn, и та же пара (obs, action) ведёт к разным последствиям.

Датасеты для двух режимов НЕ взаимозаменяемы: метки свитчей в них означают разные вещи. Режим
пишется в сайдкар датасета (`<датасет>.meta.json`), см. `training.validate_bc_dataset`.

Реализация намеренно тонкая: логика приёмов/маски/валидации остаётся в poke-env, а наши
функции только переводят индексы свитчей из одной нумерации в другую. Так невозможно
разойтись с poke-env в мелочах (SPECIAL_MOVES, reviving, trapped, base_species).
"""
import os

try:
    from .features import canonical_reserves
except ImportError:  # запуск модуля вне пакета
    from features import canonical_reserves  # type: ignore

VALID_MODES = ("indices", "embed")
DEFAULT_MODE = "indices"

# Оригиналы poke-env СНИМАЕМ ДО патчей (config.py патчит SinglesEnv через этот модуль)
from poke_env.environment.singles_env import SinglesEnv  # noqa: E402
from poke_env.player.battle_order import DefaultBattleOrder  # noqa: E402

_ORIGINAL_GET_ACTION_MASK = SinglesEnv.get_action_mask
_ORIGINAL_ACTION_TO_ORDER = SinglesEnv.action_to_order
_ORIGINAL_ORDER_TO_ACTION = SinglesEnv.order_to_action

# индексы 0..5 — свитчи, 6..25 — приёмы/движки, поэтому «зона свитчей» = [0, 6)
NUM_SWITCH_ACTIONS = 6

_MODE: str | None = None


def normalize_mode(mode: str | None) -> str:
    m = str(mode or DEFAULT_MODE).strip().lower()
    if m not in VALID_MODES:
        raise ValueError(f"неизвестный режим действий {mode!r}; допустимые: {', '.join(VALID_MODES)}")
    return m


def get_action_mode() -> str:
    """Текущий режим: переменная окружения PYBOT_ACTION_MODE, по умолчанию indices."""
    global _MODE
    if _MODE is None:
        _MODE = normalize_mode(os.environ.get("PYBOT_ACTION_MODE", DEFAULT_MODE))
    return _MODE


def set_action_mode(mode: str | None) -> str:
    """Задать режим (CLI/тесты). Влияет на маску и на разбор ордеров во всех путях сразу."""
    global _MODE
    _MODE = normalize_mode(mode) if mode is not None else None
    return get_action_mode()


def reset_action_mode() -> None:
    """Сбросить кэш режима (для тестов: снова читаем PYBOT_ACTION_MODE)."""
    global _MODE
    _MODE = None


def embed_reserves(battle) -> list:
    """Резервы в порядке, которым нумеруются действия-свитчи в режиме embed."""
    return canonical_reserves(getattr(battle, "team", {}) or {})


def _team_order(battle) -> list:
    return list((getattr(battle, "team", {}) or {}).values())


def embed_index_for_team_index(battle, team_index: int):
    """Индекс в нумерации embed для монстра, стоящего на team_index (или None)."""
    team = _team_order(battle)
    if team_index < 0 or team_index >= len(team):
        return None
    mon = team[team_index]
    for j, reserve in enumerate(embed_reserves(battle)):
        if reserve is mon:
            return j
    return None


def get_action_mask(battle) -> list:
    """Маска действий (список 0/1) в текущем режиме.

    В embed переводим НОМЕРА свитчей: маска poke-env разрешает team-индексы, а действия
    embed — индексы канонического порядка резервов. Всё остальное (приёмы, тера, wait-режим,
    trapped) берём у poke-env как есть.
    """
    base = list(_ORIGINAL_GET_ACTION_MASK(battle))
    if get_action_mode() != "embed":
        return base
    if getattr(battle, "_wait", False):
        return base                       # wait: единственное разрешённое действие — индекс 0
    reserves = embed_reserves(battle)
    if not reserves:
        return base
    team = _team_order(battle)
    # какие team-индексы маска разрешила (свитчи) + сопоставление по виду (как делает poke-env)
    allowed_by_species: dict[str, list[int]] = {}
    for i in range(min(NUM_SWITCH_ACTIONS, len(team))):
        if base[i]:
            species = getattr(team[i], "base_species", None) or getattr(team[i], "species", "")
            allowed_by_species.setdefault(str(species), []).append(i)
    out = list(base)
    for i in range(min(NUM_SWITCH_ACTIONS, len(out))):
        out[i] = 0
    for j, reserve in enumerate(reserves[:NUM_SWITCH_ACTIONS - 1]):
        species = str(getattr(reserve, "base_species", None) or getattr(reserve, "species", ""))
        pool = allowed_by_species.get(species) or []
        if pool:
            pool.pop(0)
            out[j] = 1
    return out


def action_to_order(action, battle, fake: bool = False, strict: bool = True):
    """Действие -> BattleOrder с учётом режима."""
    if get_action_mode() != "embed":
        return _ORIGINAL_ACTION_TO_ORDER(action, battle, fake=fake, strict=strict)
    try:
        act = int(action.item()) if hasattr(action, "item") else int(action)
    except Exception:
        act = int(action)
    if act < 0 or act >= NUM_SWITCH_ACTIONS:
        return _ORIGINAL_ACTION_TO_ORDER(action, battle, fake=fake, strict=strict)
    # свитч: находим team-индекс канонического резерва и отдаём его poke-env — так вся
    # валидация (valid_orders, trapped, reviving) остаётся родной
    reserves = embed_reserves(battle)
    if act >= len(reserves):
        if getattr(battle, "_wait", False):
            return DefaultBattleOrder()
        raise ValueError(
            f"embed: свитч-действие {act}, а резервов только {len(reserves)} "
            f"(канонический порядок; такого кандидата нет)")
    target = reserves[act]
    team = _team_order(battle)
    for i, mon in enumerate(team):
        if mon is target:
            return _ORIGINAL_ACTION_TO_ORDER(i, battle, fake=fake, strict=strict)
    raise ValueError(f"embed: резерв {getattr(target, 'species', '?')} не найден в battle.team")


def order_to_action(order, battle, fake: bool = False, strict: bool = True):
    """BattleOrder -> действие с учётом режима (для сбора датасетов и диагностики)."""
    idx = _ORIGINAL_ORDER_TO_ACTION(order, battle, fake=fake, strict=strict)
    if get_action_mode() != "embed":
        return idx
    try:
        value = int(idx.item()) if hasattr(idx, "item") else int(idx)
    except Exception:
        return idx
    if value < 0 or value >= NUM_SWITCH_ACTIONS:
        return idx
    j = embed_index_for_team_index(battle, value)
    return j if j is not None else value


def describe() -> str:
    """Строка для логов: какой режим и что это значит для свитчей."""
    mode = get_action_mode()
    if mode == "embed":
        return ("embed: скор кандидатов по признакам; свитч-действие j = j-й резерв в "
                "каноническом порядке (как bench-блок)")
    return "indices: раскладка poke-env (свитч-действие = индекс в порядке battle.team)"
