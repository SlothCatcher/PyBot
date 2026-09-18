"""Безопасный расчёт типовой эффективности + диагностика "неизвестных" типов.

ПРОБЛЕМА (воспроизведена на poke-env 0.16.1)
-------------------------------------------
`PokemonType.damage_multiplier` считает так:

    if self in {THREE_QUESTION_MARKS, STELLAR} or type_1 in {THREE_QUESTION_MARKS, STELLAR}:
        return 1
    damage_multiplier = type_chart[type_1.name][self.name]      # <- тут KeyError
    if type_2 is not None:
        return damage_multiplier * type_chart[type_2.name][self.name]
    return damage_multiplier

Чарт 9-го поколения (`GenData.from_gen(9).type_chart`) знает только 18 стандартных типов —
ни `STELLAR`, ни `THREE_QUESTION_MARKS` в нём нет (проверено: `python -c "..."`).

Отсюда два разных бага, оба маскируют иммунитет:

1) `type_2` = `???`/`STELLAR` (неизвестный второй тип фьюжна), а `type_1` нормальный:

       ELECTRIC.damage_multiplier(GROUND, THREE_QUESTION_MARKS, type_chart=chart)
       -> KeyError('THREE_QUESTION_MARKS')

   Монки-патч в `config.py` ловил KeyError и возвращал `1.0` на ВЕСЬ расчёт, то есть
   иммунитет Electric vs Ground (0.0) превращался в "нейтрально 1.0" — модель не видит
   иммунитет и спокойно спамит бесполезный приём по земляному покемону.

2) `type_1` = `???` (неизвестный первый тип) — KeyError нет, но poke-env сам возвращает
   `1` заранее, что тоже даёт "нейтрально" для всех приёмов.

ФИКС
----
Считаем покомпонентно: неизвестный тип защиты даёт множитель 1.0 только за СЕБЯ, а
известные компоненты сохраняют свой вклад — поэтому 0.0 от Ground никуда не исчезает:

    ELECTRIC vs (GROUND, ???)  -> 0.0 (иммунитет сохранён, было 1.0)
    ELECTRIC vs (GROUND, None) -> 0.0
    ELECTRIC vs (??? , None)   -> 1.0 (тип реально неизвестен, честный нейтрал + флаг unknown)

ДИАГНОСТИКА
-----------
`PYBOT_DEBUG_TYPES=1` включает подробный лог (сырые `-start|typechange` сообщения сервера,
сводки). Первое срабатывание "иммунитет сохранён" и "неизвестный тип" печатается всегда —
это подтверждение того, что баг реально стрелял в бою.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Optional, Tuple

# Типы, которых нет в чарте: сервер отдаёт "???" для нераспознанного типа фьюжна,
# STELLAR есть в poke-env, но отсутствует в чарте gen9 (проверено).
_UNKNOWN_TYPE_NAMES = {"THREE_QUESTION_MARKS", "STELLAR"}

_DEBUG = os.environ.get("PYBOT_DEBUG_TYPES", "").strip().lower() not in ("", "0", "false", "no", "off")

_chart_cache: dict[int, dict] = {}

_counts: dict[str, Counter] = {
    "unknown_def_type": Counter(),   # у защиты реально неизвестный тип
    "masked_immunity": Counter(),    # старый патч вернул бы 1.0 вместо иммунитета 0.0
    "keyerror": Counter(),           # сырой KeyError из оригинального poke-env
    "typechange_raw": Counter(),     # сырые -start|typechange от сервера (только debug)
}


def debug_enabled() -> bool:
    return _DEBUG


def _type_name(t: Any) -> str:
    if t is None:
        return "None"
    return str(getattr(t, "name", t))


def is_unknown_type(t: Any) -> bool:
    """True для типов, которых нет в чарте (??? / STELLAR)."""
    return _type_name(t) in _UNKNOWN_TYPE_NAMES


def get_type_chart(gen: int = 9) -> dict:
    """Кэшированный чарт (ленивая загрузка — GenData тяжёлый)."""
    if gen not in _chart_cache:
        from poke_env.data import GenData

        _chart_cache[gen] = GenData.from_gen(gen).type_chart
    return _chart_cache[gen]


def _note(kind: str, key: str, limit_unique: int = 8) -> None:
    """Считает событие; первое уникальное событие печатает (кроме сырых сообщений — они под debug)."""
    c = _counts.setdefault(kind, Counter())
    c[key] += 1
    if c[key] != 1:
        return
    if kind == "typechange_raw" and not _DEBUG:
        return
    if kind == "masked_immunity":
        print(f"[type-fix] иммунитет сохранён там, где старый патч давал 1.0: {key}")
    elif kind == "keyerror":
        print(f"[type-fix] KeyError в damage_multiplier (пойман фабрикой): {key}")
    elif kind == "unknown_def_type":
        print(f"[type-fix] неизвестный тип у защиты: {key}")
    elif kind == "typechange_raw":
        print(f"[type-debug] typechange от сервера: {key}")


def damage_multiplier_safe_ex(
    move_type: Any,
    def_type_1: Any,
    def_type_2: Any = None,
    type_chart: Optional[dict] = None,
) -> Tuple[float, bool]:
    """Покомпонентный множитель. Возвращает (mult, unknown).

    unknown=True — хотя бы один компонент типа был неизвестен (??? / STELLAR) либо
    отсутствовал в чарте; сам множитель при этом считается по известным компонентам.
    """
    if move_type is None:
        return 1.0, True

    atk_name = _type_name(move_type)
    if atk_name in _UNKNOWN_TYPE_NAMES:
        # poke-env тоже возвращает 1 для неизвестного/стелларового атакующего типа
        return 1.0, True
    if atk_name == "None":
        return 1.0, True

    if type_chart is None:
        type_chart = get_type_chart(9)

    unknown = False
    mult = 1.0
    known_components = 0
    for t in (def_type_1, def_type_2):
        if t is None:
            continue
        t_name = _type_name(t)
        if t_name in _UNKNOWN_TYPE_NAMES:
            # неизвестный тип защиты: вклад нейтральный, но остальные компоненты НЕ трогаем
            unknown = True
            continue
        try:
            mult *= type_chart[t_name][atk_name]
        except (KeyError, TypeError):
            unknown = True
            continue
        known_components += 1

    if known_components == 0:
        # тип защиты вообще не распознан — честный нейтрал + пометка unknown
        return 1.0, True

    if unknown:
        _note("unknown_def_type", f"{_type_name(def_type_1)}/{_type_name(def_type_2)}")
        if mult == 0:
            # ровно тот случай, где старый патч (return 1.0) прятал иммунитет
            _note("masked_immunity", f"{atk_name} vs {_type_name(def_type_1)}/{_type_name(def_type_2)} -> 0.0")

    return float(mult), unknown


def damage_multiplier_safe(
    move_type: Any,
    def_type_1: Any,
    def_type_2: Any = None,
    type_chart: Optional[dict] = None,
) -> float:
    """Как poke-env `damage_multiplier`, но не маскирует иммунитет при неизвестном типе."""
    return damage_multiplier_safe_ex(move_type, def_type_1, def_type_2, type_chart)[0]


def note_keyerror(self_type: Any, type_1: Any, type_2: Any, err: BaseException) -> None:
    _note("keyerror", f"{_type_name(self_type)} vs {_type_name(type_1)}/{_type_name(type_2)}: {err}")


def note_typechange_raw(message: str) -> None:
    """Сырое `-start|typechange` сообщение (для выяснения, что присылает сервер)."""
    _note("typechange_raw", message)


def summary() -> str:
    parts = []
    for kind, c in _counts.items():
        if not c:
            continue
        total = sum(c.values())
        top = ", ".join(f"{k} x{v}" for k, v in c.most_common(3))
        parts.append(f"{kind}: {total} ({top})")
    return " | ".join(parts) if parts else ""


def summary_line(prefix: str = "[type-debug]") -> str:
    s = summary()
    return f"{prefix} {s}" if s else ""


def reset_counts() -> None:
    for c in _counts.values():
        c.clear()
