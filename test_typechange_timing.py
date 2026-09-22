"""Гонка typechange: проверка, что решение не уходит со старым типом.

Факт из poke-env (`Player._handle_battle_message`): строки батча обрабатываются ПО ПОРЯДКУ,
и на строке `|request|` сразу вызывается `choose_move` (→ `embed_battle`). Всё, что сервер
прислал в том же батче ПОСЛЕ `|request|`, к моменту решения ещё не применено.

Здесь этот цикл воспроизводится на настоящем `Battle`:
  1. `typechange` после `|request|` -> без pre-apply решение видит СТАРЫЙ тип (баг);
  2. он же с pre-apply (`agents.fusion_parser._preapply_from_batch`) -> НОВЫЙ тип (фикс);
  3. `typechange` до `|request|` -> в обоих случаях корректно (pre-apply ничего не портит);
  4. итоговое состояние боя после полного батча одинаковое (двойного применения нет).

Запуск: python test_typechange_timing.py
"""
import json
import logging
import sys
import types as _t

from poke_env.battle import Battle
from poke_env.battle import PokemonType as PT

from agents.fusion_parser import _hoist_typechanges_before_request, _collect_typechanges
from agents import config as _cfg  # noqa: F401  (монки-патчи)

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def types_of(mon):
    return (getattr(getattr(mon, "type_1", None), "name", None),
            getattr(getattr(mon, "type_2", None), "name", None))


def new_battle(tag="battle-gen9fusionmonsrandombattle-1"):
    b = Battle(battle_tag=tag, username="Me", logger=logging.getLogger("quiet"), gen=9)
    for msg in (["", "player", "p1", "Me", "", ""],
                ["", "player", "p2", "Opp", "", ""],
                ["", "start"]):
        b.parse_message(msg)
    return b


def run_batch_like_poke_env(battle, batch, on_decision):
    """Точная копия цикла poke-env: |request| -> choose_move сразу, остальные строки парсятся."""
    for msg in batch[1:]:
        if len(msg) > 1 and msg[1] == "request":
            on_decision(battle)          # здесь poke-env вызывает choose_move
        else:
            battle.parse_message(msg)


def decision_logger(sink):
    def on_decision(battle):
        sink.append((battle.turn, types_of(battle.opponent_active_pokemon)))
    return on_decision


def main() -> int:
    request = ["", "request", json.dumps({"active": [{"moves": []}], "side": {}})]

    # --- 1. Баг: typechange идёт ПОСЛЕ |request| -------------------------------------
    b = new_battle()
    batch = [
        [">battle-gen9fusionmonsrandombattle-1"],
        ["", "switch", "p2a: Frost", "froslass, L50, M", "100/100"],
        request,
        ["", "-start", "p2a: Frost", "typechange", "Steel/Ice", "[silent]"],
    ]
    seen = []
    run_batch_like_poke_env(b, batch, decision_logger(seen))
    print(f"баз (без pre-apply): на решении видели {seen[0][1]}, после батча {types_of(b.opponent_active_pokemon)}")
    check("без pre-apply решение видит СТАРЫЙ тип (демонстрация гонки)", seen[0][1], ("ICE", "GHOST"))
    check("после всего батча тип уже новый", types_of(b.opponent_active_pokemon), ("STEEL", "ICE"))

    # --- 2. Фикс: pre-apply перед обработкой батча ------------------------------------
    b = new_battle()
    reordered, n_moved = _hoist_typechanges_before_request(batch)
    print(f"фикс (перестановка): перенесено строк {n_moved}")
    seen = []
    run_batch_like_poke_env(b, reordered, decision_logger(seen))
    print(f"фикс (перестановка): на решении видели {seen[0][1]}")
    check("перестановка: решение видит НОВЫЙ тип", seen[0][1], ("STEEL", "ICE"))
    check("перестановка: итоговый тип тот же", types_of(b.opponent_active_pokemon), ("STEEL", "ICE"))
    check("перестановка: ничего не потеряно",
          sorted(json.dumps(m) for m in reordered[1:]), sorted(json.dumps(m) for m in batch[1:]))
    check("перестановка: request последний, typechange перед ним",
          [m[1] if len(m) > 1 else "" for m in reordered[1:]],
          ["switch", "-start", "request"])

    # --- 3. Обычный порядок (typechange ДО request) -----------------------------------
    b_ord = new_battle()
    batch_ord = [
        [">battle-gen9fusionmonsrandombattle-1"],
        ["", "switch", "p2a: Frost", "froslass, L50, M", "100/100"],
        ["", "-start", "p2a: Frost", "typechange", "Steel/Ice", "[silent]"],
        request,
    ]
    seen = []
    run_batch_like_poke_env(b_ord, batch_ord, decision_logger(seen))
    check("обычный порядок: тип применён до решения и без фикса", seen[0][1], ("STEEL", "ICE"))
    b2 = new_battle()
    reordered_ord, n2 = _hoist_typechanges_before_request(batch_ord)
    check("обычный порядок: перестановка не нужна (0 переносов)", n2, 0)
    check("обычный порядок: батч не изменён", reordered_ord is batch_ord, True)
    seen2 = []
    run_batch_like_poke_env(b2, batch_ord, decision_logger(seen2))
    check("обычный порядок: тип на решении корректен", seen2[0][1], ("STEEL", "ICE"))

    # --- 4. Итоговое состояние одинаково (нет двойного применения) ---------------------
    check("итог батча совпадает с обычным порядком (нет двойного применения)",
          types_of(b.opponent_active_pokemon), types_of(b_ord.opponent_active_pokemon))
    mon = b.opponent_active_pokemon
    from poke_env.battle import Effect
    check("счётчик эффекта typechange не удвоился", mon.effects.get(Effect.TYPECHANGE, 0), 0)

    # --- 5. "???" от сервера: poke-env даёт TQM (неизвестный тип), это ожидаемо ---------
    b3 = new_battle()
    batch_unk = [["battle-tag"], ["", "switch", "p2a: Frost", "froslass, L50, M", "100/100"],
                 request, ["", "-start", "p2a: Frost", "typechange", "???", "[silent]"]]
    seen3 = []
    run_batch_like_poke_env(b3, batch_unk, decision_logger(seen3))
    check("??? -> THREE_QUESTION_MARKS (тип реально неизвестен)", types_of(b3.opponent_active_pokemon),
          ("THREE_QUESTION_MARKS", None))

    # --- 6. Сбор typechange из батча: флаг after_request -------------------------------
    collected = _collect_typechanges(batch)
    check("_collect_typechanges нашёл 1 запись", len(collected), 1)
    check("флаг after_request=True для гонки", collected[0][2], True)
    check("флаг after_request=False в обычном порядке", _collect_typechanges(batch_ord)[0][2], False)

    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
