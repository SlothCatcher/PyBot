"""Синхронизация типов: то, что видит модель, против того, что прислал сервер.

Проверяет два утверждения, которыми диагностируются жалобы «модель спамит иммунным приёмом»:

  1) `battle.opponent_active_pokemon.type_1/type_2` (а значит и obs) совпадают с ПОСЛЕДНИМ
     `|-start|<ident>|typechange|<types>|[silent]` из лога сервера;
  2) `moves_dmg_multiplier` в obs посчитан по СЕРВЕРНЫМ типам (а не по устаревшим типам боя).

Оба случая ловятся счётчиками в `diagnose_type_spam.py` на живых боях:
`type_mismatch` и `obs_stale`. Здесь те же проверки прогоняются на мок-бое без сервера.

Запуск: python test_type_sync.py
"""
import logging
import sys
from collections import Counter

from poke_env.battle import Battle, Move

from agents import config as _cfg  # noqa: F401  (монки-патчи)
from agents.fusion_parser import _parse_fusion_message
import diagnose_type_spam as diag

FAILED = []
TAG = "battle-gen9fusionmonsrandombattle-9"


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def mk_battle():
    b = Battle(battle_tag=TAG, username="Me", logger=logging.getLogger("quiet"), gen=9)
    for m in (["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""], ["", "start"],
              ["", "switch", "p1a: Our", "stonjourner, L50, M", "100/100"],
              ["", "switch", "p2a: +Snorlax", "snorlax, L50, M", "100/100"]):
        b.parse_message(m)
    return b


def mk_player(check_obs: bool = True):
    p = diag.TypeDiagPlayer.__new__(diag.TypeDiagPlayer)
    p.check_obs = check_obs
    p._logged_battles = set()
    p.stats = {
        "turns": 0, "immune_chosen": 0, "damaging_chosen": 0, "wasted_chosen": 0,
        "chosen_moves": Counter(), "immune_matchups": Counter(), "unknown_type_turns": 0,
        "unknown_pairs": Counter(), "best_missed": 0, "would_mask": 0,
        "type_checks": 0, "type_mismatch": 0, "server_msg_other_mon": 0,
        "obs_checks": 0, "obs_stale": 0, "mismatch_examples": [],
    }
    p.get_fusion_entry = lambda battle, is_ours: None
    p.get_protected_last_turn = lambda battle, is_ours: 0.0
    p.get_team_fusion_map = lambda battle, is_ours: None
    return p


def set_server_msg(ident: str, types: str):
    diag._RAW_TYPECHANGE[TAG] = {"p2": f"|-start|{ident}|typechange|{types}|[silent]"}


def test_mismatch_detected():
    print("--- 1. тип в бою против лога сервера ---")
    # сервер прислал новый тип, но к бою он не применён -> расхождение видно
    b = mk_battle()
    set_server_msg("p2a: +Snorlax", "Poison/Normal")
    p = mk_player()
    p._check_server_vs_battle(b)
    check("неприменённый typechange ловится", (p.stats["type_checks"], p.stats["type_mismatch"]), (1, 1))

    # тип применён -> совпадение
    b = mk_battle()
    _parse_fusion_message({}, {}, [["" + TAG], ["", "-start", "p2a: +Snorlax", "typechange", "Poison/Normal", "[silent]"]])
    b.parse_message(["", "-start", "p2a: +Snorlax", "typechange", "Poison/Normal", "[silent]"])
    p = mk_player()
    p._check_server_vs_battle(b)
    check("применённый typechange совпадает", (p.stats["type_checks"], p.stats["type_mismatch"]), (1, 0))

    # сообщение относится к прошлому покемону (свитч был, typechange ещё нет) -> не mismatch
    b = mk_battle()
    b.parse_message(["", "switch", "p2a: +Heracross", "heracross, L50, M", "100/100"])
    set_server_msg("p2a: +Snorlax", "Poison/Normal")
    p = mk_player()
    p._check_server_vs_battle(b)
    check("typechange другого монстра не считается расхождением",
          (p.stats["type_checks"], p.stats["type_mismatch"], p.stats["server_msg_other_mon"]), (0, 0, 1))

    # наш собственный покемон тоже проверяется (лог сервера по p1)
    b = mk_battle()
    diag._RAW_TYPECHANGE[TAG] = {"p1": "|-start|p1a: Our|typechange|Grass/Rock|[silent]"}
    b.parse_message(["", "-start", "p1a: Our", "typechange", "Grass/Rock", "[silent]"])
    p = mk_player()
    p._check_server_vs_battle(b)
    check("наш покемон тоже сверяется", (p.stats["type_checks"], p.stats["type_mismatch"]), (1, 0))


def test_obs_uses_server_types():
    print("--- 2. obs против серверных типов ---")
    moves = [Move("earthquake", gen=9), Move("thunderbolt", gen=9)]
    # бой синхронизирован с сервером -> obs корректен
    b = mk_battle()
    b.parse_message(["", "-start", "p2a: +Snorlax", "typechange", "Poison/Normal", "[silent]"])
    set_server_msg("p2a: +Snorlax", "Poison/Normal")
    b._available_moves = list(moves)
    p = mk_player()
    p._check_obs_against_server(b, moves)
    check("obs по серверным типам (earthquake vs Poison/Normal = 2x)", (p.stats["obs_checks"], p.stats["obs_stale"]), (2, 0))

    # бой отстал: типы боя = базовые (Normal), сервер = Poison/Normal -> obs устарел
    b = mk_battle()
    b._available_moves = list(moves)
    set_server_msg("p2a: +Snorlax", "Poison/Normal")
    p = mk_player()
    p._check_obs_against_server(b, moves)
    check("устаревший obs ловится", p.stats["obs_stale"] >= 1, True)
    check("проверены все доступные приёмы", p.stats["obs_checks"], 2)


def test_helpers():
    print("--- 3. хелперы разбора серверной строки ---")
    b = mk_battle()
    set_server_msg("p2a: +Snorlax", "Poison/Normal")
    check("_server_types_for вернул пару", diag._server_types_for(b, "p2"), ("p2a: +Snorlax", ("POISON", "NORMAL")))
    set_server_msg("p2a: +Snorlax", "???")
    check("'???' нормализуется", diag._server_types_for(b, "p2"), ("p2a: +Snorlax", ("THREE_QUESTION_MARKS",)))
    check("_to_pokemon_type('???')", diag._to_pokemon_type("???").name, "THREE_QUESTION_MARKS")
    check("_to_pokemon_type('Stellar')", diag._to_pokemon_type("Stellar").name, "STELLAR")
    check("_to_pokemon_type('Rock')", diag._to_pokemon_type("Rock").name, "ROCK")
    diag._RAW_TYPECHANGE.pop(TAG, None)
    check("нет сообщения -> None", diag._server_types_for(b, "p2"), None)


def main() -> int:
    test_mismatch_detected()
    test_obs_uses_server_types()
    test_helpers()
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
