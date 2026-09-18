"""Зеркальный урон и флаги эффектов на НАСТОЯЩЕМ `poke_env.battle.Battle`.

Юнит-тесты в `test_damage.py` работают на SimpleNamespace-заглушках, поэтому не проверяют:
  * что приёмы противника вообще попадают в `mon.moves` так, как мы их читаем;
  * что мы НЕ видим нераскрытые приёмы (утечка закрытой информации = читинг);
  * что сторонние экраны применяются в правильную сторону (ctx_ours vs ctx_theirs);
  * что в живом потоке сообщений (switch/move/-sidestart/-status) блок не падает.

Здесь боевые сообщения протокола Showdown скармливаются настоящему `Battle`, как это делает
`Player._handle_battle_message`, и проверяется содержимое obs.

Запуск: python test_mirror_features_live.py
"""
import json
import logging
import sys

from poke_env.battle import Battle

from agents import config as _cfg  # noqa: F401  (монки-патчи)
from agents.config import N_FEATURES
from agents.damage import DAMAGE_BLOCK_SIZE, EFFECT_FLAGS, FLAGS_BASE, MIRROR_BASE, TEAM_BASE
from agents.features import embed_battle_with_fusion

FAILED = []
TAG = "battle-gen9fusionmonsrandombattle-1"
# блок признаков урона добавляется в конец obs, поэтому его локальные индексы сдвинуты
OFF = N_FEATURES - DAMAGE_BLOCK_SIZE
MIRROR = OFF + MIRROR_BASE
TEAM = OFF + TEAM_BASE
FLAGS = OFF + FLAGS_BASE


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def check_true(label, cond, extra=""):
    check(label + (f": {extra}" if extra else ""), bool(cond), True)


def new_battle():
    b = Battle(battle_tag=TAG, username="Me", logger=logging.getLogger("quiet"), gen=9)
    for msg in (["", "player", "p1", "Me", "", ""],
                ["", "player", "p2", "Opp", "", ""],
                ["", "start"]):
        b.parse_message(msg)
    return b


def feed(battle, messages):
    for msg in messages:
        battle.parse_message(msg)
    return battle


def flags(obs):
    return {EFFECT_FLAGS[i]: float(obs[FLAGS + i]) for i in range(len(EFFECT_FLAGS))}


def nonzero(obs):
    return {k: v for k, v in flags(obs).items() if v}


def main() -> int:
    b = new_battle()
    feed(b, [
        ["", "poke", "p2", "skarmory, L50, F"],          # превью команды: приёмов не знаем
        ["", "switch", "p1a: Aqua", "swampert, L50, M", "100/100"],
        ["", "switch", "p2a: Rock", "steelix, L50, F", "100/100"],
        ["", "turn", "1"],
    ])
    obs0 = embed_battle_with_fusion(b, None, None)
    check("obs размер = N_FEATURES", len(obs0), N_FEATURES)
    check("утечки нет: приёмы противника до их использования неизвестны",
          len(b.opponent_active_pokemon.moves), 0)
    check_true("утечки нет: превью не добавляет монта в opponent_team",
               len(b.opponent_team) == 1, f"в команде {list(b.opponent_team)}")
    check_true("утечки нет: ни у одного монта команды противника нет приёмов",
               all(len(m.moves) == 0 for m in b.opponent_team.values()))
    check_true("без раскрытых приёмов зеркало нулевое", float(obs0[MIRROR:MIRROR + 12].max()) == 0.0)
    check_true("без раскрытых приёмов флаги нулевые", float(obs0[FLAGS:].max()) == 0.0)

    # наш собственный приём: виден нам сразу, как в настоящем бою
    feed(b, [
        ["", "move", "p1a: Aqua", "earthquake", "p2a: Rock"],
        ["", "-damage", "p2a: Rock", "70/100"],
    ])
    # противник раскрывает атакующий и статусный приёмы
    feed(b, [
        ["", "move", "p2a: Rock", "earthquake", "p1a: Aqua"],
        ["", "-damage", "p1a: Aqua", "55/100"],
        ["", "move", "p2a: Rock", "stealthrock", "p1a: Aqua"],
        ["", "-sidestart", "p1: Me", "Stealth Rock"],
    ])
    obs1 = embed_battle_with_fusion(b, None, None)
    check("раскрылись ровно 2 приёма противника", len(b.opponent_active_pokemon.moves), 2)
    check_true("зеркало: их earthquake бьёт по нашему активному", float(obs1[MIRROR]) > 0.0,
               f"min_frac={float(obs1[MIRROR]):.3f}")
    check_true("зеркало: earthquake идёт в слот 0 (статусный stealthrock урона не даёт)",
               float(obs1[MIRROR]) > float(obs1[MIRROR + 3]) == 0.0,
               f"слот0={float(obs1[MIRROR]):.3f} слот1={float(obs1[MIRROR + 3]):.3f}")
    match_row = [float(obs1[TEAM + j]) for j in range(6)]
    check_true("зеркало: наш активный виден в строке по слотам (совпадение с активным срезом)",
               any(abs(v - float(obs1[MIRROR])) < 1e-6 for v in match_row), f"row={match_row}")
    check("флаги из живого боя: hazard_stealthrock", flags(obs1)["hazard_stealthrock"], 1.0)
    check("флаги из живого боя: count_hazard_moves", flags(obs1)["count_hazard_moves"], 0.25)
    check("флаги: статусных приёмов нет (у steelix их пока не раскрыли)", flags(obs1)["status_any"], 0.0)

    # ---- экраны: направление контекста (их Reflect не должен защищать НАС) ----
    our_out_before = float(obs1[OFF + 15])          # матрица «наш -> их»
    mirror_before = float(obs1[MIRROR])
    feed(b, [
        ["", "move", "p2a: Rock", "reflect", "p1a: Aqua"],
        ["", "-sidestart", "p2: Opp", "Reflect"],
    ])
    obs2 = embed_battle_with_fusion(b, None, None)
    check_true("их Reflect режет наш урон (ctx_ours смотрит на их экраны)",
               float(obs2[OFF + 15]) < our_out_before,
               f"было={our_out_before:.3f} стало={float(obs2[OFF + 15]):.3f}")
    check_true("их Reflect НЕ трогает зеркальный урон (их экран нас не защищает)",
               abs(float(obs2[MIRROR]) - mirror_before) < 1e-6,
               f"было={mirror_before:.3f} стало={float(obs2[MIRROR]):.3f}")
    check("флаги: их Reflect виден как screens", flags(obs2)["screens"], 1.0)

    feed(b, [["", "-sidestart", "p1: Me", "Reflect"]])
    obs3 = embed_battle_with_fusion(b, None, None)
    check_true("наш Reflect режет входящий урон (ctx_theirs смотрит на НАШИ экраны)",
               float(obs3[MIRROR]) < float(obs2[MIRROR]),
               f"было={float(obs2[MIRROR]):.3f} стало={float(obs3[MIRROR]):.3f}")
    check_true("наш Reflect НЕ трогает наш исходящий урон",
               abs(float(obs3[OFF + 15]) - float(obs2[OFF + 15])) < 1e-6)

    # ---- статус на нас: флаг состояния на стороне противника ----
    feed(b, [
        ["", "move", "p2a: Rock", "willowisp", "p1a: Aqua"],
        ["", "-status", "p1a: Aqua", "brn"],
        ["", "turn", "2"],
    ])
    obs4 = embed_battle_with_fusion(b, None, None)
    check("флаги: Will-O-Wisp раскрылся как status_burn", flags(obs4)["status_burn"], 1.0)
    fl4 = flags(obs4)
    check("флаги: счётчик статусных приёмов (SR + Reflect + WoW из 4)", fl4["count_status_moves"], 0.75)
    check("флаги: счётчик атакующих приёмов (только earthquake)", fl4["count_damaging_moves"], 0.25)
    check_true("флаги: их экран учтён как screens, но не как опасность урона",
               fl4["screens"] == 1.0 and fl4["status_any"] == 1.0)
    check_true("ожог атакующего не ломает зеркало",
               float(obs4[MIRROR]) > 0.0 or float(obs4[MIRROR + 3]) > 0.0)

    # ---- смена активного у противника: флаги по всей команде, зеркало — по активному ----
    feed(b, [
        ["", "switch", "p2a: Skarm", "skarmory, L50, F", "100/100"],
        ["", "move", "p2a: Skarm", "spikes", "p1a: Aqua"],
        ["", "-sidestart", "p1: Me", "Spikes"],
    ])
    obs5 = embed_battle_with_fusion(b, None, None)
    fl5 = flags(obs5)
    check("флаги: hazard со скамейки (stealthrock от steelix) сохранился", fl5["hazard_stealthrock"], 1.0)
    check("флаги: новый сеттер виден (spikes)", fl5["hazard_spikes"], 1.0)
    check("флаги: счётчик hazard-приёмов вырос", fl5["count_hazard_moves"], 0.5)
    check_true("зеркало: у нового активного только статусный приём -> урона нет",
               float(obs5[MIRROR]) == 0.0, f"слот0={float(obs5[MIRROR]):.3f}")
    check_true("мой активный в бою тот же, obs конечен", bool(obs5.sum() == obs5.sum()))

    # ---- сломанных путей логирования быть не должно ----
    from agents import features as _f
    check("[damage] ошибки блока не было", getattr(_f, "_DAMAGE_BLOCK_ERROR_LOGGED", False), False)

    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
