"""Проверка пути ИНФЕРЕНСА (обученная модель) и разбора статов фьюжна.

Отвечает на два вопроса:
  1) успевают ли статы покемона (html-таблица) к моменту решения, если сервер шлёт их
     в том же кадре, что и |request| (и ПОСЛЕ него);
  2) применяется ли фикс тайминга typechange на всех путях, которыми играет обученная
     модель: agents.players.PolicyPlayer (eval/индекс), agents.policy_player_simple.PolicyPlayer,
     а также на игроках внутри ExampleEnv (agent1/agent2 через _attach_fusion_parser).

Тест поведенческий: настоящий poke-env Battle + точная копия цикла обработки кадра
(на |request| сразу «решение»), а не просто проверка наличия атрибутов.

Запуск: python test_inference_paths.py
"""
import asyncio
import json
import logging
import sys

from poke_env.battle import Battle
from poke_env.battle import PokemonType as PT

from agents.fusion_parser import (
    FusionInfoParser,
    _attach_fusion_parser,
    _hoist_typechanges_before_request,
    _parse_fusion_message,
    _parse_protect_message,
)
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


TAG = "battle-gen9fusionmonsrandombattle-1"

HTML_STATS = ("<b>Froslass + Corviknight base stats:</b><table>"
              "<tr><td>70</td><td>80</td><td>90</td><td>100</td><td>110</td><td>120</td></tr>"
              "</table>Possible speed: 200-320")


def make_battle():
    b = Battle(battle_tag=TAG, username="Me", logger=logging.getLogger("quiet"), gen=9)
    for msg in (["", "player", "p1", "Me", "", ""],
                ["", "player", "p2", "Opp", "", ""],
                ["", "start"]):
        b.parse_message(msg)
    return b


def race_batch():
    """Ровно тот порядок, из-за которого возникает гонка: typechange/html ПОСЛЕ |request|."""
    return [
        [">" + TAG],
        ["", "switch", "p2a: Frost", "froslass, L50, M", "100/100"],
        ["", "request", json.dumps({"active": [{"moves": []}], "side": {}})],
        ["", "-start", "p2a: Frost", "typechange", "Steel/Ice", "[silent]"],
        ["", "html", HTML_STATS],
    ]


class StubBase:
    """Копия цикла poke-env: на |request| сразу «решение», остальное — parse_message."""

    def __init__(self, *args, battle=None, on_decision=None, **kwargs):
        self._battles = {TAG: battle} if battle is not None else {}
        self.decided = []
        self._on_decision = on_decision or (lambda b: None)

    async def _handle_battle_message(self, split_messages):
        self.received = [m[1] if len(m) > 1 else "" for m in split_messages[1:]]
        battle = self._battles[TAG]
        for m in split_messages[1:]:
            if len(m) > 1 and m[1] == "request":
                self.decided.append((battle.turn, types_of(battle.opponent_active_pokemon)))
                self._on_decision(battle)
            else:
                battle.parse_message(m)


class Stub(FusionInfoParser, StubBase):
    pass


def test_stats_and_types_reach_decision():
    print("--- 1. что видит решение, если статы/typechange пришли ПОСЛЕ |request| ---")
    battle = make_battle()
    seen_stats = {}

    def on_decision(b):
        # ровно то, что читает embed_battle в момент выбора хода
        seen_stats["entry"] = st._fusion_stats.get(TAG, {}).get("p2")

    st = Stub(battle=battle, on_decision=on_decision)
    asyncio.run(st._handle_battle_message(race_batch()))

    print(f"тип активного оппонента на решении: {st.decided[0][1]}")
    print(f"статы фьюжна на решении: {seen_stats['entry']}")
    check("решение видит НОВЫЙ тип (reorder работает)", st.decided[0][1], ("STEEL", "ICE"))
    check("решение видит статы фьюжна из html того же кадра",
          (seen_stats["entry"] or {}).get("base_stats"),
          {"hp": 70, "atk": 80, "def": 90, "spa": 100, "spd": 110, "spe": 120})
    check("решение видит speed_range", (seen_stats["entry"] or {}).get("speed_range"), (200, 320))
    check("итоговый тип боя тот же (двойного применения нет)", types_of(battle.opponent_active_pokemon), ("STEEL", "ICE"))

    # базовая проверка: БЕЗ фикса (сырой кадр) решение видело бы старый тип и без статов
    battle2 = make_battle()
    seen2 = {}

    def on_decision2(b):
        seen2["entry"] = None  # статы ещё не разобраны? считаем честно: их нет в конце кадра... см. ниже

    class RawBase(StubBase):
        """poke-env как есть, без миксина FusionInfoParser."""

    raw = RawBase(battle=battle2)
    asyncio.run(raw._handle_battle_message(race_batch()))
    check("без фикса: тип на решении СТАРЫЙ (гонка)", raw.decided[0][1], ("ICE", "GHOST"))
    check("без фикса: итоговый тип уже новый", types_of(battle2.opponent_active_pokemon), ("STEEL", "ICE"))


def test_parser_regression():
    """Регресс: _parse_fusion_message/_parse_protect_message обязаны существовать и работать.

    (Один из коммитов уже однажды вырезал их при правке reorder — статы молча терялись.)
    """
    print("--- 2. регресс: разбор статов/протектов на месте ---")
    store, pending = {}, {}
    _parse_fusion_message(store, pending, race_batch())
    check("_parse_fusion_message заполнил base_stats",
          store.get(TAG, {}).get("p2", {}).get("base_stats"),
          {"hp": 70, "atk": 80, "def": 90, "spa": 100, "spd": 110, "spe": 120})
    by_species = store.get(TAG, {}).get("p2_by_species", {})
    print(f"per-species ключи: {sorted(by_species)}")
    check("статы есть в per-species мапе по id из сообщения (frost)",
          by_species.get("frost", {}).get("base_stats") is not None, True)
    check("статы есть и по id из заголовка html (froslasscorviknight)",
          by_species.get("froslasscorviknight", {}).get("base_stats") is not None, True)
    check("pending очищен после html", pending.get(TAG), None)

    protect_state = {}
    _parse_protect_message(protect_state, [
        [">" + TAG],
        ["", "-singleturn", "p2a: Frost", "move: Protect"],
    ])
    check("_parse_protect_message зафиксировал защиту", protect_state.get(TAG, {}).get("p2"), True)

    # html без [silent] у typechange тоже должен разбираться
    store2, pending2 = {}, {}
    no_silent = [
        [">" + TAG],
        ["", "-start", "p2a: Frost", "typechange", "Steel/Ice"],
        ["", "html", HTML_STATS],
    ]
    _parse_fusion_message(store2, pending2, no_silent)
    check("typechange без [silent] тоже даёт статы",
          store2.get(TAG, {}).get("p2", {}).get("base_stats") is not None, True)


def test_wiring_paths():
    print("--- 3. фикс подключён на всех путях инференса ---")
    # 3.1 миксин: батч, который получает poke-env (super()), уже переставлен
    battle = make_battle()
    st = Stub(battle=battle)
    asyncio.run(st._handle_battle_message(race_batch()))
    check("миксин: poke-env получает typechange ДО request",
          getattr(st, "received", None), ["switch", "-start", "request", "html"])

    # 3.2 _attach_fusion_parser (agent1/agent2 внутри ExampleEnv)
    class Plain:
        def __init__(self):
            pass

        async def _handle_battle_message(self, split_messages):
            self.seen = [m[1] if len(m) > 1 else "" for m in split_messages[1:]]

    p = Plain()
    _attach_fusion_parser(p)
    asyncio.run(p._handle_battle_message(race_batch()))
    check("_attach_fusion_parser: typechange до request",
          p.seen, ["switch", "-start", "request", "html"])
    check("_attach_fusion_parser: статы разобраны", p._fusion_stats.get(TAG, {}).get("p2", {}).get("base_stats") is not None, True)

    # 3.3 классы, которыми играют обученной моделью
    from agents.players import PolicyPlayer as PlayersPP
    from agents.policy_player_simple import PolicyPlayer as SimplePP
    from agents import policy_player as trainer_mod

    check("agents.players.PolicyPlayer наследует FusionInfoParser",
          FusionInfoParser in PlayersPP.__mro__, True)
    check("agents.policy_player_simple.PolicyPlayer наследует FusionInfoParser",
          FusionInfoParser in SimplePP.__mro__, True)
    check("agents.policy_player.PolicyPlayer — тот же класс (инференс index.py)",
          trainer_mod.PolicyPlayer is PlayersPP, True)

    # 3.4 правила перестановки
    print("--- 4. правила перестановки ---")
    # (а) уже корректный порядок — не трогаем объект
    ok_batch = [
        [">" + TAG],
        ["", "switch", "p2a: Frost", "froslass, L50, M", "100/100"],
        ["", "-start", "p2a: Frost", "typechange", "Steel/Ice", "[silent]"],
        ["", "request", "{}"],
    ]
    out, n = _hoist_typechanges_before_request(ok_batch)
    check("корректный порядок: 0 переносов", n, 0)
    check("корректный порядок: объект не подменён", out is ok_batch, True)

    # (б) два запроса в кадре: трогаем только хвост за последним запросом
    multi = [
        [">" + TAG],
        ["", "switch", "p2a: A", "froslass, L50, M", "100/100"],
        ["", "request", "{}"],
        ["", "-start", "p2a: A", "typechange", "Steel/Ice", "[silent]"],
        ["", "switch", "p2a: B", "garchomp, L50, M", "100/100"],
        ["", "request", "{}"],
        ["", "-start", "p2a: B", "typechange", "Water/Ghost", "[silent]"],
    ]
    out2, n2 = _hoist_typechanges_before_request(multi)
    seq = [m[1] for m in out2[1:]]
    check("два запроса: переносится только хвостовая строка", n2, 1)
    check("два запроса: порядок (тип B поднят к последнему request, тип A остался на месте)",
          seq, ["switch", "request", "-start", "switch", "-start", "request"])
    check("два запроса: состав кадра не изменился",
          sorted(json.dumps(m) for m in out2[1:]), sorted(json.dumps(m) for m in multi[1:]))


def test_vecnorm_for_inference():
    """Статистики VecNormalize должны применяться к инференсу, даже если они от старой версии."""
    import os

    from agents.config import N_FEATURES, VECNORM_PATH
    from agents.vecnorm_utils import load_vecnorm_state, load_vecnorm_stats, pad_stats

    import numpy as np

    print("--- 5. VecNormalize для инференса ---")
    m, v, changed = pad_stats(np.zeros(418, dtype=np.float64), np.ones(418, dtype=np.float64), N_FEATURES)
    check("pad_stats: добивает до N_FEATURES", (m.shape[0], v.shape[0]), (N_FEATURES, N_FEATURES))
    check("pad_stats: новые признаки mean=0, var=1", (float(m[714]), float(v[714])), (0.0, 1.0))
    check("pad_stats: старые значения сохранены", (float(m[100]), float(v[100])), (0.0, 1.0))
    check("pad_stats: пометка об изменении", changed, True)
    m2, v2, changed2 = pad_stats(np.zeros(N_FEATURES), np.ones(N_FEATURES), N_FEATURES)
    check("pad_stats: одинаковую размерность не трогает", changed2, False)

    if not os.path.isfile(VECNORM_PATH):
        print(f"SKIP {VECNORM_PATH} нет файла — пропускаю проверку на реальном pkl")
        return
    st = load_vecnorm_stats(VECNORM_PATH, N_FEATURES)
    raw = load_vecnorm_state(VECNORM_PATH)
    if st is None or raw is None:
        print(f"SKIP {VECNORM_PATH} не читается")
        return
    print(f"{st.describe()} (в файле {raw['mean'].shape[0]} признаков, norm_obs={raw['norm_obs']})")
    out = st.normalize(np.full(N_FEATURES, 5.0, dtype=np.float32))
    check("normalize: размер верный", out.shape[0], N_FEATURES)
    check("normalize: значения конечные", bool(np.isfinite(out).all()), True)
    check("normalize: клип по clip_obs", bool((np.abs(out) <= st.clip_obs + 1e-6).all()), True)
    # формула совпадает с SB3: clip((x-mean)/sqrt(var+eps), -clip, clip)
    x = np.arange(N_FEATURES, dtype=np.float32)
    manual = np.clip((x - st.mean) / np.sqrt(st.var + st.eps), -st.clip_obs, st.clip_obs)
    check("normalize: формула 1:1 с SB3", bool(np.allclose(out.dtype.type(0) + manual, st.normalize(x))), True)


def test_modules_import():
    """Дым: модули импортируются и ключевые функции на месте (ловит вырезанный код)."""
    print("--- 6. дым-тест импортов ---")
    import importlib

    mods = ["agents.type_utils", "agents.fusion_parser", "agents.players",
            "agents.vecnorm_utils", "agents.features", "agents.config"]
    for name in mods:
        try:
            importlib.import_module(name)
            print(f"  import {name}: OK")
        except Exception as e:
            FAILED.append(f"import {name}")
            print(f"  import {name}: FAIL {type(e).__name__}: {e}")
    from agents import fusion_parser as fp
    for fn in ("_parse_fusion_message", "_parse_protect_message", "_reorder_batch",
               "_hoist_typechanges_before_request", "_attach_fusion_parser"):
        check(f"fusion_parser.{fn} существует", callable(getattr(fp, fn, None)), True)


def main() -> int:
    test_stats_and_types_reach_decision()
    test_parser_regression()
    test_wiring_paths()
    test_vecnorm_for_inference()
    test_modules_import()
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
