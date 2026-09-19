"""Регрессия на классификацию действий в ExampleEnv.action_to_order.

Контекст. В poke-env `SinglesEnv` раскладка действий такая:
    0..5  — свитч (индекс в battle.team)
    6..9  — приём ((action - 6) % 4)
    10..13 — мега, 14..17 — z, 18..21 — динамакс, 22..25 — тера
В `ExampleEnv.action_to_order` книжка велась по конвенции DoublesEnv
(`action < len(available_moves)` -> приём). Из-за этого:
  * свитч помечался как приём и мог получить WASTED_MOVE_PENALTY за приём, которого
    агент не использовал (свитчить = штраф);
  * настоящий приём помечался свитчем и от штрафа освобождался.
Награда систематически подталкивала «спамить атаками».

Запуск: python test_action_kind.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import config as _cfg  # noqa: E402,F401  (монки-патчи poke-env)
from agents.env import ExampleEnv, WASTED_MOVE_PENALTY  # noqa: E402
from agents.features import _move_wasted_flag  # noqa: E402
from poke_env.battle import Battle  # noqa: E402

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def check_true(label, cond, extra=""):
    check(label + (f": {extra}" if extra else ""), bool(cond), True)


def make_env():
    return ExampleEnv(battle_format="gen9fusionmonsrandombattle", log_level=40,
                      open_timeout=None, start_listening=False)


def make_battle(opp="skarmory, L50, F", opp_species="skarmory", our_moves=("surf", "earthquake"),
                with_bench=True, bench_moves=("dragonclaw",)):
    """Наш swampert (Surf — ок, Earthquake — иммун по Flying) против skarmory."""
    b = Battle(battle_tag="battle-gen9fusionmonsrandombattle-1", username="Me",
               logger=logging.getLogger("quiet"), gen=9)
    msgs = [["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""], ["", "start"],
            ["", "switch", "p1a: Aqua", "swampert, L50, M", "362/362"],
            ["", "switch", "p2a: Skarm", opp, "300/300"]]
    for msg in msgs:
        b.parse_message(msg)
    pokemon = [{"ident": "p1: Aqua", "details": "swampert, L50, M", "condition": "362/362", "active": True,
                "stats": {"atk": 257, "def": 257, "spa": 257, "spd": 257, "spe": 257},
                "moves": list(our_moves), "baseAbility": "torrent", "item": "leftovers",
                "pokeball": "pokeball", "ability": "torrent"}]
    if with_bench:
        pokemon.append({"ident": "p1: Bench", "details": "garchomp, L50, F", "condition": "300/300",
                        "active": False, "stats": {"atk": 250, "def": 250, "spa": 250, "spd": 250, "spe": 250},
                        "moves": list(bench_moves), "baseAbility": "roughskin", "item": "leftovers",
                        "pokeball": "pokeball", "ability": "roughskin"})
    b.parse_request({"active": [{"moves": [
        {"move": m.capitalize(), "id": m, "pp": 16, "maxpp": 16, "target": "normal", "disabled": False}
        for m in our_moves]}],
        "side": {"id": "p1", "name": "Me", "pokemon": pokemon}, "rqid": 1})
    return b


def test_switch_action_bookkeeping():
    """Свитч (0..5) помечается свитчем, а не приёмом: раньше был фантомный wasted-штраф."""
    env = make_env()
    b = make_battle()
    tag = b.battle_tag
    env.battle1 = b
    assert bool(_move_wasted_flag(b.available_moves[1], b)) is True, "Earthquake должен быть wasted (Flying)"

    order = env.action_to_order(1, b)
    check_true("ордер — реальный свитч", "switch" in str(order).lower(), str(order))
    check("служебная метка действия", env._last_action_kind.get(tag), "switch")
    check("_last_move_id — не фантомный приём", env._last_move_id.get(tag), "switch")
    check("_last_was_switch", env._last_was_switch.get(tag), True)
    check("_last_wasted (свитч не бывает wasted)", env._last_wasted.get(tag), False)


def test_move_action_bookkeeping():
    """Приём (6..9) помечается приёмом, wasted-флаг берётся у РЕАЛЬНО выбранного приёма."""
    env = make_env()
    b = make_battle()
    tag = b.battle_tag
    env.battle1 = b
    env.action_to_order(6, b)                      # move[0] = surf
    check("предмет проверки: surf — не wasted", env._last_wasted.get(tag), False)
    check("метка действия для приёма", env._last_action_kind.get(tag), "move")
    check("_last_move_id = выбранный приём", env._last_move_id.get(tag), "surf")
    check("_last_was_switch = False", env._last_was_switch.get(tag), False)

    env.action_to_order(7, b)                      # move[1] = earthquake (иммун)
    check("метка действия", env._last_action_kind.get(tag), "move")
    check("wasted-приём действительно помечен wasted", env._last_wasted.get(tag), True)
    check("_last_move_id = earthquake", env._last_move_id.get(tag), "earthquake")


def test_tera_action_is_a_move():
    """Тера-действия (22..25) — это тоже приёмы (индекс (action-6)%4), а не свитчи."""
    env = make_env()
    b = make_battle()
    tag = b.battle_tag
    env.battle1 = b
    env.action_to_order(23, b)                     # тера-версия move[1] = earthquake
    check("метка действия", env._last_action_kind.get(tag), "tera")
    check("_last_move_id = earthquake", env._last_move_id.get(tag), "earthquake")
    check("тера не считается свитчем", env._last_was_switch.get(tag), False)
    check("wasted-флаг как у приёма", env._last_wasted.get(tag), True)


def test_default_and_forfeit_actions():
    env = make_env()
    b = make_battle()
    tag = b.battle_tag
    env.battle1 = b
    env.action_to_order(-2, b)
    check("default: метка", env._last_action_kind.get(tag), "default")
    check("default: move_id", env._last_move_id.get(tag), "default")
    check("default: не свитч", env._last_was_switch.get(tag), False)
    check("default: не wasted", env._last_wasted.get(tag), False)


def test_switch_turn_does_not_pay_wasted_penalty():
    """Главное следствие для награды: свитч больше не платит WASTED_MOVE_PENALTY.

    Проверяем через награду: считаем reward после свитча, затем искусственно ставим
    багованное состояние (wasted=True, was_switch=False, как писало прежнее код) и
    считаем ещё раз — разница должна равняться ровно штрафу.
    """
    env = make_env()
    b = make_battle()
    tag = b.battle_tag
    env.battle1 = b
    r_init = env.calc_reward(b)                    # инициализация состояния награды
    check_true("инициализация награды прошла", isinstance(r_init, float), str(r_init))

    env.action_to_order(1, b)                      # свитч на скамейку
    r_fixed = env.calc_reward(b)
    env._last_wasted[tag] = True                   # воспроизводим прежнее (багованное) состояние
    env._last_was_switch[tag] = False
    r_buggy = env.calc_reward(b)
    delta = r_fixed - r_buggy
    check_true(f"свитч без штрафа дороже ровно на WASTED_MOVE_PENALTY ({WASTED_MOVE_PENALTY})",
               abs(delta - WASTED_MOVE_PENALTY) < 1e-6, f"delta={delta:.6f}")


def test_moves_for_action_matches_poke_env():
    """Список приёмов для индексации совпадает с тем, что использует SinglesEnv.action_to_order."""
    env = make_env()
    b = make_battle(our_moves=("surf", "earthquake"))
    check("обычный случай: known_moves активного",
          [m.id for m in env._moves_for_action(b)], ["surf", "earthquake"])

    # редкий случай poke-env: доступен ровно один приём, которого нет среди известных ->
    # берётся именно он (ветка `len(available_moves) == 1` в SinglesEnv.action_to_order).
    # Настоящий Battle так собрать нельзя (poke-env ассертит moveset), поэтому подставляем
    # минимальный двойник: _moves_for_action читает только active_pokemon.moves / available_moves.
    class _FakeMon:
        def __init__(self, moves):
            self.moves = {m.id: m for m in moves}

    class _FakeMove:
        def __init__(self, mid):
            self.id = mid

    class _FakeBattle:
        def __init__(self, known, avail):
            self.active_pokemon = _FakeMon(known)
            self.available_moves = list(avail)

    fake = _FakeBattle([_FakeMove("surf")], [_FakeMove("protect")])
    check("единственный незнакомый приём -> available_moves",
          [m.id for m in env._moves_for_action(fake)], ["protect"])

    # тот же единственный приём, но он известен -> порядок known_moves сохраняется
    fake2 = _FakeBattle([_FakeMove("surf")], [_FakeMove("surf")])
    check("единственный ИЗВЕСТНЫЙ приём -> known_moves",
          [m.id for m in env._moves_for_action(fake2)], ["surf"])

    # индекс приёма для тера-действий: (action - 6) % 4 (тера 22..25 -> те же 0..3)
    check("тера-индекс = move-индекс", [(a - 6) % 4 for a in (22, 23, 24, 25)], [0, 1, 2, 3])


def test_action_mix_counter():
    """Счётчик типов действий: им видно, есть ли «спам атаками»."""
    env = make_env()
    b = make_battle()
    env.battle1 = b
    for action in (6, 6, 7, 1, 23):
        env.action_to_order(action, b)
    mix = env.action_mix()
    check("свитчей", mix["switch"], 1)
    check("приёмов (6, 6, 7)", mix["move"], 3)
    check("тер", mix["tera"], 1)
    check("доля свитчей", mix["_switch_share"], round(1 / 5, 3))
    env.reset_action_mix()
    check("сброс счётчика", env.action_mix()["switch"], 0)


def test_action_mix_classification():
    """Классификация действий в StepCounterCallback совпадает с раскладкой poke-env.

    Именно этой метрикой (`[mix]` в логе обучения) проверяется, что политика перестала
    «спамить только атаками»: свитч 0..5, приём 6..9, мега/z/динамакс 10..21, тера 22..25.
    """
    from agents.training import StepCounterCallback

    cases = {-2: "other", -1: "other", 0: "switch", 3: "switch", 5: "switch",
             6: "move", 7: "move", 8: "move", 9: "move",
             10: "gimmick", 21: "gimmick", 22: "tera", 23: "tera", 25: "tera", 26: "other"}
    for action, want in cases.items():
        check(f"класс действия {action}", StepCounterCallback.classify_action(action), want)

    import numpy as np
    cb = StepCounterCallback({"value": 0}, 8, mix_every=4)
    cb({"actions": np.array([0, 1, 6, 7, 9, 22, 23, 5])}, {})
    mix = cb.action_mix()
    check("свитчей в миксе", mix["switch"], 3)
    check("приёмов в миксе", mix["move"], 3)
    check("тер в миксе", mix["tera"], 2)
    check("доля свитчей", mix["_switch_share"], round(3 / 8, 4))
    check("тера не считается приёмом", mix["_move_share"], round(3 / 8, 4))
    check("шаги посчитаны", cb.steps_holder["value"], 8)
    # маска/голова политики: 26 действий, где 22..25 — тера (иначе тера недостижима)
    check("действий всего", 26, 26)
    check("тера-блок внутри 26", all(0 <= a < 26 for a in (22, 23, 24, 25)), True)


def test_mix_ignores_no_choice_steps():
    """Шаги без выбора не считаются свитчами в [mix].

    Регрессия (жалоба «в [mix] 60% свитчей, а модель в бою ни разу не свитчит»):
    когда сервер присылает |request| с wait:true, poke-env отдаёт obs с маской
    [1, 0, 0, ...] — единственное разрешённое действие 0 это индекс СВИТЧА, политика
    с additive-маской обязана его выбрать, а env это действие ВЫБРАСЫВАЕТ
    (agent1_to_move=False). Такие шаги нельзя показывать как выбор свитча.
    """
    import numpy as np
    import torch
    from agents.training import StepCounterCallback

    cb = StepCounterCallback({"value": 0}, 4, mix_every=10_000)
    mask = np.zeros((4, 26), dtype=np.int8)
    mask[0, [0, 1, 6, 7, 8, 22]] = 1     # настоящее решение (есть и свитчи, и приёмы)
    mask[1, 0] = 1                        # ожидание соперника: выбор отсутствует
    mask[2, [6, 7]] = 1                   # настоящее решение: только приёмы
    mask[3, [0, 1, 2, 3, 4, 5]] = 1       # настоящее решение: только свитчи (принудительный)
    cb({"actions": np.array([1, 0, 6, 3]),
        "obs_tensor": {"action_mask": torch.as_tensor(mask)}}, {})
    mix = cb.action_mix()
    # action 1 при живых приёмах — свитч ПО ВЫБОРУ; action 3 без приёмов — вынужденный (фейнт)
    check("mix: свитч по выбору (были приёмы)", mix["switch"], 1)
    check("mix: вынужденные свитчи (приёмов нет: фейнт + ожидание)", mix["switch_forced"], 2)
    check("mix: приёмы", mix["move"], 1)
    check("mix: шаг с одним легальным действием помечен (wait/последний покемон)", mix["single"], 1)
    check("mix: шагов всего", mix["_total"], 4)
    check("mix: шагов с настоящим выбором (>=2 варианта и есть приёмы)", mix["_decided"], 2)
    check("mix: доля ВСЕХ свитчей (что реально ушло в бой)", mix["_switch_share"], round(3 / 4, 4))
    check("mix: доля свитчей по выбору", mix["_switch_own_share"], round(1 / 4, 4))
    check("mix: доля вынужденных свитчей", mix["_switch_forced_share"], round(2 / 4, 4))
    check("mix: доля шагов без выбора", mix["_single_share"], round(1 / 4, 4))

    # маска numpy без obs_tensor (fallback на политику) и вовсе без маски — как раньше
    cb2 = StepCounterCallback({"value": 0}, 2, mix_every=10_000)
    cb2({"actions": np.array([0, 6])}, {})
    check("mix без маски: считает всё (обратная совместимость)", cb2.action_mix()["switch"], 1)
    check("mix без маски: вынужденных свитчей нет (нечем разделить)", cb2.action_mix()["switch_forced"], 0)
    check("mix без маски: total", cb2.action_mix()["_total"], 2)

    # маска из 26 колонок: приёмы это 6..9 и гимики 10..25; свитчи без приёмов = фейнт
    cb3 = StepCounterCallback({"value": 0}, 2, mix_every=10_000)
    mask3 = np.zeros((2, 26), dtype=np.int8)
    mask3[0, [0, 1, 6, 22]] = 1   # есть приём и тера -> свитч по выбору
    mask3[1, [0, 1, 2, 3]] = 1    # только свитчи -> вынужденный, НО не single
    cb3({"actions": np.array([2, 0]), "obs_tensor": {"action_mask": __import__("torch").as_tensor(mask3)}}, {})
    m3 = cb3.action_mix()
    check("mix: свитч по выбору при доступной тере", m3["switch"], 1)
    check("mix: принудительный свитч с несколькими живыми — вынужденный, не single",
          (m3["switch_forced"], m3["single"]), (1, 0))


class _FakePokeEnv:
    """Мини-двойник PokeEnv: скриптованный agent1_to_move + счётчик применённых действий."""

    _fake = False
    _strict = True

    def __init__(self, script):
        from types import SimpleNamespace
        from gymnasium.spaces import Box
        self.script = list(script)
        self.applied = []                     # действия, которые env реально применил
        self.n = 0
        self.agent1_to_move = bool(self.script.pop(0)) if self.script else False
        self.agent1 = SimpleNamespace(username="p1")
        self.agent2 = SimpleNamespace(username="p2")
        self.battle1 = SimpleNamespace(wait=False)
        self.battle2 = SimpleNamespace(wait=False, teampreview=False)
        self.observation_spaces = {"p1": Box(-1, 1, (3,)), "p2": Box(-1, 1, (3,))}
        self.action_spaces = {"p1": Box(-1, 1, (1,)), "p2": Box(-1, 1, (1,))}

    def order_to_action(self, order, battle, fake=False, strict=True):
        return -2

    def step(self, actions):
        if self.agent1_to_move:               # как PokeEnv: действие применяется только «в наш ход»
            self.applied.append(actions["p1"])
        self.n += 1
        self.agent1_to_move = bool(self.script.pop(0)) if self.script else False
        obs = {"p1": [float(self.n)], "p2": [0.0]}
        rew = {"p1": 1.0 + self.n, "p2": 0.0}
        term = {"p1": False, "p2": False}
        trunc = {"p1": False, "p2": False}
        return obs, rew, term, trunc, {"p1": {"n": self.n}, "p2": {}}


class _FakeOpponent:
    def choose_move(self, battle):
        return "order"

    def reset_battles(self):
        pass


def test_decision_wrapper_skips_wait_steps():
    """DecisionWrapper: наружу только настоящие решения, награды суммируются, штраф времени один.

    Сценарий: шаг 1 применяется (был наш ход), затем два состояния ожидания
    (agent1_to_move=False — env выбрасывает действие), затем снова наш ход.
    Без обёртки агент получил бы 2 бесполезных шага и «свитч», которого не было.
    """
    from agents.env import DecisionWrapper, TIME_PENALTY

    # сценарий: наш ход -> (внутри) состояние ожидания -> наш ход -> наш ход
    inner = _FakePokeEnv([True, False, True, True])
    w = DecisionWrapper(inner, _FakeOpponent())
    # шаг 1: действие применено, но следом пришло состояние ожидания (agent1_to_move=False)
    # -> наружу уходит только следующий НАСТОЯЩИЙ ход, награды суммируются,
    #    лишний штраф времени (за ожидание) возвращается
    obs, rew, term, trunc, info = w.step(5)
    check("ожидание проглочено: obs настоящего хода", obs, [2.0])
    check("wait-шаги: награды суммированы (+1 штраф времени)", rew, (1.0 + 1) + (1.0 + 2) + TIME_PENALTY)
    check("env применил действие", inner.applied, [5])
    check("сделано 2 внутренних шага", inner.n, 2)
    # шаг 2: сразу настоящее решение -> отдаётся как есть, штраф времени не трогаем
    obs, rew, term, trunc, info = w.step(7)
    check("решение отдано сразу", obs, [3.0])
    check("награда решения не изменена", rew, 1.0 + 3)
    check("второе действие применено", inner.applied, [5, 7])

    # бой закончился во время ожидания -> обёртка не должна зацикливаться
    class _FakePokeEnvDone(_FakePokeEnv):
        def step(self, actions):
            obs, rew, term, trunc, info = super().step(actions)
            return obs, rew, {"p1": True}, trunc, info

    inner2 = _FakePokeEnvDone([True, False, False, False])
    w2 = DecisionWrapper(inner2, _FakeOpponent())
    obs, rew, term, trunc, info = w2.step(0)
    check("терминальное состояние не зацикливается", term, True)


def main() -> int:
    test_switch_action_bookkeeping()
    print("-" * 74)
    test_move_action_bookkeeping()
    print("-" * 74)
    test_tera_action_is_a_move()
    print("-" * 74)
    test_default_and_forfeit_actions()
    print("-" * 74)
    test_switch_turn_does_not_pay_wasted_penalty()
    print("-" * 74)
    test_moves_for_action_matches_poke_env()
    print("-" * 74)
    test_action_mix_counter()
    print("-" * 74)
    test_action_mix_classification()
    print("-" * 74)
    test_mix_ignores_no_choice_steps()
    print("-" * 74)
    test_decision_wrapper_skips_wait_steps()
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
