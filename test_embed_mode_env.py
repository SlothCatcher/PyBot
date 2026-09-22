#!/usr/bin/env python3
"""Режим действий обязан доезжать до env-воркеров (иначе embed-прогон = шум).

История бага: в ветке без `--resume` `SubprocVecEnv` создавался РАНЬШЕ, чем в главном
процессе выставлялся режим `embed`, а сам режим не экспортировался в окружение. Воркеры
стартовали в `indices`: маска в obs приезжала в нумерации `battle.team`, свитч-действие `j`
означало другого монстра. Политика при этом выбирала по каноническому порядку резервов.
Итог — винрейт уровня случайной игры (53% против random, 1.7% против эвристики) при
внешне исправных логах обучения.

Что проверяем:

  A) `set_action_mode` экспортирует режим в PYBOT_ACTION_MODE, `reset_action_mode` убирает
     переменную; без явной установки режим = indices (историческая совместимость);
  B) дочерний процесс (обычный subprocess + multiprocessing spawn) видит тот же режим —
     это тот канал, по которому режим доезжает до env-воркеров;
  C) `ExampleEnv.create_env(action_mode=...)` применяет режим В ВОРКЕРЕ первой строкой и
     не теряет `opponent_weights`;
  D) в исходнике policy_player режим выставляется ДО первого SubprocVecEnv, и каждая
     фабрика env передаёт action_mode (структурный тест — ловит возврат бага);
  E) семантика на синтетическом бою: env-side `action_to_order` в embed ведёт на
     j-й резерв канонического порядка (и пишет «switch» в бухгалтерию награды), а в
     indices — на j-й индекс в порядке team; маска соответствуют своей нумерации;
  F) живой бой (если поднят Showdown): решение «свитч j» реально приводит на канонический
     резерв j — от маски до ордера сервера.

Запуск: PYTHONPATH=. python test_embed_mode_env.py
"""
import asyncio
import logging
import multiprocessing as mp
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import action_space as A  # noqa: E402
from agents import config as _cfg  # noqa: E402,F401  (монки-патчи poke-env)
from agents import env as E  # noqa: E402
from poke_env.environment.singles_env import SinglesEnv  # noqa: E402

OK = 0
FAIL: list = []

ROOT = pathlib.Path(__file__).resolve().parent


def check(name, cond, extra=""):
    global OK
    if cond:
        OK += 1
        print(f"OK   {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL.append(name)
        print(f"FAIL {name}" + (f": {extra}" if extra else ""))


def child_mode_probe(q):
    """Top-level, чтобы работал spawn (модуль переимпортируется в ребёнке)."""
    from agents.action_space import get_action_mode
    q.put(get_action_mode())


def env_factory_probe(q, action_mode):
    """Что видит воркер: применяем режим ровно так, как это делает create_env."""
    from agents.env import _apply_worker_action_mode      # первая строка create_env
    _apply_worker_action_mode(action_mode)
    from test_slot_free_mode import make_battle           # импорт внутри ребёнка
    from agents.action_space import get_action_mode
    from poke_env.environment.singles_env import SinglesEnv as SE
    battle = make_battle()
    mask = [int(x) for x in SE.get_action_mask(battle)]
    q.put((get_action_mode(), mask[:6]))


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")

    # ---------------------------------------------------- A) экспорт режима ---
    print("=" * 78)
    print("A. Режим экспортируется в окружение (канал до воркеров)")
    print("=" * 78)
    A.reset_action_mode()
    check("A: без установки режим = indices (старое поведение)",
          A.get_action_mode() == "indices", A.get_action_mode())
    check("A: в окружении режима нет", "PYBOT_ACTION_MODE" not in os.environ)
    A.set_action_mode("embed")
    check("A: set_action_mode('embed') пишет PYBOT_ACTION_MODE",
          os.environ.get("PYBOT_ACTION_MODE") == "embed", os.environ.get("PYBOT_ACTION_MODE", "<нет>"))
    A.reset_action_mode()
    check("A: reset_action_mode убирает переменную",
          "PYBOT_ACTION_MODE" not in os.environ and A.get_action_mode() == "indices")

    # ------------------------------------ B) наследование дочерним процессом ---
    print("-" * 78)
    print("B. Дочерний процесс наследует режим (subprocess и spawn)")
    print("-" * 78)
    A.set_action_mode("embed")
    code = ("import sys; sys.path.insert(0, r'%s');"
            "from agents.action_space import get_action_mode; print(get_action_mode())" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=os.environ.copy(), timeout=300)
    child = (out.stdout or "").strip().splitlines()[-1] if out.stdout.strip() else f"<пусто rc={out.returncode}>"
    check("B: обычный дочерний процесс видит embed", child == "embed", child)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child_mode_probe, args=(q,))
    p.start()
    p.join(180)
    spawn_mode = q.get() if not q.empty() else "<нет ответа>"
    check("B: spawn-процесс (как env-воркер SubprocVecEnv) видит embed",
          spawn_mode == "embed", str(spawn_mode))

    # живой воркер: SubprocVecEnv + фабрика с action_mode
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from functools import partial
    from test_slot_free_mode import _ProbeEnv  # noqa: E402

    def _factory(q_, action_mode):
        return _ProbeEnv()

    A.reset_action_mode()          # главный процесс «забыл» режим — воркер обязан знать свой
    q2 = ctx.Queue()
    p2 = ctx.Process(target=env_factory_probe, args=(q2, "embed"))
    p2.start()
    p2.join(180)
    worker_mode, worker_mask = q2.get() if not q2.empty() else ("<нет ответа>", None)
    check("B: воркер с action_mode='embed' играет в embed", worker_mode == "embed", str(worker_mode))
    check("B: маска свитчей в воркере — в embed-нумерации (канонический порядок)",
          worker_mask == [1, 1, 1, 0, 0, 0], str(worker_mask))

    # ----------------------------------- C) create_env применяет режим в воркере ---
    print("-" * 78)
    print("C. ExampleEnv.create_env(action_mode=...) применяет режим в воркере")
    print("-" * 78)
    A.reset_action_mode()
    calls = []
    orig_build = E.ExampleEnv._build_env
    try:
        E.ExampleEnv._build_env = classmethod(
            lambda cls, opponent_weights=None: (calls.append(opponent_weights), "STUB")[1])
        out_env = E.ExampleEnv.create_env(action_mode="embed", opponent_weights={"self_play": 0.5})
    finally:
        E.ExampleEnv._build_env = orig_build
    check("C: create_env(action_mode='embed') включает embed в процессе-воркере",
          A.get_action_mode() == "embed" and os.environ.get("PYBOT_ACTION_MODE") == "embed",
          A.get_action_mode())
    check("C: opponent_weights доезжают до сборки env без изменений",
          out_env == "STUB" and calls == [{"self_play": 0.5}], str(calls))
    A.reset_action_mode()
    orig_build2 = E.ExampleEnv._build_env
    try:
        E.ExampleEnv._build_env = classmethod(lambda cls, opponent_weights=None: "STUB")
        E.ExampleEnv.create_env()
    finally:
        E.ExampleEnv._build_env = orig_build2
    check("C: без action_mode режим не навязывается (None = как в процессе)",
          A.get_action_mode() == "indices", A.get_action_mode())

    # -------------------------------- D) структурный тест на policy_player ---
    print("-" * 78)
    print("D. В policy_player режим выставляется ДО первого env, все фабрики его передают")
    print("-" * 78)
    src = (ROOT / "agents" / "policy_player.py").read_text(encoding="utf-8")
    first_env = src.find("SubprocVecEnv(")
    first_mode = src.find("_set_action_mode(action_mode)")
    check("D: режим выставляется до создания первого SubprocVecEnv",
          0 <= first_mode < first_env, f"mode@{first_mode} env@{first_env}")
    bare = [m.start() for m in re.finditer(r"ExampleEnv\.create_env", src)
            if "action_mode=_mode" not in src[m.start():m.start() + 160]]
    check("D: каждая фабрика env передаёт action_mode=_mode",
          not bare, f"без режима: {len(bare)}")
    bare_sm = src.count("SubprocVecEnv(") - src.count("start_method=_vec_start_method()")
    check("D: все фабрики SubprocVecEnv задают start_method (spawn: forkserver виснет)",
          bare_sm == 0 and "def _vec_start_method" in src, f"без start_method: {bare_sm}")
    check("D: в свежей ветке нет копии блока установки режима",
          src.count("_mode = set_action_mode") + src.count("_mode = _set_action_mode") == 2,
          str(src.count("_mode = set_action_mode") + src.count("_mode = _set_action_mode")))

    # --------------------------------- E) семантика на синтетическом бою ---
    print("-" * 78)
    print("E. env-side action_to_order: embed ведёт на канонический резерв j")
    print("-" * 78)
    from test_slot_free_mode import make_battle  # noqa: E402

    def make_env_shell():
        env = E.ExampleEnv.__new__(E.ExampleEnv)      # без конструктора: сервер не нужен
        env._last_move_id = {}
        env._last_was_switch = {}
        env._last_wasted = {}
        env._last_action_kind = {}
        env._action_counts = {"switch": 0, "move": 0, "tera": 0, "unknown": 0}
        return env

    battle = make_battle()
    reserves = A.embed_reserves(battle)
    team = list(battle.team.values())
    check("E: канонический порядок резервов отличается от порядка team (иначе тест пустой)",
          [m.species for m in reserves] != [m.species for m in team[1:]],
          f"canon={[m.species for m in reserves]} team={[m.species for m in team]}")

    A.set_action_mode("embed")
    env_embed = make_env_shell()
    for j in range(len(reserves)):
        order = env_embed.action_to_order(np.int64(j), battle)
        mon = getattr(order, "order", None)
        if getattr(mon, "species", None) != reserves[j].species:
            check(f"E: embed action {j} -> резерв {reserves[j].species}", False,
                  f"получили {getattr(mon, 'species', mon)}")
            break
    else:
        check("E: embed action j -> ровно j-й резерв канонического порядка", True,
              f"{[m.species for m in reserves]}")
    check("E: бухгалтерия награды видит свитч (а не приём)",
          env_embed._last_action_kind.get(battle.battle_tag) == "switch"
          and env_embed._last_move_id.get(battle.battle_tag) == "switch",
          f"{env_embed._last_action_kind.get(battle.battle_tag)}")
    mask_embed = [int(x) for x in SinglesEnv.get_action_mask(battle)][:6]
    check("E: маска свитчей в embed разрешает ровно существующие резервы",
          mask_embed == [1] * len(reserves) + [0] * (6 - len(reserves)), str(mask_embed))

    A.set_action_mode("indices")
    env_idx = make_env_shell()
    # в indices действие = индекс в battle.team; team[0] — активный монстр, свитч в него
    # запрещён, поэтому берём индекс 1 (первый реальный резерв в порядке team)
    order_idx = env_idx.action_to_order(np.int64(1), battle)
    mon_idx = getattr(order_idx, "order", None)
    check("E: indices action 1 -> монстр с team-индексом 1 (другая семантика)",
          getattr(mon_idx, "species", None) == team[1].species,
          f"{getattr(mon_idx, 'species', mon_idx)} vs team[1]={team[1].species}")
    idx_mask = [int(x) for x in SinglesEnv.get_action_mask(battle)]
    check("E: маска в indices перечисляет team-индексы (активный запрещён)",
          idx_mask[:6] == [0] + [1] * len(reserves) + [0] * (5 - len(reserves)), str(idx_mask[:6]))
    A.set_action_mode("embed")

    # ------------------------------------------- F) живой бой (если есть сервер) ---
    print("-" * 78)
    print("F. Живой бой: свитч j реально приводит на канонический резерв j")
    print("-" * 78)
    live = os.environ.get("PYBOT_SKIP_LIVE", "0") != "1"
    if live:
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/", timeout=3).read(10)
        except Exception as e:
            live = False
            print(f"     Showdown недоступен ({type(e).__name__}) — живая секция пропущена")
    if live:
        _run_live_probe()

    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


def _run_live_probe():
    """Свитчи через embed-нумерацию в реальном бою: маска -> действие -> ордер сервера."""
    from poke_env.player import Player, RandomPlayer
    from agents.features import canonical_reserves

    class Probe(Player):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.pending = None
            self.mismatch = []
            self.switches = 0

        def choose_move(self, battle):
            if self.pending is not None:
                j, expected = self.pending
                self.pending = None
                active = battle.active_pokemon
                if expected is not None and active is not None and not active.fainted:
                    if getattr(active, "species", None) != expected:
                        self.mismatch.append((j, expected, getattr(active, "species", None)))
            mask = [int(x) for x in SinglesEnv.get_action_mask(battle)]
            switches = [i for i in range(6) if mask[i]]
            if switches and len(battle.team) > 1:
                j = switches[0]
                reserves = canonical_reserves(battle.team)
                if j < len(reserves):
                    self.pending = (j, reserves[j].species)
                    self.switches += 1
                    return SinglesEnv.action_to_order(np.int64(j), battle, fake=False, strict=False)
            return self.choose_random_move(battle)

    async def run():
        A.set_action_mode("embed")
        me = Probe(battle_format="gen9randombattle", max_concurrent_battles=1)
        opp = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=1)
        await me.battle_against(opp, n_battles=3)
        return me

    me = asyncio.run(run())
    check("F: бои сыграны, свитчи через embed-нумерацию случались",
          me.n_finished_battles >= 1 and me.switches > 0,
          f"боёв {me.n_finished_battles}, свитчей {me.switches}")
    check("F: каждый свитч привёл на ожидаемый канонический резерв",
          not me.mismatch, f"расхождений {len(me.mismatch)}: {me.mismatch[:3]}")


if __name__ == "__main__":
    sys.exit(main())
