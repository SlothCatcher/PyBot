"""Тесты index.py: 4 параллельных боя на бота + авто-перезапуск с записью ошибок в лог.

Сервер не нужен: супервизор работает с duck-typed объектом игрока, поэтому здесь стоят
поддельные игроки, которые падают / теряют websocket так же, как настоящие.

Запуск: python test_index_supervisor.py
"""
import asyncio
import logging
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import index  # noqa: E402  (после правки sys.path)

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def check_true(label, cond, extra=""):
    check(label + (f": {extra}" if extra else ""), bool(cond), True)


# --------------------------------------------------------------- фейковый игрок ---
class FakeWebsocket:
    def __init__(self):
        self.closed = False


class FakeClient:
    def __init__(self):
        self.websocket = FakeWebsocket()
        self.logged_in = asyncio.Event()
        self.logged_in.set()
        self.stop_calls = 0
        self.messages = []

    async def wait_for_login(self):
        return None

    async def send_message(self, message, room=None):
        self.messages.append((message, room))

    async def stop_listening(self):
        self.stop_calls += 1
        self.websocket.closed = True


class FakePlayer:
    """Поведение задаётся сценарием: список действий по номеру запуска."""

    def __init__(self, behaviour, log=None):
        self.ps_client = FakeClient()
        self._battles = {}
        self.n_finished_battles = 7
        self.n_won_battles = 4
        self.n_lost_battles = 3
        self.behaviour = behaviour
        self.log = log

    async def accept_challenges(self, opponent, n_challenges, packed_team=None):
        if self.log is not None:
            self.log.append("accept")
        action = self.behaviour
        if action == "raise":
            raise ValueError("сервер прислал мусор вместо запроса")
        if action == "websocket_closed":
            self.ps_client.websocket.closed = True      # имитируем обрыв сети
            await asyncio.sleep(30)
        if action == "never":
            await asyncio.sleep(30)
        if action == "return":
            return


def make_spec(name="test"):
    return index.BotSpec(name=name, account="acc", avatar="a", battle_format="gen9randombattle",
                         greeting="hi")


async def run_supervisor(behaviours, tmpdir, **kwargs):
    """Запускает supervise с заданной последовательностью поведений, возвращает счётчик запусков."""
    calls = {"n": 0}
    players = []

    def factory():
        behaviour = behaviours[min(calls["n"], len(behaviours) - 1)]
        calls["n"] += 1
        player = FakePlayer(behaviour)
        players.append(player)
        return player

    serve_kwargs = dict(login_timeout=1.0, check_interval=0.01, grace=0.0, announce=False,
                        battles=4, sleep=asyncio.sleep)
    serve_kwargs.update(kwargs)
    task = asyncio.create_task(
        index.supervise(make_spec(), factory, restart_delay=0.01, max_restart_delay=0.02,
                        healthy_run_seconds=1.0, **serve_kwargs)
    )
    await asyncio.sleep(0.25)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return calls["n"], players


import contextlib  # noqa: E402  (используется в run_supervisor)


def test_restart_on_exception(tmpdir):
    """Ошибка в бою -> traceback в лог -> бот поднят заново."""
    log_file = os.path.join(tmpdir, "restart_exception.log")
    index.setup_logging(log_file, level=logging.DEBUG, console=False)
    n, players = asyncio.run(run_supervisor(["raise", "never"], tmpdir))
    text = open(log_file, encoding="utf-8").read()
    check_true("бот перезапущен после исключения", n >= 2, f"запусков {n}")
    check_true("traceback записан в лог", "Traceback (most recent call last)" in text)
    check_true("текст ошибки записан в лог", "сервер прислал мусор вместо запроса" in text)
    check_true("в логе есть пометка о перезапуске", "Перезапуск через" in text)
    check_true("в логе есть статистика сессии", "за сессию боёв 7 (побед 4, поражений 3" in text)
    check_true("старое соединение закрыто перед перезапуском",
               all(p.ps_client.stop_calls >= 1 for p in players), f"{[p.ps_client.stop_calls for p in players]}")
    check_true("состояние боёв очищено", all(p._battles == {} for p in players))
    check_true("в логе есть ошибки уровня ERROR", "ERROR" in text)


def test_restart_on_lost_websocket(tmpdir):
    """Тихий обрыв (websocket.closed) ловится watchdog-ом, а не висит вечно."""
    log_file = os.path.join(tmpdir, "restart_websocket.log")
    index.setup_logging(log_file, level=logging.INFO, console=False)
    n, _ = asyncio.run(run_supervisor(["websocket_closed", "never"], tmpdir))
    text = open(log_file, encoding="utf-8").read()
    check_true("бот перезапущен после обрыва связи", n >= 2, f"запусков {n}")
    check_true("причина обрыва записана в лог", "websocket закрыт" in text)
    check_true("обрыв залогирован как ошибка", "BotRestartRequired" in text)


def test_no_restart_loop_on_clean_run(tmpdir):
    """Штатная работа: один запуск, никаких перезапусков и ошибок в логе."""
    log_file = os.path.join(tmpdir, "clean_run.log")
    index.setup_logging(log_file, level=logging.INFO, console=False)
    n, _ = asyncio.run(run_supervisor(["never"], tmpdir))
    text = open(log_file, encoding="utf-8").read()
    check("перезапусков не было", n, 1)
    check_true("в логе нет ERROR", "ERROR" not in text)
    check_true("логин и приём вызовов залогированы", "приём вызовов (до 4 боёв параллельно)" in text)


def test_logging_is_idempotent(tmpdir):
    """Повторный setup_logging не плодит handlers и пишет в новый файл."""
    first = os.path.join(tmpdir, "a.log")
    second = os.path.join(tmpdir, "b.log")
    index.setup_logging(first, console=False)
    n_handlers_1 = len(logging.getLogger().handlers)
    index.setup_logging(second, console=False)
    n_handlers_2 = len(logging.getLogger().handlers)
    check("handlers не дублируются", n_handlers_2, n_handlers_1)
    logging.getLogger("pybot").info("проверка ротации")
    check_true("сообщение попало в новый файл", "проверка ротации" in open(second, encoding="utf-8").read())
    check_true("старый файл закрыт (в него больше не пишем)",
               "проверка ротации" not in open(first, encoding="utf-8").read())


def test_battles_per_bot_on_real_player():
    """Настоящий PolicyPlayer из make_player получает max_concurrent_battles=4."""
    from poke_env import ServerConfiguration
    creds = ("login", "login2", "password", ServerConfiguration("ws://localhost:8000/showdown/websocket",
                                                                "http://localhost:8000"))
    spec = index.build_specs(creds)["fusion"]
    check("BATTLES_PER_BOT = 4", index.BATTLES_PER_BOT, 4)
    # start_listening=False: объект создаётся без подключения к серверу
    player = index.make_player(None, spec, creds, start_listening=False)
    check("реальный PolicyPlayer: параллельных боёв", player._max_concurrent_battles, 4)
    check("аккаунт взят из spec", spec.account, "login2")
    check("формат у fusion-бота", player._format, index.BATTLE_FORMAT_FUSION)
    player2 = index.make_player(None, index.build_specs(creds)["random"], creds,
                                battles=2, start_listening=False)
    check("--battles переопределяет число боёв", player2._max_concurrent_battles, 2)


def test_load_policy_missing_file(tmpdir):
    """Отсутствующий чекпоинт — понятная ошибка на старте, а не бесконечный перезапуск."""
    try:
        index.load_policy(os.path.join(tmpdir, "нет-такого.zip"))
        raised = None
    except RuntimeError as exc:
        raised = str(exc)
    check_true("RuntimeError с понятным текстом", raised is not None and "не удалось загрузить" in raised,
               str(raised))


def make_old_checkpoint(path, old_dim):
    """Делает «старый» чекпоинт: первый слой обрезан до old_dim, obs_space тоже.

    Тонкость: `FeaturesExtractor` строит первый слой от МОДУЛЬНОЙ константы N_FEATURES, а не
    от observation_space, поэтому `_probe_ppo(old_dim)` сам по себе даёт веса не old_dim,
    а N_FEATURES — обрезать нужно руками, иначе тест «миграции» ничего не проверяет.
    """
    import torch
    from gymnasium.spaces import Box, Dict
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
    from agents.policy_player import _probe_ppo

    ppo, env = _probe_ppo(old_dim)
    path_tmp = path + ".tmp"
    ppo.save(path_tmp)
    env.close()

    data, params, pytorch_variables = load_from_zip_file(path_tmp, device=torch.device("cpu"))
    for state in params.values():
        if not isinstance(state, dict):
            continue
        for key, tensor in list(state.items()):
            if key.endswith(".net.0.weight") and getattr(tensor, "dim", lambda: 0)() == 2 \
                    and tensor.shape[1] > old_dim:
                state[key] = tensor[:, :old_dim].clone()
    data["observation_space"] = Dict({
        "observation": Box(-1.0, 4.0, shape=(old_dim,), dtype="float32"),
        "action_mask": Box(0, 1, shape=(9,), dtype=bool),
    })
    save_to_zip_file(path, data=data, params=params, pytorch_variables=pytorch_variables)
    os.remove(path_tmp)
    return path


def test_load_policy_migrates_old_dim(tmpdir):
    """Старый чекпоинт (802 признака) грузится с миграцией под N_FEATURES=870."""
    import torch
    from agents.checkpoint_utils import checkpoint_obs_dim, load_policy_compat
    from agents.config import N_FEATURES

    old_dim = N_FEATURES - 68          # как было до зеркала/флагов
    path = make_old_checkpoint(os.path.join(tmpdir, "old.zip"), old_dim)
    check("чекпоинт действительно старой размерности", checkpoint_obs_dim(path), old_dim)

    ppo, info = load_policy_compat(path, N_FEATURES, cache_dir=tmpdir)
    check("миграция зафиксирована в info", info.get("migrated_from"), old_dim)
    policy = index.load_policy(path, cache_dir=tmpdir)
    w = policy.features_extractor.net[0].weight.detach()
    check("вес первого слоя расширен до N_FEATURES", int(w.shape[1]), N_FEATURES)
    check_true("новые столбцы нулевые (warm start)",
               float(w[:, old_dim:].abs().sum()) == 0.0,
               f"сумма {float(w[:, old_dim:].abs().sum())}")
    check_true("старые веса на месте", float(w[:, :old_dim].abs().sum()) > 0.0)
    import torch.nn as nn
    check_true("в экстракторе есть Dropout (проверка режима не вырождена)",
               any(isinstance(m, nn.Dropout) and m.p > 0 for m in policy.modules()))
    check_true("политика переведена в режим инференса (Dropout выключен)",
               policy.training is False)
    check_true("кэш миграции создан рядом",
               any(name.startswith("old__obs") for name in os.listdir(tmpdir)),
               str(os.listdir(tmpdir)))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_restart_on_exception(tmpdir)
        print("-" * 74)
        test_restart_on_lost_websocket(tmpdir)
        print("-" * 74)
        test_no_restart_loop_on_clean_run(tmpdir)
        print("-" * 74)
        test_logging_is_idempotent(tmpdir)
        print("-" * 74)
        test_battles_per_bot_on_real_player()
        print("-" * 74)
        test_load_policy_missing_file(tmpdir)
        print("-" * 74)
        test_load_policy_migrates_old_dim(tmpdir)
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
