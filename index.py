"""Запуск ботов-ассистентов: 4 параллельных боя на бота + авто-перезапуск при сбоях.

Что было в прежней версии:
  * два бота-AI (Fusionmons Random Battle на аккаунте LOGIN_2 и Random Battle на LOGIN),
    каждый принимал вызовы по одному разу за итерацию (`accept_challenges(None, 1)`);
  * обрыв связи или исключение в бою ловились только в `aiPlay` (печать в stdout и очистка
    `_battles`), а в `aiPlay2` — вообще не ловились: бот умирал молча, и его нужно было
    поднимать руками.

Что стало:
  1. `BATTLES_PER_BOT = 4` — до 4 параллельных боёв на бота (`max_concurrent_battles`),
     вызовы принимаются непрерывно, а не по одному: пока 4 боя идут, следующий вызов ждёт
     свободный слот в очереди poke-env и принимается сразу после окончания боя.
  2. `supervise()` — обёртка на каждого бота: любые ошибки, обрывы websocket и потеря логина
     ловятся, пишутся с traceback в лог-файл, после чего бот поднимается заново с нуля
     (новый логин и новое соединение). Задержка между попытками растёт экспоненциально
     (5 -> 10 -> ... -> 300 c), а после успешного «долгого» прогона сбрасывается.
  3. Логи: `logs/index.log` (ротация 5 МБ x 5 файлов) + консоль. Через файл идут и сообщения
     самого poke-env (его логгеры пишут в корневой логгер), так что причины обрывов видны.

Запуск:
    python index.py                         # оба бота
    python index.py --bot fusion            # только Fusionmons
    python index.py --battles 2 --log-file logs/debug.log
    python index.py --no-announce           # не здороваться в лобби при каждом перезапуске
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------- настройки ---
BATTLES_PER_BOT = 4            # параллельных боёв на одного бота (было 1)
ACCEPT_BATCH = 1_000_000       # сколько вызовов принимает один заход (фактически «бесконечно»)

DEFAULT_MODEL = os.path.join("models", "pretrained_10000.zip")
DEFAULT_LOG_FILE = os.path.join("logs", "index.log")
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 5

RESTART_DELAY = 5.0            # первая пауза перед перезапуском, c
RESTART_MAX_DELAY = 300.0      # предел экспоненциального backoff, c
HEALTHY_RUN_SECONDS = 120.0    # столько проработал -> считаем перезапуск успешным, backoff в 0
LOGIN_TIMEOUT = 60.0           # сколько ждём логин, c
CONNECTION_CHECK_INTERVAL = 15.0   # период проверки живости соединения, c
CONNECTION_GRACE = 30.0        # не считаем «нет websocket» ошибкой первые N секунд после логина

BATTLE_FORMAT_FUSION = "gen9fusionmonsrandombattle"
BATTLE_FORMAT_RANDOM = "gen9randombattle"

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
logger = logging.getLogger("pybot")


# ------------------------------------------------------------------- логи ---
def setup_logging(log_file: str = DEFAULT_LOG_FILE, level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """Логи в файл с ротацией (+ консоль). Идемпотентно: повторный вызов переопределяет файл.

    Файловый handler вешается на КОРНЕВОЙ логгер, поэтому в лог попадает и то, что пишет
    poke-env своими логгерами (в т.ч. «Websocket connection ... closed»), а не только наши
    сообщения — иначе причину обрыва пришлось бы искать в консоли, которой может не быть.
    """
    directory = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(directory, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_pybot_handler", False):   # не плодим дубли при повторном вызове
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = RotatingFileHandler(log_file, maxBytes=LOG_MAX_BYTES,
                                       backupCount=LOG_BACKUPS, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._pybot_handler = True
    root.addHandler(file_handler)

    if console:
        stream = sys.stdout
        with contextlib.suppress(Exception):   # консоль Windows (cp1251) не должна ронять бота
            stream.reconfigure(encoding="utf-8", errors="replace")
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler._pybot_handler = True
        root.addHandler(console_handler)
    return logger


# ----------------------------------------------------------- загрузка модели ---
def load_policy(model_path: str, cache_dir: str | None = None):
    """Политика из чекпоинта с авто-миграцией под текущий N_FEATURES.

    Раньше index.py сам читал `policy.pth` из zip и накатывал на свежую политику с размерами
    из метаданных файла. Если снапшот обучен на другой размерности obs (715/802 против
    текущих N_FEATURES), такой бот падал при первом же ходе с size mismatch. Здесь
    используется штатная миграция репозитория (`load_policy_compat`): паддинг нулями +
    кэш в `models/_migrated/`.
    """
    from agents.checkpoint_utils import load_policy_compat
    from agents.config import N_FEATURES

    ppo, info = load_policy_compat(model_path, N_FEATURES, cache_dir=cache_dir)
    if ppo is None:
        raise RuntimeError(f"не удалось загрузить модель {model_path}: {info.get('error')}")
    if info.get("migrated_from"):
        logger.info("модель %s: obs %s -> %s (мигрирована, кэш: %s)",
                    model_path, info["migrated_from"], N_FEATURES, info.get("cached"))
    else:
        logger.info("модель %s: obs %s, признаки совпадают", model_path, N_FEATURES)

    # Режим действий — свойство модели: embed-политика ждёт свитч-действие j = j-й резерв
    # канонического порядка. Если не применить его на инференсе, маска/ордера разъедутся с
    # тем, что видела политика при обучении.
    try:
        from agents.action_space import describe as _describe_mode, set_action_mode
        from agents.checkpoint_utils import action_mode_from_checkpoint
        mode = action_mode_from_checkpoint(model_path)
        if mode:
            set_action_mode(mode)
            logger.info("режим действий из чекпоинта: %s", _describe_mode())
    except Exception as e:  # noqa: BLE001 — инференс не должен падать из-за диагностики
        logger.warning("не удалось определить режим действий чекпоинта: %s", e)

    policy = ppo.policy
    # ВАЖНО: после PPO.load политика в train-режиме, то есть Dropout(0.1) в экстракторе
    # признаков работает и на инференсе: решения становятся шумными, а поведение бота
    # отличается от прежнего index.py (там был явный policy_instance.eval()).
    policy.set_training_mode(False)
    policy.to("cpu")
    logger.info("политика в режиме инференса (training=%s)", policy.training)
    return policy


# ------------------------------------------------------------------ боты ---
@dataclass
class BotSpec:
    """Описание одного бота: имя для логов, аккаунт, формат и приветствие в лобби."""
    name: str
    account: str
    avatar: str
    battle_format: str
    greeting: str = ""


def load_obs_normalizer(log=print):
    """Статистика нормализации obs из models/vecnormalize.pkl — ровно как во время обучения.

    Обучение идёт с `VecNormalize(norm_obs=True)`, то есть политика видит нормализованные
    признаки. Живой бот раньше получал СЫРЫЕ признаки: сеть работала на сдвинутом по
    масштабу obs, и решения отличались от обучения (часть колонок уходила в клип ±10).
    Возвращает объект с `normalize(obs)` или None (тогда играем на сырых признаках —
    так же, как если модель обучали с `--no-normalize-bc`).
    """
    try:
        from agents.config import N_FEATURES, VECNORM_PATH
        from agents.vecnorm_utils import load_vecnorm_stats
        import os as _os
        if not _os.path.isfile(VECNORM_PATH):
            log(f"нормализация obs: {VECNORM_PATH} нет — играю на сырых признаках")
            return None
        stats = load_vecnorm_stats(VECNORM_PATH, N_FEATURES)
        if stats is None:
            log(f"нормализация obs: {VECNORM_PATH} не читается или norm_obs=False — играю на сырых признаках")
            return None
        log(f"нормализация obs включена: {stats.describe()}")
        warn = stats.stale_warning()
        if warn:
            log(warn)
        return stats
    except Exception as e:
        log(f"нормализация obs недоступна ({e}) — играю на сырых признаках")
        return None


def make_player(policy, spec: BotSpec, credentials, *, battles: int = BATTLES_PER_BOT, **extra):
    """Создаёт PolicyPlayer с нужным числом параллельных боёв (`extra` — для тестов/тонких настроек)."""
    from poke_env import AccountConfiguration
    from agents.players import PolicyPlayer

    _, _, password, server = credentials
    return PolicyPlayer(
        policy=policy,
        avatar=spec.avatar,
        account_configuration=AccountConfiguration(spec.account, password),
        server_configuration=server,
        battle_format=spec.battle_format,
        start_timer_on_battle_start=True,
        max_concurrent_battles=battles,          # <- п.1: 4 параллельных боя
        **extra,
    )


async def announce_in_lobby(player, spec: BotSpec) -> None:
    """Здороваемся в лобби. Сбой приветствия не должен перезапускать бота."""
    await player.ps_client.send_message("/join lobby")
    if spec.greeting:
        await player.ps_client.send_message(spec.greeting, room="lobby")


def _player_stats(player) -> str:
    """Хвост для лога: сколько боёв сыграно и винрейт (если poke-env это знает)."""
    try:
        finished = getattr(player, "n_finished_battles", None)
        if not finished:
            return ""
        won = getattr(player, "n_won_battles", 0) or 0
        lost = getattr(player, "n_lost_battles", 0) or 0
        return f", за сессию боёв {finished} (побед {won}, поражений {lost}, {100 * won / finished:.0f}%)"
    except Exception:
        return ""


class BotRestartRequired(RuntimeError):
    """Соединение умерло (watchdog) — супервизор должен поднять бота заново."""


async def _connection_watchdog(player, *, interval: float, grace: float) -> str:
    """Ждёт, пока соединение сломается, и возвращает причину.

    Ловит именно «тихую» смерть: websocket закрыт со стороны сервера/сети (aiohttp ставит
    `closed=True`, в т.ч. по ping_timeout), логин потерян. Без этого `accept_challenges`
    висит в ожидании вызова вечно, и бот выглядит живым, хотя играть уже не может.
    """
    started = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        client = getattr(player, "ps_client", None)
        ws = getattr(client, "websocket", None)
        if ws is None:
            if time.monotonic() - started < grace:
                continue
            return "нет websocket-соединения"
        if getattr(ws, "closed", False):
            return "websocket закрыт (сервер разорвал связь или пропала сеть)"
        logged_in = getattr(client, "logged_in", None)
        if logged_in is not None and not logged_in.is_set():
            return "потерян логин на сервере"


async def _serve_player(player, spec: BotSpec, *, battles: int, login_timeout: float,
                        check_interval: float, grace: float, announce: bool, sleep) -> None:
    """Одна «сессия» бота: логин -> приветствие -> бесконечный приём вызовов под надзором."""
    await asyncio.wait_for(player.ps_client.wait_for_login(), timeout=login_timeout)
    logger.info("[%s] залогинен, приём вызовов (до %d боёв параллельно)", spec.name, battles)

    if announce:
        try:
            await announce_in_lobby(player, spec)
        except Exception:
            logger.exception("[%s] не удалось представиться в лобби (продолжаю играть)", spec.name)

    while True:
        accept_task = asyncio.create_task(player.accept_challenges(None, ACCEPT_BATCH))
        watch_task = asyncio.create_task(_connection_watchdog(player, interval=check_interval, grace=grace))
        done, pending = await asyncio.wait({accept_task, watch_task},
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        # ошибка приёма вызовов важнее причины watchdog: её traceback информативнее
        if accept_task in done and not accept_task.cancelled() and accept_task.exception() is not None:
            raise accept_task.exception()
        if watch_task in done:
            raise BotRestartRequired(watch_task.result())

        logger.warning("[%s] accept_challenges завершился штатно (%s боёв сыграно) — принимаю дальше",
                       spec.name, getattr(player, "n_finished_battles", "?"))
        await sleep(1.0)


async def _cleanup_player(player, name: str) -> None:
    """Закрывает соединение и чистит состояние перед перезапуском."""
    if player is None:
        return
    try:
        await player.ps_client.stop_listening()
    except Exception:
        logger.debug("[%s] stop_listening не отработал (соединение уже закрыто?)", name, exc_info=True)
    with contextlib.suppress(Exception):
        player._battles.clear()


async def supervise(spec: BotSpec, factory, *, restart_delay: float = RESTART_DELAY,
                    max_restart_delay: float = RESTART_MAX_DELAY,
                    healthy_run_seconds: float = HEALTHY_RUN_SECONDS,
                    sleep=asyncio.sleep, **serve_kwargs) -> None:
    """Обёртка «бот всегда работает»: падение -> лог с traceback -> пауза -> новый бот.

    Перезапуск создаёт бота заново (новый логин и websocket), а не пытается «починить»
    старый объект: после разрыва у poke-env остаются незакрытые задачи, и повторное
    использование того же Player нередко приводит к зависанию вместо восстановления.
    """
    name = spec.name
    attempt = 0
    while True:
        player = None
        started = time.monotonic()
        try:
            logger.info("[%s] запуск бота (попытка %d)", name, attempt + 1)
            player = factory()
            await _serve_player(player, spec, **serve_kwargs, sleep=sleep)
        except asyncio.CancelledError:
            logger.info("[%s] остановлен по запросу", name)
            await _cleanup_player(player, name)
            raise
        except Exception as exc:
            uptime = time.monotonic() - started
            attempt = 0 if uptime >= healthy_run_seconds else attempt + 1
            delay = min(max_restart_delay, restart_delay * (2 ** min(attempt - 1, 6))) if attempt else restart_delay
            logger.exception("[%s] бот упал после %.0f c (%s: %s)%s. Перезапуск через %.0f c",
                             name, uptime, type(exc).__name__, exc, _player_stats(player), delay)
            await _cleanup_player(player, name)
            await sleep(delay)
        else:   # штатное завершение (для accept_challenges(None, ...) не случается)
            await _cleanup_player(player, name)
            logger.warning("[%s] бот завершил работу без ошибки — перезапуск через %.0f c",
                           name, restart_delay)
            await sleep(restart_delay)


def make_factory(policy, spec: BotSpec, credentials, *, battles: int = BATTLES_PER_BOT, **extra):
    """Фабрика ботов одного типа: `supervise` вызывает её при каждом перезапуске."""
    def factory():
        return make_player(policy, spec, credentials, battles=battles, **extra)
    return factory


# ------------------------------------------------------------------- запуск ---
def _load_credentials():
    """LOGIN/LOGIN_2/PASSWORD/CUSTOM_SERVER из локального config.py (он в .gitignore)."""
    try:
        from config import LOGIN, LOGIN_2, PASSWORD, CUSTOM_SERVER
    except ImportError as exc:
        raise SystemExit(
            "Не найден config.py с LOGIN/LOGIN_2/PASSWORD/CUSTOM_SERVER.\n"
            "Создайте его рядом с index.py:\n"
            "    LOGIN = \"аккаунт_для_randombattle\"\n"
            "    LOGIN_2 = \"аккаунт_для_fusionmons\"\n"
            "    PASSWORD = \"пароль\"\n"
            "    CUSTOM_SERVER = ServerConfiguration(\"wss://.../showdown/websocket\", \"https://...\")\n"
        ) from exc
    return LOGIN, LOGIN_2, PASSWORD, CUSTOM_SERVER


def build_specs(credentials) -> dict:
    login, login_2, _, _ = credentials
    return {
        "fusion": BotSpec(
            name="fusion",
            account=login_2,
            avatar="schoolkid-gen4dp",
            battle_format=BATTLE_FORMAT_FUSION,
            greeting="Привет. Я бот, способный сыграть в Fusionmons Random Battle. "
                     "Просто вызови меня на бой в этом формате, и я с тобой сыграю!",
        ),
        "random": BotSpec(
            name="random",
            account=login,
            avatar="clown",
            battle_format=BATTLE_FORMAT_RANDOM,
            greeting="Привет! Я бот, способный сыграть в Random Battle. "
                     "Просто вызови меня на бой в этом формате, и я с тобой сыграю!",
        ),
    }


async def main(args) -> None:
    setup_logging(args.log_file, level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("=" * 70)
    logger.info("Запуск: до %d параллельных боёв на бота, лог %s",
                args.battles, os.path.abspath(args.log_file))

    credentials = _load_credentials()
    specs = build_specs(credentials)
    chosen = list(specs) if args.bot == "all" else [args.bot]

    policy = load_policy(args.model)   # грузим ДО выхода в сеть: битый файл — сразу понятная ошибка
    normalizer = None
    if not args.no_normalize_obs:
        normalizer = load_obs_normalizer(log=logger.info)
    else:
        logger.info("нормализация obs отключена флагом --no-normalize-obs")

    factories = []
    for key in chosen:
        spec = specs[key]
        factories.append(make_factory(policy, spec, credentials, battles=args.battles,
                                      obs_normalizer=normalizer))
        logger.info("бот '%s': аккаунт %s, формат %s, параллельных боёв %d",
                    spec.name, spec.account, spec.battle_format, args.battles)

    announce = not args.no_announce
    await asyncio.gather(*[
        supervise(spec, factory, login_timeout=args.login_timeout,
                  check_interval=CONNECTION_CHECK_INTERVAL, grace=CONNECTION_GRACE,
                  announce=announce, battles=args.battles)
        for spec, factory in zip((specs[k] for k in chosen), factories)
    ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Запуск ботов-ассистентов с авто-перезапуском")
    parser.add_argument("--bot", choices=["all", "fusion", "random"], default="all",
                        help="какого бота запускать (по умолчанию оба)")
    parser.add_argument("--battles", type=int, default=BATTLES_PER_BOT,
                        help=f"параллельных боёв на бота (по умолчанию {BATTLES_PER_BOT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"чекпоинт (по умолчанию {DEFAULT_MODEL})")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE,
                        help=f"лог-файл с ротацией (по умолчанию {DEFAULT_LOG_FILE})")
    parser.add_argument("--login-timeout", type=float, default=LOGIN_TIMEOUT,
                        help="сколько секунд ждать логин до перезапуска")
    parser.add_argument("--no-normalize-obs", action="store_true", dest="no_normalize_obs",
                        help="не применять статистику VecNormalize к obs (нужно, если модель "
                             "обучали с --no-normalize-bc, то есть без нормализации)")
    parser.add_argument("--no-announce", action="store_true",
                        help="не писать приветствие в лобби при каждом перезапуске")
    parser.add_argument("--verbose", action="store_true", help="DEBUG в лог")
    args = parser.parse_args(argv)
    if args.battles < 1:
        parser.error("--battles должен быть >= 1 (0 в poke-env означает «без лимита», "
                     "но тогда бот примет все вызовы подряд)")
    return args


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем (Ctrl+C)")
    except SystemExit:
        raise
    except Exception:
        # фатальная ошибка на старте (нет config.py, битый чекпоинт, нет прав на лог) —
        # тоже должна остаться в лог-файле, а не только в консоли
        logger.exception("Фатальная ошибка при запуске — процесс остановлен")
        sys.exit(1)
