"""Игра обученной моделью с полным набором фиксов (правильный путь инференса).

Что делает:
  * грузит PPO-снапшот и применяет нормализацию obs из VecNormalize (обучение шло с
    `norm_obs=True`, а `index.py`/`test.py` её не применяли — модель играла «вслепую»);
  * использует `agents.players.PolicyPlayer` (миксин FusionInfoParser) — то есть работают
    и перестановка `typechange` перед `|request|`, и разбор статов фьюжна из html;
  * печатает диагностику (PYBOT_DEBUG_TYPES=1 — подробнее).

Режимы:
    python play_trained.py --selfcheck                     # проверка без сервера
    python play_trained.py --ladder 10                     # лестница
    python play_trained.py --challenge Someone             # принять/бросить вызовы от игрока
    python play_trained.py --challenge-any                 # принимать любые вызовы

Если снапшот обучен на старой размерности признаков (например 715, а в config уже 802),
веса автоматически добиваются нулями (`--no-migrate` отключает) — модель играет как раньше
и может дообучаться уже на новых признаках.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

os.environ.setdefault("PYBOT_DEBUG_TYPES", "1")

import numpy as np

from agents.config import BATTLE_FORMAT, N_FEATURES, VECNORM_PATH
from agents.players import PolicyPlayer
from agents.fusion_parser import FusionInfoParser, _hoist_typechanges_before_request
from agents.vecnorm_utils import load_vecnorm_stats


def selfcheck(model_path: str, vecnorm_path: str) -> int:
    """Проверка пайплайна инференса без подключения к серверу."""
    ok = True
    print("=" * 78)
    print("SELFCHECK инференса обученной модели")
    print("=" * 78)

    # 1. размерность модели читаем из метаданных zip — работает даже если веса от старой версии
    model_dim = None
    try:
        from stable_baselines3.common.save_util import load_from_zip_file

        data, _, _ = load_from_zip_file(model_path)
        space = data["observation_space"]
        try:
            model_dim = int(space["observation"].shape[0])
        except Exception:
            model_dim = int(np.prod(getattr(space, "shape", [0])))
        print(f"[1] модель {model_path}: obs dim = {model_dim}")
        print(f"    N_FEATURES в config = {N_FEATURES} -> "
              f"{'СОВПАДАЕТ ✓' if model_dim == N_FEATURES else 'НЕ СОВПАДАЕТ ✗ (снапшот от другой версии признаков)'}")
        if model_dim != N_FEATURES:
            print(f"    -> сработает авто-миграция весов: {model_dim} -> {N_FEATURES} (warm start)")
    except Exception as e:
        print(f"[1] не удалось прочитать метаданные модели: {e}")
        ok = False

    # 1b. веса реально грузятся в текущую политику? (при несовпадении — через авто-миграцию)
    try:
        ppo_probe, migrated_from = load_policy(model_path, allow_migrate=True)
        _w = ppo_probe.policy.features_extractor.net[0].weight
        if migrated_from is None:
            print("    веса применяются к текущей политике: ✓")
        else:
            print(f"    веса применяются после авто-миграции {migrated_from}->{N_FEATURES} ✓ "
                  f"(первый слой {tuple(_w.shape)}, новые признаки с нулевыми весами)")
            print("    POLICY BEHAVIOUR: старые признаки дают тот же результат, что и раньше; "
                  "новые включатся по мере дообучения")
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"    веса НЕ применяются к текущей политике ✗: {first}")
        print("    (нужен снапшот, обученный на текущих признаках/архитектуре)")
        ok = False

    # 2. нормализация obs
    stats = None
    if os.path.isfile(vecnorm_path):
        stats = load_vecnorm_stats(vecnorm_path, N_FEATURES)
        if stats is None:
            print(f"[2] {vecnorm_path}: статистика не читается или norm_obs=False")
        else:
            print(f"[2] {stats.describe()}")
            probe = np.full(N_FEATURES, 3.0, dtype=np.float32)
            try:
                out = stats.normalize(probe)
                print(f"    normalize: {probe[:1]} -> {out[:1]} (shape {out.shape}) ✓")
            except Exception as e:
                print(f"    normalize не работает: {e} ✗")
                ok = False
    else:
        print(f"[2] {vecnorm_path} не найден — нормализация не будет применена")

    # 3. фикс тайминга typechange
    race = [
        [">battle-x"],
        ["", "switch", "p2a: A", "froslass, L50, M", "100/100"],
        ["", "request", "{}"],
        ["", "-start", "p2a: A", "typechange", "Steel/Ice", "[silent]"],
    ]
    reordered, n = _hoist_typechanges_before_request(race)
    seq = [m[1] for m in reordered[1:]]
    print(f"[3] reorder typechange: перенесено {n}, порядок {seq}")
    ok &= (n == 1 and seq == ["switch", "-start", "request"])
    print(f"    PolicyPlayer использует FusionInfoParser: "
          f"{FusionInfoParser in PolicyPlayer.__mro__} "
          f"-> {'✓' if FusionInfoParser in PolicyPlayer.__mro__ else '✗'}")
    ok &= FusionInfoParser in PolicyPlayer.__mro__

    print("-" * 78)
    print("ИТОГ:", "всё подключено ✓" if ok else "есть проблемы ✗ (см. выше)")
    return 0 if ok else 1


def load_policy(model_path: str, allow_migrate: bool = True):
    """Грузит PPO; при несовпадении размерности obs добивает веса до N_FEATURES.

    Возвращает (ppo, исходная_размерность). Если размерность совпадала, второй элемент None.
    """
    from stable_baselines3 import PPO

    from agents.policy_player import _checkpoint_obs_dim, _migrate_checkpoint_dim

    dim = _checkpoint_obs_dim(model_path)
    if dim is None:
        return PPO.load(model_path, device="cpu"), None
    if dim == N_FEATURES:
        return PPO.load(model_path, device="cpu"), None
    if not allow_migrate:
        raise SystemExit(
            f"Модель {model_path} обучена на {dim} признаках, а в env сейчас {N_FEATURES}. "
            f"Запусти без --no-migrate (веса будут добиты нулями) или возьми снапшот под текущие признаки."
        )
    print(f"  снапшот {model_path}: {dim} признаков -> {N_FEATURES}, мигрирую (warm start): "
          f"старые веса сохранены, {N_FEATURES - dim} новых признаков входят с нулевыми весами")
    return _migrate_checkpoint_dim(model_path, target_dim=N_FEATURES), dim


def build_player(ppo, vecnorm_path: str, deterministic: bool, use_norm: bool):
    agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    if use_norm and os.path.isfile(vecnorm_path):
        stats = load_vecnorm_stats(vecnorm_path, N_FEATURES)
        if stats is not None:
            orig = agent.embed_battle

            def norm_embed(battle, _orig=orig, _stats=stats):
                return _stats.normalize(_orig(battle))

            agent.embed_battle = norm_embed  # type: ignore
            print(f"  {stats.describe()} -> нормализация включена")

    # сюда модель уже приходит совместимой по размерности (см. load_policy)
    try:
        dim = int(ppo.observation_space["observation"].shape[0])
        if dim != N_FEATURES:
            print(f"  ВНИМАНИЕ: модель ждёт {dim} признаков, а env даёт {N_FEATURES} — "
                  f"играть ей нельзя, нужен снапшот, обученный на текущих признаках")
    except Exception:
        pass

    if deterministic:
        orig_choose = agent.choose_move

        def det_choose(battle, _orig=orig_choose):
            return _orig(battle)

        agent.choose_move = det_choose  # type: ignore
    return agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/self_play_snapshot_6.zip")
    ap.add_argument("--vecnorm", default=VECNORM_PATH)
    ap.add_argument("--selfcheck", action="store_true", help="проверить пайплайн без сервера")
    ap.add_argument("--ladder", type=int, default=0, help="сколько боёв сыграть в лестнице")
    ap.add_argument("--challenge", type=str, default=None, help="принимать вызовы от игрока")
    ap.add_argument("--challenge-any", action="store_true", help="принимать любые вызовы")
    ap.add_argument("--no-normalize", action="store_true", help="не применять VecNormalize")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-migrate", action="store_true",
                    help="не мигрировать снапшот со старой размерностью признаков (жёсткая ошибка)")
    args = ap.parse_args()

    if args.selfcheck or not (args.ladder or args.challenge or args.challenge_any):
        return selfcheck(args.model, args.vecnorm)

    ppo, _ = load_policy(args.model, allow_migrate=not args.no_migrate)
    agent = build_player(ppo, args.vecnorm, args.deterministic, use_norm=not args.no_normalize)

    async def run():
        if args.ladder:
            await agent.ladder(args.ladder)
        elif args.challenge:
            await agent.accept_challenges(args.challenge, 1)
        else:
            while True:
                await agent.accept_challenges(None, 1)
                await asyncio.sleep(3)

    try:
        asyncio.run(run())
    finally:
        print(f"finish: {agent.n_finished_battles}, wins: {agent.n_won_battles}")
        try:
            from agents.type_utils import summary_line

            print(summary_line("[type-debug]") or "[type-debug] счётчики пусты")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
