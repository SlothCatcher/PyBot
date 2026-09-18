"""Загрузка чекпоинтов, обученных на другой размерности obs (авто-миграция + кэш).

Зачем: после добавления 87 признаков урона (N_FEATURES 715 -> 802) все ранее сохранённые
снапшоты — включая qualified-снапшоты, которые self-play подставляет как оппонентов, —
перестали грузиться (`size mismatch ... [512, 715] vs [512, 802]`). Обучение при этом
молча шло без self-play, а лог засыпало повторяющимися ошибками на каждой фазе.

Здесь: определяем размерность из весов, при расхождении мигрируем (паддинг нулями, см.
`policy_player._migrate_checkpoint_dim`) и кэшируем результат в `models/_migrated/`,
чтобы 8 воркеров не делали одну и ту же работу на каждой фазе.
"""
from __future__ import annotations

import os
import shutil
import tempfile

DEFAULT_CACHE_DIR = os.path.join("models", "_migrated")

_notified: set = set()


def checkpoint_obs_dim(path: str):
    """Размерность obs из весов/метаданных чекпоинта. None, если не прочитать."""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file

        data, params, _ = load_from_zip_file(path, device="cpu")
        for _, v in params.items():
            if isinstance(v, dict):
                for kk, tt in v.items():
                    if kk.endswith("features_extractor.net.0.weight") and getattr(tt, "dim", lambda: 0)() == 2:
                        return int(tt.shape[1])
        space = data["observation_space"]
        try:
            return int(space["observation"].shape[0])
        except Exception:
            return int(getattr(space, "shape", [0])[0]) or None
    except Exception:
        return None


def cached_migration_path(path: str, target_dim: int, cache_dir: str | None = None) -> str:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_dir, f"{stem}__obs{int(target_dim)}.zip")


def load_policy_compat(path: str, target_obs_dim: int, cache_dir: str | None = None,
                       quiet: bool = False):
    """PPO из `path` с autor-миграцией под target_obs_dim. Возвращает (ppo, info).

    info: {"loaded": bool, "migrated_from": int|None, "cached": bool, "error": str|None}
    """
    info = {"loaded": False, "migrated_from": None, "cached": False, "error": None}
    if not os.path.isfile(path):
        info["error"] = f"файл не найден: {path}"
        return None, info

    dim = checkpoint_obs_dim(path)
    from stable_baselines3 import PPO

    if dim is None or int(dim) == int(target_obs_dim):
        try:
            info["loaded"] = True
            return PPO.load(path, device="cpu"), info
        except Exception as e:
            info["error"] = str(e).splitlines()[0]
            return None, info

    info["migrated_from"] = int(dim)
    cached = cached_migration_path(path, target_obs_dim, cache_dir)
    if os.path.isfile(cached):
        try:
            info["loaded"] = True
            info["cached"] = True
            return PPO.load(cached, device="cpu"), info
        except Exception:
            try:
                os.remove(cached)   # битый кэш — пересоберём
            except Exception:
                pass

    try:
        from agents.policy_player import _migrate_checkpoint_dim

        ppo = _migrate_checkpoint_dim(path, target_dim=int(target_obs_dim), verbose=False)
    except Exception as e:
        info["error"] = str(e).splitlines()[0]
        return None, info

    # кэшируем: атомарная запись через временный файл (гонка 8 воркеров безопасна —
    # результат миграции идентичен)
    try:
        os.makedirs(cache_dir or DEFAULT_CACHE_DIR, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="mig_", dir=cache_dir or DEFAULT_CACHE_DIR)
        tmp_path = os.path.join(tmp_dir, "snap")
        ppo.save(tmp_path)
        produced = tmp_path + ".zip" if os.path.isfile(tmp_path + ".zip") else tmp_path
        os.replace(produced, cached)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        if not quiet:
            key = (path, int(target_obs_dim))
            if key not in _notified:
                _notified.add(key)
                print(f"  self-play снапшот {os.path.basename(path)}: мигрирован "
                      f"{dim} -> {target_obs_dim}, кэш {cached}")
    except Exception:
        pass  # кэш — оптимизация; без него просто будем мигрировать снова

    info["loaded"] = True
    return ppo, info
