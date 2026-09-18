"""Загрузка статистик VecNormalize для инференса и resume с любой старой размерностью.

Зачем: `models/vecnormalize.pkl` в репозитории хранит `obs_rms["observation"].mean` формы
(418,) — статистика от старой версии признаков. В `agents/policy_player.py` миграция умела
только 713 -> 715, поэтому 418 либо падал на `normalize_obs` (mean (418,) против obs (715,)),
либо молча давал мусор. Плюс автономный инференс (`index.py`) вообще не применял
нормализацию, хотя обучение шло с `norm_obs=True` — модель получала ненормализованный obs.

Здесь:
  * `load_vecnorm_state` — читает статистику из pkl устойчиво (обычный dict-пикл SB3 или
    объект со `__dict__`), размерность может быть любой;
  * `pad_stats` — добивает mean/var нулями/единицами до целевой размерности;
  * `VecNormStats` — минимальная замена `VecNormalize.normalize_obs` (формула 1:1 с SB3):
    `clip((x - mean) / sqrt(var + eps), -clip_obs, clip_obs)`;
  * `load_vecnormalize_for_dim` — версия для обучения (resume): грузит SB3 VecNormalize и
    подгоняет obs_rms под целевую размерность через `pad_stats`.
"""
from __future__ import annotations

import pickle
from typing import Any, Optional

import numpy as np

DEFAULT_EPS = 1e-8  # SB3 использует epsilon=1e-8 в _normalize_obs


class _Unpickler(pickle.Unpickler):
    """SB3-пикл может ссылаться на класс VecNormalize — грузим его как «мешок» со state."""

    def find_class(self, module, name):  # noqa: D102
        if "vec_normalize" in module or "VecNormalize" in name:
            class _Bag:
                def __setstate__(self, state):
                    self.__dict__.update(state if isinstance(state, dict) else {"_state": state})
            return _Bag
        return super().find_class(module, name)


def _state_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    d = getattr(obj, "__dict__", None)
    return dict(d) if d else {}


def load_vecnorm_state(path: str) -> Optional[dict]:
    """Статистика VecNormalize из pkl (любой формы/размерности). None, если не прочитать."""
    try:
        with open(path, "rb") as f:
            data = _state_dict(_Unpickler(f).load())
    except Exception:
        try:
            with open(path, "rb") as f:
                data = _state_dict(pickle.load(f))
        except Exception:
            return None

    obs_rms = data.get("obs_rms")
    if obs_rms is None:
        # иногда статистика лежит под ключом "stats"/"_stats"
        for key in ("stats", "_stats"):
            if isinstance(data.get(key), dict) and "obs_rms" in data[key]:
                obs_rms = data[key]["obs_rms"]
                break
    if obs_rms is None:
        return None
    rms = obs_rms.get("observation") if isinstance(obs_rms, dict) else obs_rms
    if rms is None:
        return None
    return {
        "mean": np.asarray(getattr(rms, "mean", None) if not isinstance(rms, dict) else rms.get("mean"), dtype=np.float64),
        "var": np.asarray(getattr(rms, "var", None) if not isinstance(rms, dict) else rms.get("var"), dtype=np.float64),
        "count": float(getattr(rms, "count", 0.0) if not isinstance(rms, dict) else rms.get("count", 0.0)),
        "norm_obs": bool(data.get("norm_obs", True)),
        "norm_reward": bool(data.get("norm_reward", False)),
        "clip_obs": float(data.get("clip_obs", 10.0)),
        "epsilon": float(data.get("epsilon", DEFAULT_EPS) or DEFAULT_EPS),
    }


def pad_stats(mean: np.ndarray, var: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Подгоняет статистику под target_dim. Новые признаки: mean=0, var=1 (нейтрально).

    Возвращает (mean, var, изменилось_ли)."""
    mean = np.asarray(mean, dtype=np.float64).ravel()
    var = np.asarray(var, dtype=np.float64).ravel()
    if mean.shape[0] == target_dim and var.shape[0] == target_dim:
        return mean, var, False
    new_mean = np.zeros(target_dim, dtype=np.float64)
    new_var = np.ones(target_dim, dtype=np.float64)
    n = min(target_dim, mean.shape[0], var.shape[0])
    new_mean[:n] = mean[:n]
    new_var[:n] = var[:n]
    return new_mean, new_var, True


class VecNormStats:
    """Применение статистик к obs (замена VecNormalize.normalize_obs для одного вектора)."""

    def __init__(self, mean, var, clip_obs: float = 10.0, epsilon: float = DEFAULT_EPS,
                 target_dim: int | None = None, source_dim: int | None = None, migrated: bool = False):
        self.clip_obs = float(clip_obs)
        # SB3 добавляет epsilon=1e-8, поэтому var=0 не даёт деления на ноль
        self.eps = float(epsilon) if epsilon else DEFAULT_EPS
        self.source_dim = int(source_dim if source_dim is not None else np.asarray(mean).size)
        self.target_dim = int(target_dim) if target_dim is not None else self.source_dim
        m, v, self.migrated = pad_stats(mean, var, self.target_dim)
        self.mean = m.astype(np.float32)
        self.var = v.astype(np.float32)

    @property
    def ready(self) -> bool:
        return self.mean.size == self.target_dim and self.var.size == self.target_dim

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.shape[-1] != self.mean.size:
            raise ValueError(f"obs {x.shape[-1]} != статистика {self.mean.size}")
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -self.clip_obs, self.clip_obs).astype(np.float32)

    def describe(self) -> str:
        s = (f"VecNormalize: stats {self.source_dim} -> {self.target_dim}"
             f"{' (добито mean=0, var=1)' if self.migrated else ''}, clip_obs={self.clip_obs}")
        return s


def load_vecnorm_stats(path: str, target_dim: int) -> Optional[VecNormStats]:
    """Статистики для инференса: читает pkl и добивает до target_dim (None, если не вышло)."""
    st = load_vecnorm_state(path)
    if st is None or st["mean"] is None or st["var"] is None:
        return None
    if not st["norm_obs"]:
        return None
    return VecNormStats(st["mean"], st["var"], st["clip_obs"], st["epsilon"], target_dim=target_dim)


def load_vecnormalize_for_dim(vec_path: str, base_env, target_dim: int | None = None):
    """SB3 VecNormalize с подгонкой obs_rms под целевую размерность (для обучения/resume).

    Заменяет частную миграцию 713->715: работает с любой старой размерностью (418, 713, ...).
    """
    from stable_baselines3.common.vec_env import VecNormalize

    vec = VecNormalize.load(vec_path, base_env)
    try:
        if target_dim is None:
            space = getattr(base_env, "observation_space", None)
            try:
                target_dim = space["observation"].shape[0]  # type: ignore[index]
            except Exception:
                target_dim = None
        rms = vec.obs_rms["observation"] if isinstance(vec.obs_rms, dict) else vec.obs_rms
        old_mean = np.asarray(rms.mean)
        if target_dim is not None and old_mean.shape[0] != int(target_dim):
            new_mean, new_var, _ = pad_stats(rms.mean, rms.var, int(target_dim))
            rms.mean = new_mean.astype(rms.mean.dtype)
            rms.var = new_var.astype(rms.var.dtype)
            print(f"  VecNormalize: obs_rms {old_mean.shape[0]} -> {target_dim} "
                  f"(новые признаки mean=0, var=1, count={getattr(rms, 'count', '?')})")
    except Exception as e:
        print(f"  VecNormalize: не удалось подогнать obs_rms: {e}")
    return vec
