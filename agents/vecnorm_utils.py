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
  * `stats_verdict` — решает, можно ли ДОВЕРЯТЬ статистике из файла (см. ниже);
  * `reset_obs_rms` — сброс статистики (mean=0, var=1, count=0) с переоценкой по текущим данным;
  * `save_vecnormalize_with_meta` — сохраняет VecNormalize вместе с сайдкаром
    `*.meta.json` (размерность + хеш кода признаков): только так можно понять, что
    статистика относится к ТЕКУЩЕЙ раскладке, а не к старой;
  * `load_vecnormalize_for_dim` — версия для обучения (resume): грузит SB3 VecNormalize,
    проверяет статистику через `stats_verdict` и при несовместимости сбрасывает её, а не
    «добивает нулями».

Почему нельзя переиспользовать статистику чужой раскладки. VecNormalize хранит per-колонку
mean/var, то есть статистику КОНКРЕТНЫХ признаков. Раскладка менялась несколько раз
(418 -> 713 -> 715 -> 802 -> 870), причём 713 -> 715 вставлял два признака В СЕРЕДИНУ (перед
tera_type). Простое добивание нулями оставляет старые 418 колонок рядом с уже другими
признаками, а `count` у SB3 ~200k, из-за чего испорченная статистика почти не вымывается
новыми данными: модель тренируется на нормализации чужими статистиками (часть колонок
уходит в клип ±10). Поэтому при несовместимости статистику надо сбрасывать и переоценивать.
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

    stale: bool = False          # статистика не от текущей раскладки (см. stats_verdict)
    provenance: str = ""         # человекочитаемая причина вердикта

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
        if self.provenance:
            s += f"; {self.provenance}"
        return s

    def stale_warning(self) -> str:
        """Предупреждение для инференса/self-play, если статистика от чужой раскладки."""
        if not self.stale:
            return ""
        return (f"ВНИМАНИЕ: статистика нормализации не от текущей раскладки признаков "
                f"({self.provenance}). Инференс применяет её же, как при обучении — это "
                f"корректно только для моделей, обученных на ней. Для чистого обучения "
                f"используйте --reset-obs-stats.")


def load_vecnorm_stats(path: str, target_dim: int) -> Optional[VecNormStats]:
    """Статистики для инференса: читает pkl, проверяет провенанс и добивает до target_dim.

    Инференс обязан применять РОВНО ту нормализацию, с которой модель обучалась: если
    статистика старая (другая раскладка) и её когда-то добили нулями при обучении, то
    на инференсе нужно добить так же — иначе obs разъедется с обучающим. Поэтому здесь
    статистика НЕ отбрасывается, но помечается (`stale`, `provenance`) для предупреждения.
    """
    st = load_vecnorm_state(path)
    if st is None or st["mean"] is None or st["var"] is None:
        return None
    if not st["norm_obs"]:
        return None
    out = VecNormStats(st["mean"], st["var"], st["clip_obs"], st["epsilon"], target_dim=target_dim)
    try:
        keep, reason = stats_verdict(path, int(target_dim), st["mean"], st["var"])
        out.provenance = reason
        out.stale = not keep
    except Exception:
        out.provenance = "провенанс не определён"
        out.stale = False
    return out


# ------------------------------------------------------- проверка статистики ---

def features_fingerprint() -> str:
    """Хеш кода признаков (features.py + damage.py + config.py) — тот же, что у кэша датасета."""
    try:
        from .training import _features_fingerprint
        return str(_features_fingerprint())
    except Exception:
        return ""


def stats_meta_path(path: str) -> str:
    return path + ".meta.json"


def read_stats_meta(path: str) -> Optional[dict]:
    """Сайдкар статистики, если он есть: {obs_dim, features_hash, saved_at}."""
    try:
        import json
        with open(stats_meta_path(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_stats_meta(path: str, obs_dim: int, fingerprint: Optional[str] = None) -> Optional[dict]:
    try:
        import json
        import time
        meta = {
            "obs_dim": int(obs_dim),
            "features_hash": fingerprint if fingerprint is not None else features_fingerprint(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(stats_meta_path(path), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return meta
    except Exception:
        return None


def padded_column_mask(mean, var) -> np.ndarray:
    """Колонки, которые выглядят как паддинг: mean ровно 0.0 и var ровно 1.0.

    У реальных признаков такие значения одновременно практически не встречаются
    (var=1.0 и mean=0.0 с точностью float — это подпись «добивания нулями»).
    """
    m = np.asarray(mean, dtype=np.float64).ravel()
    v = np.asarray(var, dtype=np.float64).ravel()
    if m.size == 0 or m.size != v.size:
        return np.zeros(0, dtype=bool)
    return (m == 0.0) & (v == 1.0)


def stats_verdict(path: str, target_dim: int, mean, var,
                  fingerprint: Optional[str] = None,
                  min_padded_share: float = 0.05) -> tuple[bool, str]:
    """Можно ли доверять статистике из `path` для текущих признаков: (keep, причина).

    Порядок проверок:
      1. Есть сайдкар и он совпадает (размерность + хеш кода признаков) -> доверяем.
      2. Размерность статистики != текущей -> доверять нельзя (раскладка менялась,
         старые колонки означают уже другие признаки).
      3. Похоже на паддинг (много колонок mean=0/var=1) -> доверять нельзя.
      4. Размерность совпадает, сайдкара нет -> доверяем с оговоркой про провенанс
         (пользователь может форсировать сброс флагом).
    """
    src = int(np.asarray(mean).size)
    meta = read_stats_meta(path)
    fp = fingerprint if fingerprint is not None else features_fingerprint()
    if meta and int(meta.get("obs_dim", -1)) == int(target_dim) and fp and meta.get("features_hash") == fp:
        return True, f"сайдкар совпадает (obs {target_dim}, код признаков тот же)"
    if src != int(target_dim):
        return False, (f"статистика на {src} признаков, а сейчас {target_dim}: раскладка признаков "
                       f"менялась, старые колонки означают другое")
    n_pad = int(np.count_nonzero(padded_column_mask(mean, var)))
    if n_pad and n_pad >= max(4, int(min_padded_share * src)):
        return False, (f"{n_pad} из {src} колонок выглядят как паддинг (mean=0, var=1) — "
                       f"статистика собрана из старой раскладки")
    if meta is None:
        # сайдкара нет (файл от старой версии кода) — размерность совпала, доверяем с оговоркой:
        # безусловно сбрасывать нельзя, иначе потеряем корректную статистику у пользователей,
        # у которых раскладка не менялась
        return True, "размерность совпадает, сайдкара нет (провенанс неизвестен)"
    return False, ("сайдкар есть, но хеш кода признаков другой: при той же размерности раскладка "
                   "могла измениться (порядок/значения колонок)")


def reset_obs_rms(vec, dim: Optional[int] = None) -> bool:
    """Сбрасывает статистику obs (mean=0, var=1, count=0): нормализация переоценится заново.

    `dim` — размерность ТЕКУЩИХ признаков: без неё размерность осталась бы старой, и
    `normalize_obs` падал бы на broadcast (mean (418,) против obs (870,)).
    """
    try:
        rms = vec.obs_rms["observation"] if isinstance(vec.obs_rms, dict) else vec.obs_rms
        if rms is None:
            return False
        n = int(dim) if dim else int(np.asarray(rms.mean).size)
        rms.mean = np.zeros(n, dtype=np.float64)
        rms.var = np.ones(n, dtype=np.float64)
        rms.count = 0.0
        return True
    except Exception:
        return False


def vecnormalize_of(env):
    """Внутренний VecNormalize (env может быть обёрнут, например CuriosityVecWrapper)."""
    from stable_baselines3.common.vec_env import VecNormalize

    seen = set()
    while env is not None and id(env) not in seen:
        seen.add(id(env))
        if isinstance(env, VecNormalize):
            return env
        env = getattr(env, "venv", None)
    return None


def save_vecnormalize_with_meta(env, path: str, fingerprint: Optional[str] = None) -> bool:
    """`env.save(path)` + сайдкар с размерностью и хешем признаков.

    Без сайдкара на следующем resume нельзя отличить «статистику текущей раскладки» от
    «статистики, добитой нулями от старой» — и в модель уезжает неверная нормализация.
    """
    vec = vecnormalize_of(env)
    try:
        env.save(path)
    except Exception:
        return False
    try:
        rms = vec.obs_rms["observation"] if isinstance(vec.obs_rms, dict) else vec.obs_rms
        write_stats_meta(path, int(np.asarray(rms.mean).size), fingerprint)
    except Exception:
        pass
    return True


class LiveVecNormalizeAdapter:
    """Adapter вокруг живого SB3 VecNormalize: `normalize(obs)` для одного вектора признаков."""

    def __init__(self, vec_norm, target_dim: Optional[int] = None):
        self.vec_norm = vec_norm
        self.target_dim = int(target_dim) if target_dim is not None else None

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if self.target_dim is not None and arr.shape[-1] != self.target_dim:
            raise ValueError(f"obs {arr.shape[-1]} != {self.target_dim}")
        out = self.vec_norm.normalize_obs({"observation": arr[None, :]})
        return np.asarray(out["observation"][0], dtype=np.float32)

    def describe(self) -> str:
        try:
            rms = self.vec_norm.obs_rms["observation"] if isinstance(self.vec_norm.obs_rms, dict) else self.vec_norm.obs_rms
            return f"живой VecNormalize: obs {int(np.asarray(rms.mean).size)}, clip_obs={getattr(self.vec_norm, 'clip_obs', '?')}"
        except Exception:
            return "живой VecNormalize"


def _load_vecnormalize_tolerant(vec_path: str, base_env, target_dim: int):
    """VecNormalize из pkl БЕЗ проверки размерности статистики.

    Штатный `VecNormalize.load` падает, если `observation_space` из pkl не совпадает с env,
    поэтому статистику сначала правим (сброс/паддинг), а уже потом привязываем env.
    Возвращает None, если штатный путь и так сработает (размерности совпали) или не вышло.
    """
    import pickle

    from stable_baselines3.common.vec_env import VecNormalize

    try:
        with open(vec_path, "rb") as fh:
            vec = pickle.load(fh)   # SB3 хранит объект без venv (__getstate__ его выкидывает)
    except Exception:
        return None
    if not isinstance(vec, VecNormalize):
        return None
    return vec


def _bind_venv(vec, base_env, *, quiet: bool = False) -> bool:
    """Привязывает env к загруженному VecNormalize, игнорируя расхождение shape."""
    try:
        vec.observation_space = base_env.observation_space
    except Exception:
        pass
    try:
        vec.set_venv(base_env)
        return True
    except Exception as e:
        if not quiet:
            print(f"  VecNormalize: не удалось привязать env ({e})")
        return False


def load_vecnormalize_for_dim(vec_path: str, base_env, target_dim: Optional[int] = None,
                              *, force_reset: bool = False, keep_stale: bool = False,
                              quiet: bool = False):
    """SB3 VecNormalize под текущую размерность признаков, с проверкой статистики.

    force_reset — сбросить статистику независимо от вердикта (`--reset-obs-stats`);
    keep_stale  — оставить даже несовместимую статистику (`--keep-obs-stats`).
    """
    from stable_baselines3.common.vec_env import VecNormalize

    if target_dim is None:
        try:
            target_dim = int(base_env.observation_space["observation"].shape[0])
        except Exception:
            target_dim = None

    vec = _load_vecnormalize_tolerant(vec_path, base_env, int(target_dim) if target_dim else 0)
    if vec is None:
        # штатный путь: статистика совпала с env (или pkl не читается — тогда прежнее поведение)
        vec = VecNormalize.load(vec_path, base_env)
        try:
            rms = vec.obs_rms["observation"] if isinstance(vec.obs_rms, dict) else vec.obs_rms
            mean = np.asarray(rms.mean)
        except Exception:
            mean = None
        if mean is not None and not force_reset:
            keep, reason = stats_verdict(vec_path, int(target_dim or mean.size), mean,
                                         np.asarray(rms.var))
            if keep:
                if not quiet:
                    print(f"  VecNormalize: статистика принята ({reason})")
                return vec
            if not keep_stale:
                reset_obs_rms(vec, int(target_dim) if target_dim else None)
                print(f"  VecNormalize: статистика не подходит ({reason}) -> сброшена "
                      f"(mean=0, var=1, count=0), переоценится по текущим признакам")
                return vec
        return vec

    try:
        rms = vec.obs_rms["observation"] if isinstance(vec.obs_rms, dict) else vec.obs_rms
    except Exception:
        rms = None
    if rms is None:
        _bind_venv(vec, base_env, quiet=quiet)
        return vec

    src = int(np.asarray(rms.mean).size)
    if force_reset:
        reset_obs_rms(vec, int(target_dim) if target_dim else None)
        print(f"  VecNormalize: статистика {src} сброшена по флагу --reset-obs-stats "
              f"(mean=0, var=1, count=0)")
    else:
        keep, reason = stats_verdict(vec_path, int(target_dim or src), rms.mean, rms.var)
        if not keep and not keep_stale:
            reset_obs_rms(vec, int(target_dim) if target_dim else None)
            print(f"  VecNormalize: статистика не подходит ({reason}) -> сброшена "
                  f"(mean=0, var=1, count=0), переоценится по текущим признакам")
        else:
            if not keep:
                print(f"  VecNormalize: статистика несовместима ({reason}), но оставлена "
                      f"по флагу --keep-obs-stats")
            if target_dim is not None and src != int(target_dim):
                new_mean, new_var, _ = pad_stats(rms.mean, rms.var, int(target_dim))
                rms.mean = new_mean.astype(np.asarray(rms.mean).dtype)
                rms.var = new_var.astype(np.asarray(rms.var).dtype)
                print(f"  VecNormalize: obs_rms {src} -> {target_dim} (добито mean=0, var=1)")

    _bind_venv(vec, base_env, quiet=quiet)
    return vec
