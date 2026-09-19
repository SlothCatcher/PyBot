import argparse
import os
import time
from functools import partial

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from agents.training import collect_or_load_dataset
import numpy as np
from agents.config import BATTLE_FORMAT, MIN_WINRATE_TO_QUALIFY, N_FEATURES, QUALIFIED_PREFIX, SELF_PLAY_PATH, VECNORM_PATH
import json as _json
_QUALIFIED_META_PATH = "models/qualified_meta.json"

def _load_qualified_meta(base_min: int = 25) -> dict:
    """Грузит meta qualified: threshold + history. Если нет — создаём с base_min."""
    try:
        if os.path.isfile(_QUALIFIED_META_PATH):
            with open(_QUALIFIED_META_PATH, "r") as f:
                meta = _json.load(f)
                # валидация
                if "threshold" not in meta:
                    meta["threshold"] = base_min
                if "history" not in meta:
                    meta["history"] = []
                if "base" not in meta:
                    meta["base"] = base_min
                return meta
    except Exception as e:
        print(f"warn _load_qualified_meta: {e}")
    return {"threshold": int(base_min), "base": int(base_min), "history": []}

def _save_qualified_meta(meta: dict):
    try:
        os.makedirs(os.path.dirname(_QUALIFIED_META_PATH), exist_ok=True)
        # atomic write
        tmp = _QUALIFIED_META_PATH + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(meta, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _QUALIFIED_META_PATH)
    except Exception as e:
        print(f"warn _save_qualified_meta: {e}")

def _get_current_threshold(base_min: int) -> int:
    """Рэтчет-порог: max(base_min, max(history winrate), saved threshold). Никогда не падает."""
    meta = _load_qualified_meta(base_min)
    thr = int(meta.get("threshold", base_min))
    # также смотрим history — вдруг ручной правкой threshold занизили, но в history есть выше
    try:
        max_hist = max([int(round(h.get("winrate_vs_heuristics", 0))) for h in meta.get("history",[])] or [thr])
        thr = max(thr, max_hist)
    except Exception:
        pass
    # если base_min подняли аргументом — тоже учитываем
    thr = max(thr, int(base_min))
    return int(thr)

def _record_qualified(save_path: str, win_rates: dict, base_min: int, phase_counter: int):
    """Пишет history + обновляет threshold = max(threshold, heuristics_rate). Также пишет per-file json рядом с моделью."""
    try:
        meta = _load_qualified_meta(base_min)
        heur = float(win_rates.get("SimpleHeuristicsPlayer", 0))
        # обновляем threshold рэтчетом
        old_thr = int(meta.get("threshold", base_min))
        new_thr = max(old_thr, int(round(heur)), int(base_min))
        meta["threshold"] = new_thr
        meta["base"] = int(base_min)
        entry = {
            "file": os.path.basename(save_path),
            "path": save_path,
            "winrate_vs_heuristics": float(heur),
            "win_rates": {k: float(v) for k,v in win_rates.items()},
            "threshold_before": int(old_thr),
            "threshold_after": int(new_thr),
            "phase": int(phase_counter),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta["history"].append(entry)
        _save_qualified_meta(meta)
        # per-file json для удобства
        per_file = save_path + ".json"
        try:
            with open(per_file, "w") as f:
                _json.dump(entry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"warn per-file meta {per_file}: {e}")
        print(f"[qualified meta] порог {old_thr}% -> {new_thr}%, записан {save_path}.json (heur {heur}%)")
        return meta
    except Exception as e:
        print(f"warn _record_qualified: {e}")
        import traceback; traceback.print_exc()
        return None
from agents.env import ExampleEnv
from agents.policy import MaskedActorCriticPolicy
from agents.players import PolicyPlayer
from agents.training import (
    StepCounterCallback,
    make_lr_schedule,
    make_ent_schedule,
    _get_opponent_weights,
    _next_snapshot_index,
    _update_opponent_weights,
    collect_heuristic_dataset,
    evaluate_win_rates,
    pretrain_policy_bc,
)
import asyncio
import torch
import torch.nn as nn
try:
    from agents.curiosity import ICM, CuriosityVecWrapper
except Exception:
    ICM = None
    CuriosityVecWrapper = None

try:  # gymnasium есть всегда (зависимость SB3), но модуль должен импортироваться и без него
    from gymnasium import Env as _GymBaseEnv
except Exception:  # pragma: no cover
    class _GymBaseEnv:  # type: ignore
        pass


class _DimProbeEnv(_GymBaseEnv):
    """Минимальный env нужной размерности для сборки политики без poke-env сервера.

    Используется только для того, чтобы инстанцировать PPO с правильными shapes.
    """

    metadata = {"render_modes": []}

    def __init__(self, dim: int, action_dim: int = 26):
        import numpy as _np

        from gymnasium import spaces

        self._np = _np
        # ВАЖНО: action_dim должен совпадать с реальным env (gen9 = 6 свитчей + 5 блоков по 4
        # приёма = 26). Раньше тут жёстко стояло 9 (как у DoublesEnv): политика собиралась с
        # головой на 9 действий, и в fallback-пути миграции загрузка весов падала на
        # `size mismatch for action_net` (torch ругается на shape даже при strict=False),
        # то есть миграция не выживала вовсе — снапшот оставался без политики.
        self._action_dim = int(action_dim)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 4.0, shape=(int(dim),), dtype=_np.float32),
            "action_mask": spaces.Box(0, 1, shape=(self._action_dim,), dtype=bool),
        })
        self.action_space = spaces.Discrete(self._action_dim)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

    def close(self):
        return None


def _probe_ppo(target_dim: int, features_dim: int = 512, action_dim: int = 26,
               net_arch=None, legacy_extractor: bool = False):
    """PPO со случайной политикой нужной формы (для fallback-миграции).

    net_arch и класс экстрактора берутся из самого чекпоинта: старые снапшоты обучены с
    net_arch=[512,256,128] (общие головы) и identity-экстрактором (LegacyFeaturesExtractor,
    без `features_extractor.net.*` весов). Если собрать пробу с текущими дефолтами
    ([512,256] + Linear-экстрактор), `load_state_dict` падает на size mismatch и снапшот
    вообще не загружается ("Failed to load qualified snapshot").
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.policy import LegacyFeaturesExtractor, MaskedActorCriticPolicy

    kwargs = {}
    if legacy_extractor:
        kwargs["features_extractor_class"] = LegacyFeaturesExtractor
    else:
        kwargs["features_extractor_kwargs"] = dict(features_dim=int(features_dim))
    if net_arch is not None:
        kwargs["net_arch"] = net_arch
    env = DummyVecEnv([lambda: _DimProbeEnv(int(target_dim), action_dim=int(action_dim))])
    return PPO(MaskedActorCriticPolicy, env, device="cpu", verbose=0, n_steps=8, batch_size=8,
               policy_kwargs=kwargs), env


def _migrate_ppo_713_to_715(ppp_path: str, target_dim: int | None = None):
    """Старое имя миграции (713->715). Сохранено для совместимости со скриптами."""
    return _migrate_checkpoint_dim(ppp_path, target_dim=target_dim)


def _looks_like_dim_mismatch(msg: str) -> bool:
    """Похоже ли сообщение об ошибке на несовпадение размерности obs при загрузке весов."""
    m = (msg or "").lower()
    for marker in ("size mismatch", "loading state_dict", "shapes cannot be multiplied",
                   "mat1 and mat2", "copying a param with shape"):
        if marker in m:
            return True
    return "size" in m and "shape" in m


def explain_missing_checkpoint(path: str) -> str:
    """Человеческая подсказка, если файл снапшота не найден (частый случай: забыли .zip)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    try:
        files = sorted(os.listdir(d))
    except Exception:
        files = []
    hints = [f for f in files if base.lower().split(".")[0] in f.lower()][:8]
    lines = [f"Файл снапшота не найден: {path}"]
    if hints:
        lines.append("Похожие файлы рядом: " + ", ".join(os.path.join(d, h) for h in hints))
    elif files:
        lines.append(f"В каталоге {d} есть: " + ", ".join(files[:8]))
    lines.append("Укажите путь целиком (обычно с расширением .zip): "
                 "--resume models/self_play_qualified_19.zip")
    return "\n".join(lines)


PI_LAYERS = (512, 256)
VF_LAYERS = (512, 256)


def arch_usage_metrics(ppo) -> dict:
    """Насколько политика реально использует новые признаки (последние DAMAGE_BLOCK_SIZE).

    Пока сеть не «включила» новые входы, отношение норм весов близко к 0 — по этому числу
    видно, надо ли расширять экстрактор или поднимать lr.
    """
    try:
        from agents.damage import DAMAGE_BLOCK_SIZE

        w = ppo.policy.features_extractor.net[0].weight.detach()
        if w.dim() != 2 or w.shape[1] <= DAMAGE_BLOCK_SIZE:
            return {}
        n_new = int(DAMAGE_BLOCK_SIZE)
        w_new, w_old = w[:, -n_new:], w[:, :-n_new]
        # RMS, а не норма Фробениуса: норма растёт как sqrt(числа элементов) и сравнивать
        # 87 столбцов с 715 напрямую нельзя. RMS-отношение = 1, когда блок урона выучен
        # наравне с остальными признаками, и ~0 сразу после warm start (нулевые веса).
        rms_new = float(w_new.pow(2).mean().sqrt())
        rms_old = float(w_old.pow(2).mean().sqrt())
        abs_new = float(w_new.abs().mean())
        abs_old = float(w_old.abs().mean())
        return {
            "arch/feat_dim": int(w.shape[0]),
            "arch/obs_dim": int(w.shape[1]),
            "arch/new_cols_rms_ratio": rms_new / max(rms_old, 1e-12),
            "arch/new_cols_absmean_ratio": abs_new / max(abs_old, 1e-12),
            "arch/new_cols_rms": rms_new,
        }
    except Exception:
        return {}


def resolve_checkpoint_path(path: str) -> str:
    """Дополняет путь до существующего файла: `models/x` -> `models/x.zip`, если так есть.

    Пользователь часто пишет --resume models/self_play_qualified_19 без расширения;
    SB3 сам расширение не добавляет, поэтому падало FileNotFoundError.
    """
    if not path:
        return path
    if os.path.isfile(path):
        return path
    for suffix in (".zip", ".pkl"):
        cand = path + suffix
        if os.path.isfile(cand):
            print(f"  {path}: файла нет, использую {cand}")
            return cand
    return path


def _checkpoint_obs_dim(path: str):
    """Размерность obs из метаданных чекпоинта (без загрузки весов). None, если не прочитать."""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(path, device="cpu")
        space = data["observation_space"]
        try:
            return int(space["observation"].shape[0])
        except Exception:
            return int(np.prod(getattr(space, "shape", [0]))) or None
    except Exception:
        return None


def _checkpoint_arch(path: str) -> dict:
    """Архитектура из весов чекпоинта: obs_dim, features_dim, net_arch.

    Читаем из весов, а не из policy_kwargs: у старых снапшотов policy_kwargs пустой
    (класс подставлял 512 сам), поэтому метаданным доверять нельзя.
    """
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        _, params, _ = load_from_zip_file(path, device="cpu")
    except Exception:
        return {}
    state = None
    # ВАЖНО: у старых снапшотов (identity-экстрактор LegacyFeaturesExtractor) в state_dict
    # вообще нет ключей features_extractor — раньше такой чекпоинт не опознавался (arch={}),
    # проба собиралась с текущими дефолтами и загрузка падала на size mismatch.
    for _, v in params.items():
        if not isinstance(v, dict):
            continue
        keys = list(v.keys())
        if any(("features_extractor" in k or "mlp_extractor" in k or k.endswith("action_net.weight"))
               for k in keys):
            state = v
            break
    if state is None:
        return {}
    info = {}
    pi_layers = []       # (индекс слоя, размер выхода) — порядок берём по номеру, а не по dict
    vf_layers = []
    shared_layers = []
    for k, tensor in state.items():
        if not hasattr(tensor, "shape"):
            continue
        if k.endswith("features_extractor.net.0.weight") and tensor.dim() == 2:
            info["features_dim"] = int(tensor.shape[0])
            info["obs_dim"] = int(tensor.shape[1])
            info["has_feature_net"] = True
        elif k.endswith("action_net.weight") and tensor.dim() == 2:
            # сколько действий у чекпоинта (gen9 = 26); нужно, чтобы fallback-миграция
            # не собрала политику с чужой головой
            info["action_dim"] = int(tensor.shape[0])
        elif "mlp_extractor.policy_net." in k and k.endswith(".weight") and tensor.dim() == 2:
            try:
                layer_idx = int(k.split("policy_net.")[1].split(".")[0])
            except Exception:
                layer_idx = len(pi_layers)
            pi_layers.append((layer_idx, int(tensor.shape[0])))
        elif "mlp_extractor.value_net." in k and k.endswith(".weight") and tensor.dim() == 2:
            try:
                layer_idx = int(k.split("value_net.")[1].split(".")[0])
            except Exception:
                layer_idx = len(vf_layers)
            vf_layers.append((layer_idx, int(tensor.shape[0])))
        elif "mlp_extractor.shared_net." in k and k.endswith(".weight") and tensor.dim() == 2:
            try:
                layer_idx = int(k.split("shared_net.")[1].split(".")[0])
            except Exception:
                layer_idx = len(shared_layers)
            shared_layers.append((layer_idx, int(tensor.shape[0])))
    info["has_feature_net"] = bool(info.get("has_feature_net", False))
    pi = [h for _, h in sorted(pi_layers)]
    vf = [h for _, h in sorted(vf_layers)]
    shared = [h for _, h in sorted(shared_layers)]
    if pi and vf:
        info["net_arch"] = {"pi": pi, "vf": vf}
        info["heads"] = "split"
    elif shared:
        # старая общая архитектура (net_arch списком): головы identity, последний слой — общий
        info["net_arch"] = {"pi": shared, "vf": shared}
        info["heads"] = "shared"
    elif pi:
        info["net_arch"] = {"pi": pi, "vf": None}
    return info


def _migrate_checkpoint_dim(ppp_path: str, target_dim: int | None = None, force_fallback: bool = False,
                            target_features_dim: int | None = None, verbose: bool = True):
    """Миграция чекпоинта на актуальный N_FEATURES через паддинг весов в zip.

    Обобщено: раньше умела ровно 713->715, теперь определяет старую размерность из самих
    весов и добивает до target_dim (по умолчанию N_FEATURES из config). Новые признаки
    получают нулевые веса — модель стартует с того же поведения, что и раньше, и доучивает
    вклад новых признаков (warm start вместо обучения с нуля).
    """
    import torch
    import tempfile
    import os
    arch = _checkpoint_arch(ppp_path)
    OLD_N = arch.get("obs_dim") if arch else _checkpoint_obs_dim(ppp_path)
    OLD_F = int(arch.get("features_dim") or 512)
    NEW_N = int(target_dim or N_FEATURES)
    NEW_F = int(target_features_dim or OLD_F)
    if OLD_N is not None and OLD_N == NEW_N and OLD_F == NEW_F:
        print(f"  {ppp_path}: уже {NEW_N} признаков и features_dim={NEW_F} — миграция не нужна")
        from stable_baselines3 import PPO as _PPO
        return _PPO.load(ppp_path, device="cpu")
    def _say(msg, *a):
        if verbose:
            print(msg if not a else msg % a)

    _say(f"  Миграция {ppp_path}: obs {OLD_N}->{NEW_N}, features_dim {OLD_F}->{NEW_F}"
         if OLD_F != NEW_F else f"  Миграция {ppp_path}: obs {OLD_N}->{NEW_N} (features_dim {NEW_F} без изменений)")
    try:
        if force_fallback:
            raise RuntimeError("force_fallback: основная миграция пропущена по запросу")
        from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
        from gymnasium.spaces import Box, Dict
        data, params, pytorch_variables = load_from_zip_file(ppp_path, device=torch.device("cpu"))
        # params: dict like {"policy": OrderedDict, "policy.optimizer": ...} или {"policy": state_dict}
        # находим policy state dict
        policy_state = None
        policy_key = None
        # SB3 stores params as dict {name: state_dict}
        for k, v in list(params.items()):
            if isinstance(v, dict) and any("features_extractor" in kk for kk in v.keys()):
                policy_state = v
                policy_key = k
                break
            # также может быть flat: params itself contains weights directly? тогда policy_state = params
        if policy_state is None:
            # fallback: assume params itself is policy state dict
            if any("features_extractor" in k for k in params.keys()):
                policy_state = params
                policy_key = None
            else:
                # try first dict value
                for k, v in params.items():
                    if isinstance(v, dict):
                        policy_state = v
                        policy_key = k
                        break
        if policy_state is None:
            raise RuntimeError(f"Не нашёл policy state dict в {list(params.keys())[:5]}")

        padded = 0

        def _pad_2d(tensor, new_rows=None, new_cols=None):
            """Паддинг 2D-тензора нулями справа/снизу (или обрезка, если цель меньше)."""
            r = int(new_rows if new_rows is not None else tensor.shape[0])
            c = int(new_cols if new_cols is not None else tensor.shape[1])
            out = torch.zeros((r, c), dtype=tensor.dtype, device=tensor.device)
            out[:min(r, tensor.shape[0]), :min(c, tensor.shape[1])] = tensor[:r, :c]
            return out

        def _pad_1d(tensor, new_len, fill):
            out = torch.full((int(new_len),), float(fill), dtype=tensor.dtype, device=tensor.device)
            out[:min(int(new_len), tensor.shape[0])] = tensor[:int(new_len)]
            return out

        for key in list(policy_state.keys()):
            tensor = policy_state[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            # первый слой экстрактора признаков: [features_dim, obs_dim]
            if "features_extractor" in key and key.endswith(".net.0.weight") and tensor.dim() == 2:
                if tensor.shape[0] != NEW_F or tensor.shape[1] != NEW_N:
                    policy_state[key] = _pad_2d(tensor, NEW_F, NEW_N)
                    padded += 1
                    _say(f"    паддинг {key} {list(tensor.shape)} -> {list(policy_state[key].shape)} "
                         f"(новые признаки и нейроны входят с нулевыми весами)")
            # сдвиг/масштаб LayerNorm после первого слоя
            elif "features_extractor" in key and key.endswith(".net.1.weight") and tensor.dim() == 1:
                if tensor.shape[0] != NEW_F:
                    policy_state[key] = _pad_1d(tensor, NEW_F, 1.0)   # новые нейроны: weight=1
                    padded += 1
                    _say(f"    паддинг {key} {list(tensor.shape)} -> {list(policy_state[key].shape)} (новые = 1)")
            elif "features_extractor" in key and key.endswith(".net.1.bias") and tensor.dim() == 1:
                if tensor.shape[0] != NEW_F:
                    policy_state[key] = _pad_1d(tensor, NEW_F, 0.0)   # новые нейроны: bias=0
                    padded += 1
            elif "features_extractor" in key and key.endswith(".net.0.bias") and tensor.dim() == 1:
                if tensor.shape[0] != NEW_F:
                    policy_state[key] = _pad_1d(tensor, NEW_F, 0.0)
                    padded += 1
            # вход pi/vf-головы: [hidden, features_dim]
            elif ("mlp_extractor.policy_net." in key or "mlp_extractor.value_net." in key) \
                    and key.endswith(".0.weight") and tensor.dim() == 2 and tensor.shape[1] != NEW_F:
                policy_state[key] = _pad_2d(tensor, None, NEW_F)
                padded += 1
                _say(f"    паддинг {key} {list(tensor.shape)} -> {list(policy_state[key].shape)} "
                     f"(новые нейроны экстрактора входят с нулевыми весами)")
        if padded == 0:
            _say(f"    WARN: не нашёл весов для паддинга (obs {OLD_N}->{NEW_N}, features {OLD_F}->{NEW_F})")

        # обновляем policy_kwargs: иначе SB3 соберёт политику с дефолтным features_dim=512
        try:
            pk = dict(data.get("policy_kwargs") or {})
            fek = dict(pk.get("features_extractor_kwargs") or {})
            fek["features_dim"] = int(NEW_F)
            pk["features_extractor_kwargs"] = fek
            pk.setdefault("net_arch", dict(arch.get("net_arch") or {"pi": [512, 256], "vf": [512, 256]}))
            if pk.get("net_arch", {}).get("vf") is None:
                pk["net_arch"] = dict(pk["net_arch"])
                pk["net_arch"]["vf"] = list(pk["net_arch"]["pi"])
            data["policy_kwargs"] = pk
            _say(f"    policy_kwargs: features_dim={NEW_F}, net_arch={pk['net_arch']}")
        except Exception as e:
            _say(f"    не удалось обновить policy_kwargs: {e}")

        # обновляем observation_space в data если есть
        try:
            if isinstance(data, dict) and "observation_space" in data:
                obs_space = data["observation_space"]
                if hasattr(obs_space, "spaces") and "observation" in obs_space.spaces:
                    old_shape = obs_space.spaces["observation"].shape
                    if old_shape != (NEW_N,):
                        am_space = obs_space.spaces.get("action_mask", Box(0, 1, shape=(9,), dtype=bool))
                        new_obs_space = Dict({"observation": Box(-1, 4, shape=(NEW_N,), dtype="float32"), "action_mask": am_space})
                        data["observation_space"] = new_obs_space
                        _say(f"    обновил data['observation_space'] {old_shape} -> {(NEW_N,)}")
        except Exception as e:
            _say(f"    не удалось обновить observation_space в data: {e}")

        # сбрасываем optimizer state чтобы не тянуть 713 моменты
        if pytorch_variables is not None:
            _say(f"    сбрасываю optimizer state ({OLD_N}->{NEW_N}) — будет новый оптимизатор")
            pytorch_variables = None

        # сохраняем пропатченный чекпоинт во временный файл и грузим как обычный PPO
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_to_zip_file(tmp_path, data=data, params=params, pytorch_variables=pytorch_variables)
            _say(f"  Сохраняю пропатченный чекпоинт во временный файл {tmp_path}")
            ppo_new = PPO.load(tmp_path, device="cpu")
            _say(f"  Успешно загрузил мигрированный PPO")
            # критично: сбрасываем Adam моменты старой размерности — иначе exp_avg old vs grad new -> RuntimeError
            try:
                if hasattr(ppo_new, "policy") and hasattr(ppo_new.policy, "optimizer") and ppo_new.policy.optimizer is not None:
                    ppo_new.policy.optimizer.state.clear()
                    _say(f"    сбросил optimizer.state ({OLD_N}->{NEW_N})")
            except Exception as _e:
                print(f"    не удалось сбросить optimizer.state: {_e}")
            try:
                # на всякий: SB3 иногда хранит optimizer в ppo.optimizer
                if hasattr(ppo_new, "optimizer") and ppo_new.optimizer is not None:
                    ppo_new.optimizer.state.clear()
            except Exception:
                pass
            return ppo_new
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        # старая известная причина: в zip лежит lr_schedule-функция, снятая ДРУГОЙ версией
        # Python — cloudpickle не может её разобрать при пересохранении ("tuple index out of
        # range"). Это не повод терять снапшот: идём через сборку политики по весам.
        if "tuple index out of range" in str(e) or "cloudpickle" in str(e) or isinstance(e, IndexError):
            print(f"  Штатная миграция невозможна ({msg}: в чекпоинте lr_schedule от другой версии "
                  f"Python), собираю политику по весам")
        else:
            print(f"  load_from_zip_file миграция не удалась: {msg}, пробую fallback через создание нового PPO и копирование весов")
            import traceback
            traceback.print_exc()
        # fallback: собираем политику нужной размерности (без сервера) и копируем веса вручную
        try:
            # размер головы берём из самого чекпоинта (или из его action_space), по умолчанию gen9 = 26
            try:
                # локальный импорт: при force_fallback Primary-ветка до импортов не доходит
                from stable_baselines3.common.save_util import load_from_zip_file as _lf_dim
                _ck_data, _, _ = _lf_dim(ppp_path, device=torch.device("cpu"))
                act_dim = int(arch.get("action_dim") or getattr(_ck_data.get("action_space"), "n", 0) or 0)
            except Exception:
                act_dim = 0
            if act_dim <= 0:
                act_dim = 26
            legacy_extractor = not bool(arch.get("has_feature_net"))
            ck_net_arch = arch.get("net_arch") or None
            # у legacy identity-экстрактора выход = N_FEATURES (а не features_dim)
            mlp_in_target = int(NEW_N) if legacy_extractor else int(NEW_F)
            _say(f"  Fallback: probe-политика строится под action_dim={act_dim}, "
                 f"net_arch={ck_net_arch}, экстрактор={'legacy identity' if legacy_extractor else 'Linear'} "
                 f"(вход mlp {mlp_in_target})")
            ppo_new, dummy_env = _probe_ppo(NEW_N, NEW_F, action_dim=act_dim,
                                            net_arch=ck_net_arch, legacy_extractor=legacy_extractor)
            # грузим старый state dict через load_from_zip_file снова но теперь паддим и грузим напрямую
            try:
                from stable_baselines3.common.save_util import load_from_zip_file as _lf
                data2, params2, _ = _lf(ppp_path, device=torch.device("cpu"))
                # найдём policy state снова
                for k, v in params2.items():
                    if isinstance(v, dict) and any("weight" in kk for kk in v.keys()):
                        policy_state2 = v
                        break
                else:
                    policy_state2 = params2
                # паддинг
                for kk in list(policy_state2.keys()):
                    tt = policy_state2[kk]
                    # размерности берём из самих тензоров: метаданные могут быть пустыми
                    if (isinstance(tt, torch.Tensor) and tt.dim() == 2
                            and "features_extractor" in kk and kk.endswith(".net.0.weight")):
                        if tt.shape[0] != NEW_F or tt.shape[1] != NEW_N:
                            nt = torch.zeros((NEW_F, NEW_N), dtype=tt.dtype, device=tt.device)
                            nt[:tt.shape[0], :tt.shape[1]] = tt
                            policy_state2[kk] = nt
                            _say(f"  Fallback: паддинг {kk} {list(tt.shape)} -> {list(nt.shape)}")
                    elif (isinstance(tt, torch.Tensor) and tt.dim() == 1
                            and "features_extractor" in kk and kk.endswith(".net.1.weight")
                            and tt.shape[0] != NEW_F):
                        nt = torch.ones((NEW_F,), dtype=tt.dtype, device=tt.device)
                        nt[:tt.shape[0]] = tt
                        policy_state2[kk] = nt
                    elif (isinstance(tt, torch.Tensor) and tt.dim() == 1
                            and "features_extractor" in kk and (kk.endswith(".net.1.bias") or kk.endswith(".net.0.bias"))
                            and tt.shape[0] != NEW_F):
                        nt = torch.zeros((NEW_F,), dtype=tt.dtype, device=tt.device)
                        nt[:tt.shape[0]] = tt
                        policy_state2[kk] = nt
                    elif (isinstance(tt, torch.Tensor) and tt.dim() == 2
                            and ("mlp_extractor.policy_net." in kk or "mlp_extractor.value_net." in kk
                                 or "mlp_extractor.shared_net." in kk)
                            and kk.endswith(".0.weight") and tt.shape[1] != mlp_in_target):
                        nt = torch.zeros((tt.shape[0], mlp_in_target), dtype=tt.dtype, device=tt.device)
                        nt[:, :tt.shape[1]] = tt
                        policy_state2[kk] = nt
                        _say(f"  Fallback: паддинг {kk} {list(tt.shape)} -> {list(nt.shape)}")
                # загружаем в ppo_new
                try:
                    _load_res = ppo_new.policy.load_state_dict(policy_state2, strict=False)
                    _say("  Fallback: загрузил падденный state_dict напрямую в новый PPO (strict=False)")
                    # strict=False молча пропускает веса с несовпавшим shape -> голова/экстрактор
                    # остались бы случайными. Такое молчание уже один раз стоило обученной
                    # политики, поэтому печатаем громко и явно.
                    _missing = list(getattr(_load_res, "missing_keys", []) or [])
                    _unexpected = list(getattr(_load_res, "unexpected_keys", []) or [])
                    _critical = [k for k in _missing
                                 if ("action_net" in k or "features_extractor.net.0" in k or "mlp_extractor" in k)]
                    if _critical:
                        print(f"  ВНИМАНИЕ: {len(_critical)} критичных весов НЕ загружены (остались случайными): "
                              f"{_critical[:4]}{' ...' if len(_critical) > 4 else ''}")
                    if _missing:
                        print(f"  Fallback: не загружено ключей: {len(_missing)} (первые: {_missing[:3]})")
                    if _unexpected:
                        print(f"  Fallback: лишних ключей в чекпоинте: {len(_unexpected)} (первые: {_unexpected[:3]})")
                    # правим observation_space
                    from gymnasium.spaces import Box, Dict
                    try:
                        am_space = ppo_new.observation_space.spaces["action_mask"]
                    except Exception:
                        am_space = Box(0,1,shape=(9,), dtype=bool)
                    new_obs_space = Dict({"observation": Box(-1,4,shape=(NEW_N,), dtype="float32"), "action_mask": am_space})
                    ppo_new.observation_space = new_obs_space
                    ppo_new.policy.observation_space = new_obs_space
                    dummy_env.close()
                    return ppo_new
                except Exception as ne:
                    print(f"  Fallback load_state_dict упал: {ne}")
                    raise
            except Exception as ne2:
                raise ne2
        except Exception as e2:
            print(f"  Fallback тоже упал: {e2}")
            raise e

# флаги CLI: управляют судьбой статистики нормализации при resume (см. vecnorm_utils.stats_verdict)
RESET_OBS_STATS = False   # --reset-obs-stats: сбросить статистику даже если размерности совпали
KEEP_OBS_STATS = False    # --keep-obs-stats: оставить даже несовместимую (старое поведение)


def _migrate_vecnormalize_713_to_715(vec_path: str, base_env):
    """Грузит VecNormalize и проверяет, что статистика относится к ТЕКУЩЕЙ раскладке признаков.

    Раньше умела ТОЛЬКО 713 -> 715 и просто добивала статистику нулями до нужной размерности
    (418, 713 -> 870). Это неверно: раскладка признаков менялась, в т.ч. вставками в середину
    (713 -> 715), поэтому старые колонки — уже другие признаки, а `count` ~200k не даёт
    испорченной статистике вымыться. Теперь несовместимая статистика сбрасывается
    (mean=0, var=1, count=0) и переоценивается по текущим данным (BC warm-up / первые роллауты).
    """
    try:
        target_dim = None
        try:
            target_dim = base_env.observation_spaces[base_env.possible_agents[0]].shape[0]
        except Exception:
            try:
                target_dim = base_env.observation_space["observation"].shape[0]
            except Exception:
                target_dim = None
        from .vecnorm_utils import load_vecnormalize_for_dim
        return load_vecnormalize_for_dim(vec_path, base_env, target_dim,
                                         force_reset=RESET_OBS_STATS, keep_stale=KEEP_OBS_STATS)
    except Exception as e:
        msg = str(e)
        if "713" in msg or "715" in msg or "418" in msg or "shape" in msg.lower():
            print(f"  VecNormalize.load упал ({e}), создаю новый VecNormalize (статистика сброшена)")
            return VecNormalize(base_env, norm_obs=True, norm_reward=False, gamma=0.99, norm_obs_keys=["observation"])
        raise

def _save_vecnorm(env, path: str = None) -> None:
    """Сохраняет VecNormalize вместе с сайдкаром (размерность + хеш кода признаков).

    Без сайдкара следующий resume не может отличить статистику текущей раскладки от
    «добитой нулями» старой — и молча уезжает в неверную нормализацию.
    """
    from .config import VECNORM_PATH as _VP
    from .vecnorm_utils import save_vecnormalize_with_meta
    path = path or _VP
    try:
        if not save_vecnormalize_with_meta(env, path):
            print(f"Не удалось сохранить VecNormalize в {path}")
    except Exception as e:
        print(f"Не удалось сохранить VecNormalize: {e}")



def run(
    resume_from: str | None = None,
    total_timesteps: int = 2_000_000,
    num_envs: int = 8,
    phase_size: int = 200_000,
    norm_reward: bool = False,
    pretrain_battles: int = 0,
    ent_coef: float | None = None,
    epochs: int = 5,
    dataset_path: str | None = None,
    force_recollect: bool = False,
    no_normalize_bc: bool = False,
    learning_rate: float = 2e-4,
    contrastive: bool = False,
    neg_weight: float = 0.3,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 128,
    features_dim: int = 512,
    vf_coef: float = 0.5,
    bc_value_coef: float = 0.0,
    value_warmup_steps: int = 0,
    min_winrate: int = 25,
    reset_schedules: bool = False,
    eval_battles: int = 20,
    skip_eval: bool = False,
    # --- ICM Variant B ---
    icm: bool = False,
    icm_beta: float = 0.05,
    icm_anneal: bool = True,
    icm_lr: float = 3e-4,
    icm_feat_dim: int = 256,
    icm_train_freq: int = 2048,
    icm_batch_size: int = 128,
    reset_obs_stats: bool = False,
    keep_obs_stats: bool = False,
):
    # судьба статистики нормализации при resume (см. vecnorm_utils.stats_verdict)
    global RESET_OBS_STATS, KEEP_OBS_STATS
    RESET_OBS_STATS = bool(reset_obs_stats)
    KEEP_OBS_STATS = bool(keep_obs_stats)
    # self-play оппоненты в env-процессах: нормализовать obs тем же способом, что и обучение
    os.environ["PYBOT_SELF_PLAY_NORM"] = "0" if no_normalize_bc else "1"
    if not no_normalize_bc:
        try:
            # логируем один раз в главном процессе (в воркерах печать подавляется)
            from agents.env import _self_play_obs_normalizer
            if _self_play_obs_normalizer() is None:
                print("self-play оппоненты: статистика нормализации obs не найдена — "
                      "играют на сырых признаках (это разойдётся с обучением)")
        except Exception as e:
            print(f"self-play: не удалось проверить нормализацию obs: {e}")

    # phase_size должен делиться на фактический размер роллаута n_steps*num_envs (с учётом целочисленного деления)
    rollout_size = (3072 // num_envs) * num_envs
    if phase_size % rollout_size != 0:
        print(f"WARNING: phase_size {phase_size} не кратен фактическому размеру роллаута {rollout_size} (n_steps {3072 // num_envs} * num_envs {num_envs}). Будет обрезка последнего роллаута.")

    run_name = f"{'retrain' if resume_from else 'train'}_{time.strftime('%Y%m%d_%H%M%S')}"
    icm_wrapper = None
    icm_module = None

    if resume_from:
        # размерность чекпоинта против текущего N_FEATURES: если разошлась — паддинг весов
        resume_from = resolve_checkpoint_path(resume_from)
        if not os.path.isfile(resume_from):
            raise SystemExit(explain_missing_checkpoint(resume_from))
        _arch = _checkpoint_arch(resume_from)
        _resume_dim = _arch.get("obs_dim") if _arch else _checkpoint_obs_dim(resume_from)
        _resume_feat = int(_arch.get("features_dim") or 512)
        if (_resume_dim is not None and _resume_dim != N_FEATURES) or _resume_feat != int(features_dim):
            print(f"Снапшот {resume_from}: obs {_resume_dim}, features_dim {_resume_feat} — "
                  f"мигрирую на obs {N_FEATURES}, features_dim {features_dim} "
                  f"(паддинг весов + сброс optimizer state)...")
            ppo = _migrate_checkpoint_dim(resume_from, target_dim=N_FEATURES,
                                          target_features_dim=int(features_dim))
        else:
            try:
                ppo = PPO.load(resume_from, device="cpu")
            except RuntimeError as e:
                if _looks_like_dim_mismatch(str(e)):
                    print(f"Несовпадение размеров при загрузке {resume_from} — мигрирую...")
                    ppo = _migrate_checkpoint_dim(resume_from, target_dim=N_FEATURES)
                else:
                    raise
            except Exception as e:
                # SB3 иногда оборачивает RuntimeError
                if _looks_like_dim_mismatch(str(e)):
                    print(f"Несовпадение размеров при загрузке {resume_from} — мигрирую...")
                    ppo = _migrate_checkpoint_dim(resume_from, target_dim=N_FEATURES)
                else:
                    raise
        # стартовое состояние признаков урона: после миграции новые веса ровно нулевые,
        # полезно видеть это до обучения, а не только в конце первой фазы
        try:
            _m0 = arch_usage_metrics(ppo)
            if _m0:
                print(f"[arch] старт: features_dim={_m0['arch/feat_dim']}, использование новых "
                      f"признаков урона (RMS весов, доля от старых) = "
                      f"{_m0['arch/new_cols_rms_ratio']:.4f}")
                if float(_m0.get("arch/new_cols_rms_ratio", 1.0)) < 0.02:
                    print("       веса новых признаков нулевые (warm start): они включатся по мере "
                          "обучения; смотри метрику [arch] между фазами")
        except Exception:
            pass
        if reset_schedules:
            print(f"Сбрасываю счетчики lr/ent: было {ppo.num_timesteps} шагов -> 0 (модель {resume_from})")
            steps_done_holder = {"value": 0}
            # сбрасываем и внутренний счетчик PPO чтобы логи и TB не прыгали
            ppo.num_timesteps = 0
        else:
            steps_done_holder = {"value": ppo.num_timesteps}
        if ent_coef is not None:
            print(f"Переопределяю ent_coef: {ppo.ent_coef} -> {ent_coef}")
            ppo.ent_coef = ent_coef
        # прокидываем остальные рекомендательные гиперпараметры и на resume
        from stable_baselines3.common.utils import get_schedule_fn
        for attr, val in [("clip_range", clip_range), ("n_epochs", n_epochs), ("batch_size", batch_size), ("vf_coef", vf_coef)]:
            if hasattr(ppo, attr):
                old = getattr(ppo, attr)
                # clip_range в SB3 - это schedule-функция, сравниваем по значению при progress=1.0
                old_val = old(1.0) if attr == "clip_range" and callable(old) else old
                if old_val != val:
                    print(f"Переопределяю {attr}: {old_val} -> {val}")
                    if attr == "clip_range":
                        setattr(ppo, attr, get_schedule_fn(val))
                    else:
                        setattr(ppo, attr, val)
        # создаём env заново; если был VecNormalize — загружаем (с миграцией 713->715 если нужно)
        base_env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        if not no_normalize_bc:
            if os.path.isfile(VECNORM_PATH):
                env = _migrate_vecnormalize_713_to_715(VECNORM_PATH, base_env)
                print(f"Загрузил VecNormalize из {VECNORM_PATH}")
            else:
                env = VecNormalize(base_env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
        else:
            env = base_env
        # --- ICM Variant B (resume) ---
        if icm:
            if ICM is None or CuriosityVecWrapper is None:
                print("ICM не доступен (agents/curiosity.py не загружен), игнорирую --icm")
            else:
                try:
                    try:
                        action_dim = int(base_env.action_space.n)
                    except Exception:
                        try:
                            action_dim = int(env.action_space.n)
                        except Exception:
                            action_dim = 26
                    icm_module = ICM(obs_dim=N_FEATURES, action_dim=action_dim, feat_dim=int(icm_feat_dim)).to("cpu")
                    env = CuriosityVecWrapper(env, icm=icm_module, beta=float(icm_beta), anneal=bool(icm_anneal), total_timesteps=int(total_timesteps), lr=float(icm_lr), device="cpu", train_freq=int(icm_train_freq), batch_size=int(icm_batch_size))
                    icm_wrapper = env
                    print(f"ICM включён (resume): beta={icm_beta} anneal={icm_anneal} feat={icm_feat_dim} action_dim={action_dim} lr={icm_lr}")
                except Exception as e:
                    print(f"Не удалось включить ICM: {e}")
                    import traceback; traceback.print_exc()
                    icm_wrapper = None
    else:
        steps_done_holder = {"value": 0}
        base_env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        if not no_normalize_bc:
            env = VecNormalize(base_env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
        else:
            env = base_env
        # --- ICM Variant B (fresh) ---
        if icm:
            if ICM is None or CuriosityVecWrapper is None:
                print("ICM не доступен (agents/curiosity.py не загружен), игнорирую --icm")
            else:
                try:
                    try:
                        action_dim = int(base_env.action_space.n)
                    except Exception:
                        try:
                            action_dim = int(env.action_space.n)
                        except Exception:
                            action_dim = 26
                    icm_module = ICM(obs_dim=N_FEATURES, action_dim=action_dim, feat_dim=int(icm_feat_dim)).to("cpu")
                    env = CuriosityVecWrapper(env, icm=icm_module, beta=float(icm_beta), anneal=bool(icm_anneal), total_timesteps=int(total_timesteps), lr=float(icm_lr), device="cpu", train_freq=int(icm_train_freq), batch_size=int(icm_batch_size))
                    icm_wrapper = env
                    print(f"ICM включён (fresh): beta={icm_beta} anneal={icm_anneal} feat={icm_feat_dim} action_dim={action_dim} lr={icm_lr}")
                except Exception as e:
                    print(f"Не удалось включить ICM: {e}")
                    import traceback; traceback.print_exc()
                    icm_wrapper = None
        ppo = PPO(
            MaskedActorCriticPolicy,
            env,
            ent_coef=ent_coef if ent_coef is not None else 0.01,
            learning_rate=learning_rate,
            n_steps=3072 // num_envs,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            vf_coef=vf_coef,
            device="cpu",
            tensorboard_log="./tb_logs/",
            # фиксируем размер экстрактора и голов в чекпоинте: иначе policy_kwargs пуст
            # и миграция не знает features_dim, поэтому не может расширять сеть
            policy_kwargs=dict(
                features_extractor_kwargs=dict(features_dim=int(features_dim)),
                net_arch=dict(pi=list(PI_LAYERS), vf=list(VF_LAYERS)),
            ),
        )
        # BC: либо собираем с нуля (pretrain_battles>0), либо грузим готовый только если пользователь ЯВНО указал --dataset-path
        # dataset_path по умолчанию None — чтобы наличие models/heuristic_dataset.npz от прошлого прогона не включало BC неожиданно
        need_bc = False
        dataset = None
        if pretrain_battles > 0:
            actual_path = dataset_path or "models/heuristic_dataset.npz"
            print(f"Собираю датасет на {pretrain_battles} боях SimpleHeuristicsPlayer -> {actual_path}...")
            dataset = collect_or_load_dataset(
                n_battles=pretrain_battles, path=actual_path, force_recollect=force_recollect
            )
            need_bc = True
        elif dataset_path and os.path.isfile(dataset_path):
            # пользователь явно указал готовый датасет (replay/heuristic) — грузим даже при pretrain_battles=0
            from agents.training import load_dataset as _load_ds
            try:
                dataset = _load_ds(dataset_path)
                need_bc = True
                print(f"Загружен готовый датасет {dataset_path} ({len(dataset)} семплов) для BC")
            except Exception as e:
                print(f"Не удалось загрузить {dataset_path}: {e}")
        if need_bc and dataset is not None:
            print(f"Претрейн через behavioral cloning (epochs={epochs}, contrastive={contrastive}, neg_weight={neg_weight}, bc_value_coef={bc_value_coef})...")
            pretrain_policy_bc(ppo, dataset, epochs=epochs, normalize=not no_normalize_bc, contrastive=contrastive, neg_weight=neg_weight, value_coef=bc_value_coef)

    # FIX: SB3 хранит расписание в ppo.lr_schedule (FloatSchedule), а не в learning_rate.
    # Раньше делали ppo.lr_schedule = schedule без обёртки или ppo.learning_rate = schedule —
    # в обоих случаях _update_learning_rate читал старое значение.
    # При total_timesteps==0 (только BC) расписания нет — оставляем константу, деления на 0 быть не должно.
    from stable_baselines3.common.utils import FloatSchedule
    lr_schedule_fn = make_lr_schedule(learning_rate, total_timesteps, steps_done_holder)
    ppo.lr_schedule = FloatSchedule(lr_schedule_fn)
    ppo.learning_rate = lr_schedule_fn  # для совместимости/логов
    if total_timesteps and total_timesteps > 0:
        try:
            print(f"LR schedule установлен: {lr_schedule_fn(1.0):.2e} -> {lr_schedule_fn(0.0):.2e} за {total_timesteps} шагов (initial {learning_rate:.2e})")
        except ZeroDivisionError:
            print(f"LR schedule установлен: {learning_rate:.2e} (константа, total_timesteps={total_timesteps})")
    else:
        print(f"RL пропущен (total_timesteps={total_timesteps}), LR зафиксирован: {learning_rate:.2e}")
    # если resume — сразу применим текущий LR к оптимизатору
    try:
        ppo._update_learning_rate(ppo.policy.optimizer)
    except Exception:
        pass

    # FIX: ent_coef тоже должен аннилиться, иначе агент быстро детерминизируется и застревает 30-40%
    ent_schedule = make_ent_schedule(total_timesteps, steps_done_holder)
    # Ставим callback который будет обновлять ppo.ent_coef каждые num_envs шагов
    # Передаём ссылку на ppo, чтобы callback менял поле вживую
    # Если ent_coef был переопределён вручную — всё равно аннилим от него?
    # Пусть ent_schedule стартует от текущего ent_coef, но проще использовать schedule как есть (0.05->0.001)
    # Чтобы не ломать ручной ent_coef, если он задан — используем его как начальный и не трогаем schedule
    use_ent_schedule = ent_coef is None  # если пользователь явно задал ent_coef — не аннилим
    if use_ent_schedule:
        counter_callback = StepCounterCallback(steps_done_holder, num_envs, ent_schedule=ent_schedule, ppo_ref=ppo)
        print("Включён ent_coef annealing 0.01->0.001 (мягкий, после BC не размывает)")
    else:
        counter_callback = StepCounterCallback(steps_done_holder, num_envs)
        print(f"ent_coef фиксирован: {ent_coef}")

    # VecNormalize training flag
    if hasattr(env, "training"):
        env.training = True
    if hasattr(env, "norm_reward"):
        env.norm_reward = norm_reward
    ppo.set_env(env)

    counter = _next_snapshot_index()
    # динамический порог на старте
    try:
        init_thr = _get_current_threshold(min_winrate)
        if init_thr != min_winrate:
            print(f"[init] qualified порог рэтчет: база {min_winrate}% -> текущий {init_thr}% (из { _QUALIFIED_META_PATH })")
        else:
            print(f"[init] qualified порог {init_thr}% (база {min_winrate}%)")
    except Exception as e:
        print(f"warn init threshold: {e}")
    # для адаптации opponent_weights: запомним последний словарь весов
    current_weights: dict[str, float] | None = None

    # опциональный прогрев value-сети после BC (лечит -3 -> -29 просадку из-за random value)
    warmup_remaining = value_warmup_steps
    if warmup_remaining and resume_from:
        print(f"Включен value warmup {warmup_remaining} шагов: замораживаю policy+extractor, учу только value")
        # Замораживаем ВСЁ кроме value-головы, иначе shared FeaturesExtractor поедет от value loss
        # и policy деградирует даже с замороженным action_net (было 31% -> 13% после 50k warmup)
        for p in ppo.policy.parameters():
            p.requires_grad = False
        # размораживаем только value-часть
        for p in ppo.policy.value_net.parameters():
            p.requires_grad = True
        try:
            for p in ppo.policy.mlp_extractor.value_net.parameters():
                p.requires_grad = True
        except Exception:
            pass
        # альтернативный путь для старых SB3 где mlp_extractor хранит value_net как список
        try:
            # на всякий: если extractor - Sequential, просто оставляем value_net выше
            pass
        except Exception:
            pass

    while steps_done_holder["value"] < total_timesteps:
        # если warmup еще не отработан - режем phase_size до warmup_remaining
        cur_phase = phase_size
        if warmup_remaining > 0:
            cur_phase = min(phase_size, warmup_remaining)
            print(f"[warmup] phase {counter} учим {cur_phase} шагов только value (осталось {warmup_remaining})")
        ppo.learn(cur_phase, callback=counter_callback, reset_num_timesteps=False, tb_log_name=run_name)
        if warmup_remaining > 0:
            warmup_remaining -= cur_phase
            if warmup_remaining <= 0:
                print("[warmup] размораживаю policy+extractor")
                for p in ppo.policy.parameters():
                    p.requires_grad = True
                # сбрасываем оптимизатор чтобы не тянуть моменты с warmup'а
                try:
                    ppo.policy.optimizer.state.clear()
                except Exception:
                    pass
                # также сбрасываем clip/n_epochs к нормальным после warmup если были занижены
                # оставляем как есть - пользователь уже задал 0.1/3

        ppo.save(f"{SELF_PLAY_PATH}_{counter}")

        # оценка — теперь с нормализацией (см. training.evaluate_win_rates)
        if skip_eval:
            print(f"[phase {counter}] skip eval (--skip-eval)")
            win_rates = {"SimpleHeuristicsPlayer": 0, "RandomPlayer": 0, "MaxBasePowerPlayer": 0, "self_play": 0}
        else:
            print(f"[phase {counter}] оценка {eval_battles} боев vs каждого бота (может занять 2-4 мин)...")
            try:
                win_rates = evaluate_win_rates(ppo, n_battles=eval_battles)
            except Exception as e:
                print(f"[phase {counter}] eval упал: {e}, ставлю 0")
                import traceback; traceback.print_exc()
                win_rates = {"SimpleHeuristicsPlayer": 0, "RandomPlayer": 0, "MaxBasePowerPlayer": 0, "self_play": 0}
        heuristics_rate = win_rates.get("SimpleHeuristicsPlayer", 0)
        # диагностика неизвестных типов (включается PYBOT_DEBUG_TYPES=1)
        try:
            try:
                from .type_utils import debug_enabled as _t_dbg, summary as _t_sum
            except ImportError:
                from type_utils import debug_enabled as _t_dbg, summary as _t_sum
            if _t_dbg():
                _s = _t_sum()
                if _s:
                    print(f"[phase {counter}] type-debug: {_s}")
        except Exception:
            pass
        # динамический рэтчет-порог: если уже пробивали выше — требуем не меньше прошлого максимума
        cur_threshold = _get_current_threshold(min_winrate)
        if cur_threshold != min_winrate:
            print(f"[phase {counter}] динамический порог {cur_threshold}% (база {min_winrate}%) — рэтчет с прошлого максимума")
        # иногда ключа нет если оценка упала — fallback
        if heuristics_rate >= cur_threshold:
            # дополнительно проверяем что файл не перезапишет существующий qualified
            save_path = f"models/{QUALIFIED_PREFIX}{counter}"
            ppo.save(save_path)
            print(f"[phase {counter}] снапшот прошёл порог ({heuristics_rate}% >= {cur_threshold}%) -> {save_path}")
            # мета-запись + рэтчет threshold
            _record_qualified(save_path, win_rates, min_winrate, counter)
        else:
            print(f"[phase {counter}] снапшот НЕ прошёл порог ({heuristics_rate}% < {cur_threshold}%) -> пропущен (база {min_winrate}%)")

        _update_opponent_weights(win_rates)
        # формируем веса для следующей фазы: ключи должны совпадать с тем, что ждёт env.create_env
        # env ждёт ["RandomPlayer","MaxBasePowerPlayer","SimpleHeuristicsPlayer","self_play"]
        # win_rates содержит первые три + возможно self_play
        # строим список имён в порядке ожидаемых категорий
        # Для консистентности берём юнион ключей win_rates + self_play
        all_names = list(win_rates.keys())
        # гарантируем наличие self_play категории даже если его не оценивали (будет дефолт 0.5)
        if "self_play" not in all_names:
            all_names.append("self_play")
        weights_list = _get_opponent_weights(all_names)
        current_weights = dict(zip(all_names, weights_list))
        print(f"[phase {counter}] opponent_weights { {k: round(v,3) for k,v in current_weights.items()} }")

        for name, rate in win_rates.items():
            ppo.logger.record(f"eval/winrate_{name}", rate)
        # также логируем ent_coef и lr для дебага
        try:
            ppo.logger.record("train/ent_coef", float(ppo.ent_coef))
            cur_lr = ppo.lr_schedule(ppo._current_progress_remaining) if hasattr(ppo, "lr_schedule") else 2e-4
            ppo.logger.record("train/learning_rate", float(cur_lr))
        except Exception:
            pass
        # ICM логи
        if icm and icm_wrapper is not None:
            try:
                ppo.logger.record("curiosity/beta", float(getattr(icm_wrapper, "beta", icm_beta)))
                ppo.logger.record("curiosity/r_int_mean", float(getattr(icm_wrapper, "last_r_int_mean", 0.0)))
                ppo.logger.record("curiosity/fwd_loss", float(getattr(icm_wrapper, "last_fwd_loss", 0.0)))
                ppo.logger.record("curiosity/inv_loss", float(getattr(icm_wrapper, "last_inv_loss", 0.0)))
                print(f"[curiosity] beta={getattr(icm_wrapper, 'beta', 0):.4f} r_int={getattr(icm_wrapper, 'last_r_int_mean', 0):.4f} fwd={getattr(icm_wrapper, 'last_fwd_loss', 0):.4f} inv={getattr(icm_wrapper, 'last_inv_loss', 0):.4f} replay={len(getattr(icm_wrapper, 'replay', []))}")
            except Exception:
                pass
        try:
            _arch_m = arch_usage_metrics(ppo)
            if _arch_m:
                for _k, _v in _arch_m.items():
                    ppo.logger.record(_k, _v)
                _ratio = _arch_m.get("arch/new_cols_rms_ratio", 1.0)
                _warn = ""
                if _ratio < 0.02:
                    _warn = ("  <- признаки урона почти не используются (веса ~0): "
                             "подними --learning-rate или --features-dim")
                elif _ratio < 0.3:
                    _warn = "  <- признаки урона включаются медленно"
                print(f"[arch] features_dim={_arch_m['arch/feat_dim']} использование новых"
                      f" признаков (RMS весов, доля от старых) = {_ratio:.4f}{_warn}")
        except Exception:
            pass
        ppo.logger.dump(steps_done_holder["value"])
        print(f"[phase {counter}] {win_rates}")

        counter += 1

        # FIX: раньше пересоздавали env только if not no_normalize_bc — из-за этого
        # при --no-normalize-bc self-play веса никогда не применялись (плато).
        # Теперь всегда пересоздаём, но ветвимся по нормализации.
        if not no_normalize_bc:
            # сохраняем статистику нормализации (+ сайдкар с отпечатком признаков)
            _save_vecnorm(env, VECNORM_PATH)
            env.close()
            raw_env = SubprocVecEnv(
                [partial(ExampleEnv.create_env, opponent_weights=current_weights) for _ in range(num_envs)]
            )
            try:
                env = _migrate_vecnormalize_713_to_715(VECNORM_PATH, raw_env)
            except Exception as e:
                print(f"Не удалось загрузить VecNormalize, создаю новый: {e}")
                env = VecNormalize(raw_env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
            env.training = True
            env.norm_reward = norm_reward
        else:
            env.close()
            raw_env = SubprocVecEnv(
                [partial(ExampleEnv.create_env, opponent_weights=current_weights) for _ in range(num_envs)]
            )
            env = raw_env
        # --- ICM re-wrap для новой фазы (сохраняем тот же icm_module и optimizer) ---
        if icm and icm_wrapper is not None and ICM is not None and CuriosityVecWrapper is not None:
            try:
                try:
                    icm_module_reuse = icm_wrapper.icm
                    old_beta = float(getattr(icm_wrapper, "beta", icm_beta))
                    old_initial_beta = float(getattr(icm_wrapper, "initial_beta", icm_beta))
                    old_steps = int(getattr(icm_wrapper, "steps_done", 0))
                    old_replay = getattr(icm_wrapper, "replay", None)
                    old_opt_state = None
                    try:
                        old_opt_state = icm_wrapper.optimizer.state_dict()
                    except Exception:
                        old_opt_state = None
                except Exception:
                    icm_module_reuse = icm_module
                    old_beta = float(icm_beta)
                    old_initial_beta = float(icm_beta)
                    old_steps = 0
                    old_replay = None
                    old_opt_state = None
                try:
                    action_dim = int(raw_env.action_space.n)
                except Exception:
                    try:
                        action_dim = int(env.action_space.n)
                    except Exception:
                        action_dim = getattr(icm_module_reuse, "action_dim", 26) if icm_module_reuse is not None else 26
                if icm_module_reuse is None:
                    icm_module_reuse = ICM(obs_dim=N_FEATURES, action_dim=action_dim, feat_dim=int(icm_feat_dim)).to("cpu")
                    old_initial_beta = float(icm_beta)
                    old_beta = float(icm_beta)
                    old_opt_state = None
                # создаём новый wrapper с исходным initial_beta, чтобы аннилинг продолжался корректно 0.05->0.01
                env = CuriosityVecWrapper(env, icm=icm_module_reuse, beta=float(old_initial_beta), anneal=bool(icm_anneal), total_timesteps=int(total_timesteps), lr=float(icm_lr), device="cpu", train_freq=int(icm_train_freq), batch_size=int(icm_batch_size))
                try:
                    if old_replay is not None:
                        env.replay = old_replay
                    env.steps_done = int(old_steps)
                    # восстанавливаем текущий beta (иначе сбросится к initial)
                    env.beta = float(old_beta)
                    # сохраняем оптимизатор (моменты)
                    if old_opt_state is not None:
                        try:
                            env.optimizer.load_state_dict(old_opt_state)
                        except Exception as _oe:
                            print(f"ICM optimizer restore warn: {_oe}")
                    # пробрасываем последние логи
                    for k in ("last_r_int_mean", "last_fwd_loss", "last_inv_loss"):
                        try:
                            setattr(env, k, getattr(icm_wrapper, k, 0.0))
                        except Exception:
                            pass
                except Exception:
                    pass
                icm_wrapper = env
                icm_module = icm_module_reuse
                print(f"ICM перенесён на новый env (phase {counter+1}) beta={env.beta:.4f} steps={env.steps_done} replay={len(env.replay)}")
            except Exception as e:
                print(f"ICM re-wrap failed: {e}")
                import traceback; traceback.print_exc()
        ppo.set_env(env)

    ppo.save("models/ppo_policy_final")
    # сохраняем VecNormalize финальный
    if not no_normalize_bc and hasattr(env, "save"):
        _save_vecnorm(env, VECNORM_PATH)
    env.close()

    # финальная оценка — тоже с нормализацией (патчим PolicyPlayer внутри evaluate, но тут просто выводим)
    # создаём агента с нормализацией вручную
    from agents.training import evaluate_win_rates as eval2
    # быстрый прогон без нормализации патча — evaluate уже патчит
    # для финального вывода используем evaluate
    if skip_eval:
        final_rates = {"skipped": 0}
    else:
        print(f"Финальная оценка {eval_battles} боев...")
        final_rates = evaluate_win_rates(ppo, n_battles=eval_battles)
    print("--- Final win rates (eval) ---")
    for k, v in final_rates.items():
        print(f"{k}: {v}%")

    # также старый способ для совместимости (сырой, без нормализации) — покажем оба
    agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    opponents = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))
    print("--- Win rates vs bots (raw, без VecNormalize) ---")
    for opp in opponents:
        if opp.n_finished_battles:
            win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
            print(f"{opp.username} ({opp.__class__.__name__}): {win_rate}% ({opp.n_finished_battles} battles)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--phase-size", type=int, default=200_000)
    parser.add_argument("--norm-reward", action="store_true", help="Нормализовать награды VecNormalize (рекомендуется для стабильности)")
    parser.add_argument("--no-normalize-bc", action="store_true")
    parser.add_argument("--reset-obs-stats", action="store_true", dest="reset_obs_stats",
                        help="сбросить статистику нормализации obs даже если размерность совпала "
                             "(полезно, если модель обучена на статистике старой раскладки признаков)")
    parser.add_argument("--keep-obs-stats", action="store_true", dest="keep_obs_stats",
                        help="оставить статистику нормализации даже при несовместимой размерности "
                             "(прежнее поведение: добить нулями)")
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4, dest="learning_rate", help="Начальный learning_rate (линейно аннилится до 0). По умолчанию 2e-4, для дообучения после BC рекомендуется 5e-5..1e-4")
    parser.add_argument("--lr", type=float, default=None, dest="lr_alias", help="Алиас для --learning-rate")
    parser.add_argument("--pretrain-battles", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset-path", type=str, default=None, help="Путь к готовому датасету для BC; если указан и файл существует — BC включится даже без --pretrain-battles. По умолчанию None, чтобы старый models/heuristic_dataset.npz не включал BC неявно")
    parser.add_argument("--force-recollect", action="store_true", help="Пересобрать датасет заново, игнорируя кэш")
    parser.add_argument("--contrastive", action="store_true", help="Контрастивный BC: отталкиваться от ходов проигравшего (w=-neg_weight)")
    parser.add_argument("--neg-weight", type=float, default=0.3, help="Вес лузер-ходов при --contrastive (0.3 слабее, 1.0 симметрично)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip_range (0.2 по умолчанию, после BC ставить 0.1 чтобы не снести BC)")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO n_epochs на один роллаут (10 по умолчанию, после BC ставить 3)")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO batch_size (128 по умолчанию, после BC ставить 256)")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="PPO vf_coef вес value loss (0.5 по умолчанию)")
    parser.add_argument("--bc-value-coef", type=float, default=0.0, help="BC value_coef вес value loss при претреине (0.0 только policy, 0.5 учит и value)")
    parser.add_argument("--value-warmup-steps", type=int, default=0, help="Сколько шагов после resume учить только value (заморозить policy) чтобы вылечить просадку -3->-29. Рекомендую 50000")
    parser.add_argument("--min-winrate", type=int, default=25, help="Порог %% vs Heuristics для сохранения qualified снапшота в self-play (было 50 -> 25, + fallback на обычные снапшоты когда нет qualified)")
    parser.add_argument("--reset-schedules", action="store_true", help="Сбросить счетчик шагов для lr/ent расписаний при resume (lr 3e-5 снова с начала, ent 0.01). Нужно когда берешь фазу 300k и хочешь доучивать как с нуля)")
    parser.add_argument("--eval-battles", type=int, default=20, help="Сколько боев на каждого бота в оценке между фазами (было 60 -> 20, 60*4=240 боев виснет на 5-10 мин)")
    parser.add_argument("--skip-eval", action="store_true", help="Пропустить оценку winrate между фазами (самый быстрый, если виснет на 60 боев)")
    parser.add_argument("--features-dim", type=int, default=512, help="Размер выхода экстрактора признаков (по умолчанию 512). При смене веса старого чекпоинта паддятся, новые нейроны входят с нулевыми весами (warm start); 640 стоит пробовать, если признаки урона не включаются")
    # --- ICM Variant B ---
    parser.add_argument("--icm", action="store_true", help="Включить Intrinsic Curiosity Module (Variant B) r = r_ext + beta*r_int")
    parser.add_argument("--icm-beta", type=float, default=0.05, help="Вес intrinsic награды (0.05 -> 0.01 с anneal)")
    parser.add_argument("--icm-anneal", dest="icm_anneal", action="store_true", help="Аннилить beta 0.05->0.01 (по умолчанию вкл)")
    parser.add_argument("--no-icm-anneal", dest="icm_anneal", action="store_false", help="Не аннилить beta")
    parser.set_defaults(icm_anneal=True)
    parser.add_argument("--icm-lr", type=float, default=3e-4, help="LR для ICM")
    parser.add_argument("--icm-feat-dim", type=int, default=256, help="Размер фич ICM encoder")
    parser.add_argument("--icm-train-freq", type=int, default=2048, help="Как часто тренировать ICM (шагов)")
    parser.add_argument("--icm-batch-size", type=int, default=128, help="Batch для ICM")
    args = parser.parse_args()

    # поддержка алиаса --lr
    if args.lr_alias is not None:
        args.learning_rate = args.lr_alias

    run(
        resume_from=args.resume,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        phase_size=args.phase_size,
        norm_reward=args.norm_reward,
        pretrain_battles=args.pretrain_battles,
        ent_coef=args.ent_coef,
        epochs=args.epochs,
        dataset_path=args.dataset_path,
        force_recollect=args.force_recollect,
        no_normalize_bc=args.no_normalize_bc,
        learning_rate=args.learning_rate,
        contrastive=args.contrastive,
        neg_weight=args.neg_weight,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        vf_coef=args.vf_coef,
        bc_value_coef=args.bc_value_coef,
        value_warmup_steps=args.value_warmup_steps,
        min_winrate=args.min_winrate,
        reset_schedules=args.reset_schedules,
        eval_battles=args.eval_battles,
        skip_eval=args.skip_eval,
        features_dim=args.features_dim,
        icm=args.icm,
        icm_beta=args.icm_beta,
        icm_anneal=args.icm_anneal,
        icm_lr=args.icm_lr,
        icm_feat_dim=args.icm_feat_dim,
        icm_train_freq=args.icm_train_freq,
        icm_batch_size=args.icm_batch_size,
        reset_obs_stats=args.reset_obs_stats,
        keep_obs_stats=args.keep_obs_stats,
    )
