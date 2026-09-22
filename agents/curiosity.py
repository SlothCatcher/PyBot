"""Intrinsic Curiosity Module (ICM) для Variant B.

Архитектура:
- encoder: N_FEATURES -> feat_dim (256) — отдельный от policy FeaturesExtractor, чтобы не мешать PPO
- inverse: [phi(s), phi(s')] -> logits(a)  (CE, без маски)
- forward: [phi(s), one_hot(a)] -> phi(s')  (MSE, phi_next.detach() — см. forward_loss)

Награда: r_int = || pred_phi_next - phi_next ||^2  (per-env, clip 0..5)
Итог: r_total = r_ext + beta * r_int   (beta anneal initial -> 0.01)

Интеграция: CuriosityVecWrapper(VecNormalize(SubprocVecEnv)) — считает r_int в step_wait,
добавляет к r_ext, копит переходы в replay и раз в train_freq шагов делает SGD шаг по ICM.
PPO видит уже суммарную награду, VecNormalize нормализует только r_ext (wrapper снаружи).

ВАЖНО про нормализацию наблюдений (дрейф статистики).
VecNormalize обновляет obs_rms на КАЖДОМ шаге, поэтому «нормализованные» наблюдения разных шагов
отмасштабированы разными статистиками: если просто складывать их в replay, старые записи будут
в старом масштабе, а forward-модель учится предсказывать phi(s') в текущем масштабе — тихий шум
в fwd_loss тем больше, чем длиннее прогон. Поэтому wrapper хранит в replay СЫРЫЕ наблюдения
(VecNormalize.get_original_obs()) и приводит их к ТЕКУЩЕМУ масштабу в момент обучения ICM
(normalize_replay=True). Тем же масштабом нормализуются s и s' при подсчёте r_int, так что
внутри одного перехода нет рассинхрона между «до» и «после».

Про terminal_observation: SB3 SubprocVecEnv/DummyVecEnv кладут в info["terminal_observation"]
СЫРОЕ последнее наблюдение эпизода (до reset), а VecNormalize.step_wait его нормализует
(vec_normalize.py: «Normalize the terminal observations»). Если ключа нет — next-наблюдение
принадлежит НОВОМУ эпизоду, и такой переход в replay не пишется (см. missing_terminal_obs).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env import VecNormalize

from .config import N_FEATURES


def _default_action_dim(gen: int = 9) -> int:
    """Число действий SinglesEnv для поколения (gen9 -> 26), из самой poke-env.

    Раньше здесь было жёстко 26, а obs_dim — жёстко 715: N_FEATURES в проекте менялся
    (715 -> 802 -> 870), и такие константы становятся скрытой точкой рассинхрона.
    Спрашиваем у библиотеки, а не у своей памяти о ней; при любой ошибке — 26.
    """
    try:
        from poke_env.environment.singles_env import SinglesEnv
        n = int(SinglesEnv.get_action_space_size(int(gen)))
        if n > 0:
            return n
    except Exception:
        pass
    return 26


class ICM(nn.Module):
    def __init__(self, obs_dim: int | None = None, action_dim: int | None = None,
                 feat_dim: int = 256, hidden: int = 256):
        super().__init__()
        # Размерности по умолчанию: obs — из config (единственный источник правды о признаках),
        # действий — из poke-env. Явный аргумент по-прежнему важнее.
        obs_dim = int(N_FEATURES if obs_dim is None else obs_dim)
        action_dim = int(_default_action_dim() if action_dim is None else action_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )
        # inverse: phi(s), phi(s') -> a
        self.inverse = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # forward: phi(s), a_onehot -> phi(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feat_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )
        # init orthogonal как в policy
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward_loss(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor):
        """Считает inv_loss, fwd_loss, r_int (per-sample). actions: LongTensor [B]"""
        actions = torch.clamp(actions, 0, self.action_dim - 1)
        phi = self.encode(obs)
        phi_next = self.encode(next_obs)
        # inverse
        inv_logits = self.inverse(torch.cat([phi, phi_next], dim=-1))
        inv_loss = F.cross_entropy(inv_logits, actions)
        # forward
        a_onehot = F.one_hot(actions, num_classes=self.action_dim).float().to(obs.device)
        pred_phi_next = self.forward_model(torch.cat([phi, a_onehot], dim=-1))
        # per-sample MSE for intrinsic reward.
        # phi_next.detach(): forward-модель не должна «учить» энкодер под себя через этот путь
        # (иначе тривиальный коллапс: энкодер подстраивается, чтобы предсказывать легче)
        fwd_mse_per = (pred_phi_next - phi_next.detach()).pow(2).mean(dim=-1)  # [B]
        fwd_loss = fwd_mse_per.mean()
        # для логов
        return inv_loss, fwd_loss, fwd_mse_per.detach()

    def intrinsic_reward(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray,
                         device: str | None = None):
        """Быстрый r_int без градиента, для VecEnv step.
        obs: [N, D], actions: [N], next_obs: [N, D] -> [N]

        device=None -> берём устройство параметров модуля (у nn.Module НЕТ атрибута .device,
        туда легко сослаться по ошибке: тензоры оказались бы на другом устройстве, чем веса).
        """
        if device is None:
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
        else:
            dev = device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
            act_t = torch.as_tensor(actions, dtype=torch.long, device=dev)
            # clamp actions to valid range (на случай если action_space 26 а env даёт 9, или наоборот)
            act_t = torch.clamp(act_t, 0, self.action_dim - 1)
            phi = self.encode(obs_t)
            phi_next = self.encode(next_t)
            a_onehot = F.one_hot(act_t, num_classes=self.action_dim).float().to(dev)
            pred = self.forward_model(torch.cat([phi, a_onehot], dim=-1))
            r = (pred - phi_next).pow(2).mean(dim=-1).cpu().numpy()
        self.train(was_training)
        return r


def _shaping_episode_cap() -> float:
    """SHAPING_EPISODE_CAP из env.py (для предупреждения о перевесе curiosity)."""
    try:
        from .env import SHAPING_EPISODE_CAP
        return float(SHAPING_EPISODE_CAP)
    except Exception:
        return 12.0


class CuriosityVecWrapper(VecEnvWrapper):
    """VecEnvWrapper который добавляет r_int к r_ext и тренирует ICM.

    Оборачивает VecNormalize(SubprocVecEnv) СНАРУЖИ, поэтому VecNormalize нормализует только r_ext,
    а r_int добавляется уже после нормализации (стабильнее).

    Хранит last_obs для вычисления перехода, replay для обучения ICM. В replay — СЫРЫЕ
    наблюдения (см. докстринг модуля про дрейф статистики нормализации).
    """

    def __init__(self, venv: VecEnv, icm: ICM, beta: float = 0.05, anneal: bool = True,
                 total_timesteps: int = 2_000_000, lr: float = 3e-4, device: str = "cpu",
                 train_freq: int = 2048, batch_size: int = 128, replay_size: int = 10000,
                 normalize_replay: bool = True):
        super().__init__(venv)
        # РЕАЛЬНЫЙ ФИКС: без .to(device) тензоры создаются на device, а веса остаются на CPU.
        # На cpu незаметно, на cuda — RuntimeError: Expected all tensors to be on the same device
        # на первом же вызове. Делаем ДО создания оптимизатора (иначе он бы ссылался на старые тензоры).
        self.icm = icm.to(device)
        self.device = device
        self.initial_beta = float(beta)
        self.beta = float(beta)
        self.anneal = bool(anneal)
        self.total_timesteps = int(total_timesteps) if total_timesteps and total_timesteps > 0 else 2_000_000
        self.train_freq = int(train_freq)
        self.batch_size = int(batch_size)
        self.replay = deque(maxlen=replay_size)
        self.normalize_replay = bool(normalize_replay)
        self.optimizer = torch.optim.Adam(self.icm.parameters(), lr=float(lr))
        self.steps_done = 0
        self.last_obs = None    # dict (нормализованный, как его видит PPO)
        self.last_raw_obs = None  # np.ndarray [N, D] — сырые наблюдения того же шага
        self._actions = None
        self._r_int_episode = np.zeros(int(getattr(venv, "num_envs", 1)), dtype=np.float64)
        # для логирования
        self.last_inv_loss = 0.0
        self.last_fwd_loss = 0.0
        self.last_r_int_mean = 0.0
        self.last_episode_r_int = 0.0      # curiosity-вклад (beta*r_int) за последний завершённый эпизод
        self.max_episode_r_int = 0.0
        self.missing_terminal_obs = 0      # done-шагов без terminal_observation (переход не пишем)
        self.episode_r_int_warn_at = _shaping_episode_cap()
        self._episode_r_int_warned = False
        self._missing_terminal_warned = False

    @property
    def training(self):
        inner = self.get_vec_normalize_env()
        if inner is not None and hasattr(inner, "training"):
            return inner.training
        return getattr(self.venv, "training", True)

    @training.setter
    def training(self, value):
        inner = self.get_vec_normalize_env()
        if inner is not None and hasattr(inner, "training"):
            inner.training = value
        else:
            try:
                self.venv.training = value
            except Exception:
                pass

    @property
    def norm_reward(self):
        inner = self.get_vec_normalize_env()
        if inner is not None and hasattr(inner, "norm_reward"):
            return inner.norm_reward
        return getattr(self.venv, "norm_reward", False)

    @norm_reward.setter
    def norm_reward(self, value):
        inner = self.get_vec_normalize_env()
        if inner is not None and hasattr(inner, "norm_reward"):
            inner.norm_reward = value
        else:
            try:
                self.venv.norm_reward = value
            except Exception:
                pass

    def _get_obs_array(self, obs_dict):
        """Достаёт np array [N, D] из VecEnv obs (может быть dict или ndarray)."""
        if isinstance(obs_dict, dict) and "observation" in obs_dict:
            return np.asarray(obs_dict["observation"], dtype=np.float32)
        if isinstance(obs_dict, np.ndarray):
            return np.asarray(obs_dict, dtype=np.float32)
        try:
            return np.asarray(obs_dict["observation"], dtype=np.float32)
        except Exception:
            return np.asarray(obs_dict, dtype=np.float32)

    def _raw_array(self):
        """СЫРЫЕ (до VecNormalize) наблюдения последнего шага/reset, если их можно достать."""
        vn = self.get_vec_normalize_env()
        if vn is None or not hasattr(vn, "get_original_obs"):
            return None
        try:
            raw = vn.get_original_obs()
        except Exception:
            return None
        if raw is None:
            return None
        try:
            arr = self._get_obs_array(raw)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        return arr

    def _raw_of_terminal(self, term):
        """terminal_observation приходит УЖЕ нормализованным (VecNormalize) — возвращаем сырой.

        ВАЖНО: при dict-наблюдениях obs_rms у VecNormalize — СЛОВАРЬ, и unnormalize_obs
        требует тот же dict (голый ndarray падает на assert isinstance(obs_rms, RunningMeanStd)).
        Поэтому передаём объект как есть (dict остаётся dict).
        """
        if term is None:
            return None
        vn = self.get_vec_normalize_env()
        obj = term
        if vn is not None and hasattr(vn, "unnormalize_obs"):
            try:
                obj = vn.unnormalize_obs(term)
            except Exception:
                return None
        try:
            arr = self._get_obs_array(obj)
        except Exception:
            return None
        return arr if arr.ndim == 1 else None

    def _normalize(self, arr):
        """Сырые наблюдения -> масштаб ТЕКУЩЕЙ статистики VecNormalize (None, если нельзя).

        При dict-наблюдениях obs_rms — словарь, поэтому голый ndarray передавать нельзя
        (`normalize_obs` уйдёт в ветку для RunningMeanStd и упадёт на assert). Оборачиваем
        в тот же dict, что ждёт VecNormalize, и достаём обратно.
        """
        vn = self.get_vec_normalize_env()
        if vn is None or not hasattr(vn, "normalize_obs"):
            return None
        arr = np.asarray(arr, dtype=np.float32)
        try:
            rms = getattr(vn, "obs_rms", None)
            if isinstance(rms, dict):
                out = vn.normalize_obs({"observation": arr.copy()})
                out = out["observation"] if isinstance(out, dict) else out
            else:
                out = vn.normalize_obs(arr.copy())
            out = np.asarray(out, dtype=np.float32)
        except Exception:
            return None
        return out if out.shape == arr.shape else None

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        self.last_obs = obs
        self.last_raw_obs = self._raw_array()
        self._r_int_episode = np.zeros(len(self._get_obs_array(obs)), dtype=np.float64)
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        raw_after = None
        try:
            if self.last_obs is not None and self._actions is not None:
                actions = np.asarray(self._actions).reshape(-1)
                n = len(actions)
                last_norm = self._get_obs_array(self.last_obs)
                next_norm = self._get_obs_array(obs)
                dones_arr = (np.asarray(dones, dtype=bool).reshape(-1) if dones is not None
                             else np.zeros(n, dtype=bool))

                # --- 1) переходы в СЫРОМ виде + терминальные наблюдения ---
                # raw_after — сырые obs, которые вернул env; у done-сред это уже НОВЫЙ эпизод,
                # поэтому для ICM/replay подменяем их истинным terminal_observation.
                raw_after = self._raw_array()
                next_raw = None if raw_after is None else raw_after.copy()
                invalid = np.zeros(n, dtype=bool)
                for i in np.nonzero(dones_arr)[0]:
                    i = int(i)
                    info = infos[i] if (infos is not None and i < len(infos)) else None
                    term = info.get("terminal_observation") if isinstance(info, dict) else None
                    raw_term = self._raw_of_terminal(term)
                    if raw_term is not None and next_raw is not None and raw_term.shape == next_raw[i].shape:
                        next_raw[i] = raw_term
                    else:
                        # terminal_observation нет (или нечем вернуть в сырой вид): next — это obs
                        # НОВОГО эпизода, то есть артефакт. Такой переход в replay не пишем и
                        # curiosity за него не начисляем.
                        invalid[i] = True
                        self.missing_terminal_obs += 1
                        if not self._missing_terminal_warned:
                            self._missing_terminal_warned = True
                            why = ("нет ключа terminal_observation" if term is None
                                   else "terminal_observation не удалось вернуть в сырой вид")
                            print(f"ICM: у done-шага {why} — переход не идёт в replay "
                                  f"(иначе ICM учился бы на наблюдении нового эпизода)")

                use_raw = (self.last_raw_obs is not None and next_raw is not None
                           and self.last_raw_obs.shape == last_norm.shape
                           and next_raw.shape == next_norm.shape)
                if use_raw:
                    if self.normalize_replay:
                        n_last, n_next = self._normalize(self.last_raw_obs), self._normalize(next_raw)
                        if n_last is None or n_next is None:
                            use_raw = False
                    else:
                        n_last, n_next = self.last_raw_obs, next_raw
                if not use_raw:
                    n_last, n_next = last_norm, next_norm

                if self.anneal and self.total_timesteps > 0:
                    progress = min(self.steps_done / self.total_timesteps, 1.0)
                    self.beta = self.initial_beta * (1 - progress) + 0.01 * progress

                r_int = self.icm.intrinsic_reward(n_last, actions, n_next, device=self.device)
                r_int = np.clip(np.asarray(r_int, dtype=np.float64).reshape(-1), 0, 5.0)
                if invalid.any():
                    r_int[invalid] = 0.0   # за артефактный переход не награждаем
                self.last_r_int_mean = float(r_int.mean()) if r_int.size else 0.0

                contrib = self.beta * r_int
                rewards = np.asarray(rewards, dtype=np.float32) + contrib.astype(np.float32)

                for i in range(n):
                    if invalid[i]:
                        continue
                    s = self.last_raw_obs[i] if use_raw else last_norm[i]
                    s2 = next_raw[i] if use_raw else next_norm[i]
                    if not (np.isfinite(s).all() and np.isfinite(s2).all()):
                        continue
                    a = int(np.clip(int(actions[i]), 0, self.icm.action_dim - 1))
                    self.replay.append((np.asarray(s, dtype=np.float32).copy(), a,
                                        np.asarray(s2, dtype=np.float32).copy(), bool(use_raw)))

                # curiosity-вклад за эпизод: сравнивать с SHAPING_EPISODE_CAP (иначе intrinsic
                # может перевесить терминальную награду, ради чего и вводился cap на shaping)
                if len(self._r_int_episode) != n:
                    self._r_int_episode = np.zeros(n, dtype=np.float64)
                self._r_int_episode += contrib
                for i in np.nonzero(dones_arr)[0]:
                    i = int(i)
                    ep = float(self._r_int_episode[i])
                    self._r_int_episode[i] = 0.0
                    self.last_episode_r_int = ep
                    self.max_episode_r_int = max(self.max_episode_r_int, ep)
                    if infos is not None and i < len(infos) and isinstance(infos[i], dict):
                        infos[i]["r_int_episode"] = ep
                    if ep >= self.episode_r_int_warn_at and not self._episode_r_int_warned:
                        self._episode_r_int_warned = True
                        print(f"ВНИМАНИЕ: curiosity-вклад за эпизод {ep:.2f} >= SHAPING_EPISODE_CAP "
                              f"({self.episode_r_int_warn_at:.1f}): intrinsic перевешивает плотный shaping. "
                              f"Уменьшите --icm-beta или --icm-feat-dim.")

                self.steps_done += n
                if len(self.replay) >= self.batch_size and self.steps_done % self.train_freq < n:
                    self._train_icm_step()
                for i, info in enumerate(infos):
                    if isinstance(info, dict):
                        info["r_int"] = float(r_int[i]) if i < len(r_int) else 0.0
                        info["beta"] = float(self.beta)
        except Exception as e:
            print(f"CuriosityVecWrapper step_wait warn: {e}")
            import traceback
            traceback.print_exc()
        self.last_obs = obs
        self.last_raw_obs = raw_after
        return obs, rewards, dones, infos

    def _train_icm_step(self):
        if len(self.replay) < self.batch_size:
            return
        try:
            idx = np.random.choice(len(self.replay), self.batch_size, replace=False)
            batch = [self.replay[int(i)] for i in idx]
            obs_b = np.stack([b[0] for b in batch])
            act_b = np.array([b[1] for b in batch], dtype=np.int64)
            next_b = np.stack([b[2] for b in batch])
            raw_rows = np.array([b[3] for b in batch], dtype=bool)

            # Сырые записи приводим к ТЕКУЩЕМУ масштабу: obs_rms дрейфует за время обучения,
            # и запись, сделанная 10k шагов назад, отмасштабирована иначе, чем текущие obs.
            if raw_rows.any():
                if self.normalize_replay:
                    n_obs = self._normalize(obs_b[raw_rows])
                    n_next = self._normalize(next_b[raw_rows])
                else:
                    n_obs = n_next = None
                if n_obs is not None and n_next is not None:
                    obs_b[raw_rows] = n_obs
                    next_b[raw_rows] = n_next
                else:
                    # нормализовать нечем — сырые и нормализованные мешать нельзя, убираем сырые
                    keep = ~raw_rows
                    if int(keep.sum()) < 2:
                        return
                    obs_b, act_b, next_b = obs_b[keep], act_b[keep], next_b[keep]

            obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(act_b, dtype=torch.long, device=self.device)
            next_t = torch.as_tensor(next_b, dtype=torch.float32, device=self.device)
            inv_loss, fwd_loss, _ = self.icm.forward_loss(obs_t, act_t, next_t)
            # классический вес ICM: 0.2*inv + 0.8*fwd, но делаем равный для простоты; можно тюнить
            loss = 0.2 * inv_loss + 0.8 * fwd_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
            self.optimizer.step()
            self.last_inv_loss = float(inv_loss.item())
            self.last_fwd_loss = float(fwd_loss.item())
        except Exception as e:
            print(f"ICM train step warn: {e}")

    def get_vec_normalize_env(self):
        # пробрасываем чтобы PPO.get_vec_normalize_env() нашёл VecNormalize внутри
        try:
            if hasattr(self.venv, "get_vec_normalize_env"):
                return self.venv.get_vec_normalize_env()
            if isinstance(self.venv, VecNormalize):
                return self.venv
        except Exception:
            pass
        return None

    def save(self, path):
        # делегируем VecNormalize.save если внутри
        try:
            inner = self.get_vec_normalize_env()
            if inner is not None and hasattr(inner, "save"):
                return inner.save(path)
        except Exception:
            pass
        return None
