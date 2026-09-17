"""Intrinsic Curiosity Module (ICM) для Variant B.

Архитектура:
- encoder: 715 -> feat_dim (256) — отдельный от policy FeaturesExtractor, чтобы не мешать PPO
- inverse: [phi(s), phi(s')] -> logits(a)  (9-26 действий, без маски — CE)
- forward: [phi(s), one_hot(a)] -> phi(s')  (MSE)

Награда: r_int = || pred_phi_next - phi_next ||^2  (per-env, detach)
Итог: r_total = r_ext + beta * r_int   (beta anneal 0.05->0.01)

Интеграция: CuriosityVecWrapper(VecNormalize(SubprocVecEnv)) — считает r_int в step_wait,
добавляет к r_ext, копит переходы в replay и раз в N шагов делает SGD шаг по ICM.
PPO видит уже суммарную награду, VecNormalize нормализует только r_ext (т.к. wrapper снаружи).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env import VecNormalize


class ICM(nn.Module):
    def __init__(self, obs_dim: int = 715, action_dim: int = 26, feat_dim: int = 256, hidden: int = 256):
        super().__init__()
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
        # per-sample MSE for intrinsic reward
        fwd_mse_per = (pred_phi_next - phi_next.detach()).pow(2).mean(dim=-1)  # [B]
        fwd_loss = fwd_mse_per.mean()
        # для логов
        return inv_loss, fwd_loss, fwd_mse_per.detach()

    def intrinsic_reward(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray, device="cpu"):
        """Быстрый r_int без градиента, для VecEnv step. obs: [N,715], actions: [N], next_obs: [N,715] -> [N]"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(actions, dtype=torch.long, device=device)
            # clamp actions to valid range (на случай если action_space 26 а env даёт 9, или наоборот)
            act_t = torch.clamp(act_t, 0, self.action_dim - 1)
            phi = self.encode(obs_t)
            phi_next = self.encode(next_t)
            a_onehot = F.one_hot(act_t, num_classes=self.action_dim).float().to(device)
            pred = self.forward_model(torch.cat([phi, a_onehot], dim=-1))
            r = (pred - phi_next).pow(2).mean(dim=-1).cpu().numpy()
        self.train(was_training)
        return r


class CuriosityVecWrapper(VecEnvWrapper):
    """VecEnvWrapper который добавляет r_int к r_ext и тренирует ICM.

    Оборачивает VecNormalize(SubprocVecEnv) СНАРУЖИ, поэтому VecNormalize нормализует только r_ext,
    а r_int добавляется уже после нормализации (стабильнее).

    Хранит last_obs для вычисления перехода, replay для обучения ICM.
    """

    def __init__(self, venv: VecEnv, icm: ICM, beta: float = 0.05, anneal: bool = True, total_timesteps: int = 2_000_000, lr: float = 3e-4, device: str = "cpu", train_freq: int = 2048, batch_size: int = 128, replay_size: int = 10000):
        super().__init__(venv)
        self.icm = icm
        self.device = device
        self.initial_beta = float(beta)
        self.beta = float(beta)
        self.anneal = bool(anneal)
        self.total_timesteps = int(total_timesteps) if total_timesteps and total_timesteps > 0 else 2_000_000
        self.train_freq = int(train_freq)
        self.batch_size = int(batch_size)
        self.replay = deque(maxlen=replay_size)
        self.optimizer = torch.optim.Adam(self.icm.parameters(), lr=float(lr))
        self.steps_done = 0
        self.last_obs = None  # dict
        self._actions = None
        # для логирования
        self.last_inv_loss = 0.0
        self.last_fwd_loss = 0.0
        self.last_r_int_mean = 0.0

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
        """Достаёт np array [N,715] из VecEnv obs (может быть dict или ndarray)."""
        if isinstance(obs_dict, dict) and "observation" in obs_dict:
            return np.asarray(obs_dict["observation"], dtype=np.float32)
        # VecNormalize с Dict obs вернёт dict, SubprocVecEnv без VecNormalize вернёт dict тоже
        # на всякий: если уже ndarray
        if isinstance(obs_dict, np.ndarray):
            return np.asarray(obs_dict, dtype=np.float32)
        # fallback: пробуем достать из .observation
        try:
            return np.asarray(obs_dict["observation"], dtype=np.float32)
        except Exception:
            return np.asarray(obs_dict, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        self.last_obs = obs
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        try:
            if self.last_obs is not None and self._actions is not None:
                last_arr = self._get_obs_array(self.last_obs)
                next_arr = self._get_obs_array(obs)
                # done: заменяем reset-obs на истинный terminal_observation (иначе ICM учится на артефакте)
                try:
                    dones_arr = np.asarray(dones, dtype=bool)
                    if dones_arr.any():
                        for i, d in enumerate(dones_arr):
                            if d and i < len(infos) and isinstance(infos[i], dict):
                                term = infos[i].get("terminal_observation")
                                if term is not None:
                                    try:
                                        if isinstance(term, dict) and "observation" in term:
                                            next_arr[i] = np.asarray(term["observation"], dtype=np.float32)
                                        else:
                                            arr = np.asarray(term, dtype=np.float32)
                                            # иногда VecNormalize даёт terminal_observation как dict с ключами
                                            if arr.shape != next_arr[i].shape and arr.size == next_arr[i].size:
                                                arr = arr.reshape(next_arr[i].shape)
                                            next_arr[i] = arr
                                    except Exception:
                                        pass
                except Exception:
                    pass
                actions = self._actions
                if self.anneal and self.total_timesteps > 0:
                    progress = min(self.steps_done / self.total_timesteps, 1.0)
                    self.beta = self.initial_beta * (1 - progress) + 0.01 * progress
                r_int = self.icm.intrinsic_reward(last_arr, actions, next_arr, device=self.device)
                r_int = np.clip(r_int, 0, 5.0)
                self.last_r_int_mean = float(r_int.mean()) if r_int.size else 0.0
                rewards = np.asarray(rewards, dtype=np.float32) + self.beta * r_int.astype(np.float32)
                for i in range(len(actions)):
                    if not np.isfinite(last_arr[i]).all() or not np.isfinite(next_arr[i]).all():
                        continue
                    a = int(actions[i])
                    # страховка от несовпадения action_dim (26 vs 9)
                    a = int(np.clip(a, 0, self.icm.action_dim - 1))
                    self.replay.append((last_arr[i].copy(), a, next_arr[i].copy()))
                self.steps_done += len(actions)
                if len(self.replay) >= self.batch_size and self.steps_done % self.train_freq < len(actions):
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
        return obs, rewards, dones, infos

    def _train_icm_step(self):
        if len(self.replay) < self.batch_size:
            return
        try:
            idx = np.random.choice(len(self.replay), self.batch_size, replace=False)
            batch = [self.replay[i] for i in idx]
            obs_b = np.stack([b[0] for b in batch])
            act_b = np.array([b[1] for b in batch], dtype=np.int64)
            next_b = np.stack([b[2] for b in batch])
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
