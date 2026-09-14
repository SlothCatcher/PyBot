import asyncio
import os
import numpy as np
import torch
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from .config import BATTLE_FORMAT
from .players import HeuristicRecorder, PolicyPlayer

class LRSchedule:
    def __call__(self, progress): return 3e-5
class ENTSchedule:
    def __call__(self, progress): return 0.03

class StepCounterCallback:
    def __init__(self, steps_holder: dict, num_envs: int):
        self.steps_holder = steps_holder
        self.num_envs = num_envs

    def __call__(self, _locals, _globals) -> bool:
        # Принимаем аргументы от SB3 и инкрементируем глобальный счетчик
        self.steps_holder["value"] += self.num_envs
        return True


def make_lr_schedule(initial_lr: float, total_timesteps: int, steps_holder: dict):
    # SB3 передает progress_remaining, но мы считаем прогресс по вашему глобальному счетчику
    def lr_schedule(progress_remaining: float) -> float:
        global_progress = max(1.0 - (steps_holder["value"] / total_timesteps), 0.0)
        return initial_lr * global_progress
    return lr_schedule

def make_ent_schedule(total_timesteps: int, steps_holder: dict):
    def ent_schedule(progress_remaining: float) -> float:
        # Считаем текущий прогресс от 0.0 (старт) до 1.0 (конец)
        current_step = steps_holder["value"]
        progress = min(current_step / total_timesteps, 1.0)
        
        # Фаза 1: Старт (0% - 20%)
        if progress <= 0.2:
            return 0.05
            
        # Фаза 2: Середина (20% - 70%)
        elif progress <= 0.7:
            # Нормализуем прогресс внутри этой фазы от 0.0 до 1.0
            phase_progress = (progress - 0.2) / (0.7 - 0.2)
            # Линейно интерполируем от 0.05 до 0.01
            return 0.05 - phase_progress * (0.05 - 0.01)
            
        # Фаза 3: Конец (70% - 100%)
        else:
            # Нормализуем прогресс внутри этой фазы от 0.0 до 1.0
            phase_progress = (progress - 0.7) / (1.0 - 0.7)
            # Линейно интерполируем от 0.01 до 0.001
            return 0.01 - phase_progress * (0.01 - 0.001)
            
    return ent_schedule


def save_dataset(dataset: list, path: str):
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    mask_arr = np.stack([d[1] for d in dataset]).astype(np.int8)
    action_arr = np.array([d[2] for d in dataset], dtype=np.int64)
    return_arr = np.array([d[3] for d in dataset], dtype=np.float32)
    np.savez_compressed(path, obs=obs_arr, mask=mask_arr, action=action_arr, ret=return_arr)
    print(f"Датасет сохранён: {path} ({len(dataset)} примеров)")


def load_dataset(path: str) -> list:
    data = np.load(path)
    dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
    print(f"Датасет загружен: {path} ({len(dataset)} примеров)")
    return dataset

def collect_or_load_dataset(n_battles: int, path: str, force_recollect: bool = False) -> list:
    if os.path.exists(path) and not force_recollect:
        return load_dataset(path)
    dataset = collect_heuristic_dataset(n_battles=n_battles)
    save_dataset(dataset, path)
    return dataset

def _next_snapshot_index() -> int:
    from os import listdir
    from .config import SELF_PLAY_PATH

    prefix = SELF_PLAY_PATH.split("/")[-1] + "_"
    existing = [f for f in listdir("models/") if prefix in f]
    nums = []
    for f in existing:
        suffix = f.split("_")[-1].split(".")[0]
        if suffix.isdigit():
            nums.append(int(suffix))
    return max(nums, default=-1) + 1

def _compute_bc_returns(raw_dataset: list, battles: dict, gamma: float = 0.99, victory_value: float = 30.0) -> list:
    from collections import defaultdict

    grouped = defaultdict(list)
    for obs, mask, action, tag in raw_dataset:
        grouped[tag].append((obs, mask, action))

    final = []
    for tag, transitions in grouped.items():
        battle = battles.get(tag)
        if battle is None or battle.won is None:
            continue  # не нашли исход — пропускаем, надёжный target важнее объёма
        outcome = victory_value if battle.won else -victory_value
        n = len(transitions)
        for i, (obs, mask, action) in enumerate(transitions):
            steps_remaining = n - i - 1
            ret = outcome * (gamma ** steps_remaining)
            final.append((obs, mask, action, ret))
    return final


def collect_heuristic_dataset(n_battles: int = 200) -> list:
    dataset: list = []
    recorder = HeuristicRecorder(dataset=dataset, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    asyncio.run(recorder.battle_against(opponent, n_battles=n_battles))

    battles = getattr(recorder, "battles", None) or getattr(recorder, "_battles", {})
    final_dataset = _compute_bc_returns(dataset, battles)
    print(f"Собрано {len(final_dataset)} примеров (с return) из {n_battles} боёв")
    return final_dataset

_win_rate_ema: dict[str, float] = {}
_EMA_ALPHA = 0.3
_MIN_WEIGHT = 0.10
_MAX_WEIGHT = 0.45

def warm_up_vec_normalize(vec_normalize, dataset):
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    vec_normalize.obs_rms["observation"].update(obs_arr)

def _update_opponent_weights(win_rates: dict[str, float]):
    for name, rate in win_rates.items():
        rate_frac = rate / 100.0
        prev = _win_rate_ema.get(name, rate_frac)
        _win_rate_ema[name] = _EMA_ALPHA * rate_frac + (1 - _EMA_ALPHA) * prev


def _get_opponent_weights(names: list[str]) -> list[float]:
    raw = [max(1.0 - _win_rate_ema.get(n, 0.5), 0.05) for n in names]
    total = sum(raw)
    weights = [w / total for w in raw]
    weights = [min(max(w, _MIN_WEIGHT), _MAX_WEIGHT) for w in weights]
    total2 = sum(weights)
    return [w / total2 for w in weights]


def evaluate_win_rates(ppo, n_battles: int = 180) -> dict[str, float]:
    agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=n_battles))
    rates = {}
    for opp in opponents:
        rates[opp.__class__.__name__] = round(100 * opp.n_lost_battles / opp.n_finished_battles)
    return rates





def pretrain_policy_bc(
    ppo: PPO, dataset: list, epochs: int = 50, batch_size: int = 256,
    normalize: bool = False, value_coef: float = 0.5, val_frac: float = 0.1,
    patience: int = 5,
):
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    mask_arr = np.stack([d[1] for d in dataset]).astype(np.float32)
    action_arr = np.array([d[2] for d in dataset], dtype=np.int64)
    return_arr = np.array([d[3] for d in dataset], dtype=np.float32)

    if normalize:
        vec_normalize = ppo.get_vec_normalize_env()
        if vec_normalize is not None:
            warm_up_vec_normalize(vec_normalize, dataset)
            obs_arr = vec_normalize.normalize_obs({"observation": obs_arr})["observation"]

    n = len(dataset)
    n_val = int(n * val_frac)
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = ppo.policy.device
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(epochs):
        # --- train ---
        train_perm = np.random.permutation(train_idx)
        total_policy_loss, total_value_loss, n_batches = 0.0, 0.0, 0
        for start in range(0, len(train_perm), batch_size):
            idx = train_perm[start:start + batch_size]
            obs_dict = {
                "observation": torch.as_tensor(obs_arr[idx], device=device),
                "action_mask": torch.as_tensor(mask_arr[idx], device=device),
            }
            action_batch = torch.as_tensor(action_arr[idx], device=device)
            return_batch = torch.as_tensor(return_arr[idx], device=device)

            features = ppo.policy.extract_features(obs_dict)
            latent_pi, latent_vf = ppo.policy.mlp_extractor(features)
            ppo.policy._mask = obs_dict["action_mask"]
            distribution = ppo.policy._get_action_dist_from_latent(latent_pi)
            policy_loss = -distribution.log_prob(action_batch).mean()
            values = ppo.policy.value_net(latent_vf).flatten()
            value_loss = torch.nn.functional.mse_loss(values, return_batch)
            loss = policy_loss + value_coef * value_loss

            ppo.policy.optimizer.zero_grad()
            loss.backward()
            ppo.policy.optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        # --- val (без backward) ---
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs_arr[val_idx], device=device),
                "action_mask": torch.as_tensor(mask_arr[val_idx], device=device),
            }
            action_batch = torch.as_tensor(action_arr[val_idx], device=device)
            return_batch = torch.as_tensor(return_arr[val_idx], device=device)
            features = ppo.policy.extract_features(obs_dict)
            latent_pi, latent_vf = ppo.policy.mlp_extractor(features)
            ppo.policy._mask = obs_dict["action_mask"]
            distribution = ppo.policy._get_action_dist_from_latent(latent_pi)
            val_policy_loss = -distribution.log_prob(action_batch).mean().item()
            values = ppo.policy.value_net(latent_vf).flatten()
            val_value_loss = torch.nn.functional.mse_loss(values, return_batch).item()
            val_loss = val_policy_loss + value_coef * val_value_loss

        print(
            f"[BC epoch {epoch}] train_policy={total_policy_loss/n_batches:.4f} "
            f"train_value={total_value_loss/n_batches:.4f} "
            f"val_policy={val_policy_loss:.4f} val_value={val_value_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in ppo.policy.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Ранняя остановка на эпохе {epoch} (val loss не улучшается {patience} эпох подряд)")
                break

    if best_state is not None:
        ppo.policy.load_state_dict(best_state)
        print("Восстановлены веса с лучшей val_loss")