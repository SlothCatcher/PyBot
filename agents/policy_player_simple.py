import asyncio
import os
import random
from typing import Any, Awaitable

import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from os import listdir
from os.path import isfile, join
from poke_env.battle import AbstractBattle, SideCondition, Status
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

BATTLE_FORMAT = "gen9fusionmonsrandombattle"
N_FEATURES = 38
_STATUSES = [None, Status.BRN, Status.PAR, Status.SLP, Status.FRZ, Status.PSN, Status.TOX]
def _status_one_hot(status) -> np.ndarray:
    vec = np.zeros(len(_STATUSES), dtype=np.float32)
    idx = _STATUSES.index(status) if status in _STATUSES else 0
    vec[idx] = 1.0
    return vec

SELF_PLAY_PATH = "models/self_play_snapshot"


def _make_self_play_opponents():
    onlyfiles = [f for f in listdir("models/") if isfile(join("models/", f)) and "self_play_snapshot_" in f]
    players = []
    try:        
        for i in onlyfiles:
            snap = PPO.load(onlyfiles[i], device="cpu")
            players.append(PolicyPlayer(
                policy=snap.policy, battle_format=BATTLE_FORMAT, start_listening=False
            ))
        
        return players
    except Exception:
        return players

def _hazards(side_conditions: dict) -> np.ndarray:
    return np.array(
        [
            1.0 if SideCondition.STEALTH_ROCK in side_conditions else 0.0,
            side_conditions.get(SideCondition.SPIKES, 0) / 3.0,
            side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0,
            1.0 if SideCondition.STICKY_WEB in side_conditions else 0.0,
        ],
        dtype=np.float32,
    )


def _switch_summary(team: dict) -> np.ndarray:
    """Aggregate info about the bench: avg HP fraction and fraction still alive."""
    reserves = [mon for mon in team.values() if not mon.active]
    if not reserves:
        return np.zeros(2, dtype=np.float32)
    alive = [mon for mon in reserves if not mon.fainted]
    avg_hp = float(np.mean([mon.current_hp_fraction for mon in alive])) if alive else 0.0
    alive_frac = len(alive) / max(len(reserves), 1)
    return np.array([avg_hp, alive_frac], dtype=np.float32)

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[128,128],
            features_extractor_class=FeaturesExtractor,
        )
    
    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = self._mask

        # Если для какого-то элемента батча маска пустая (0 валидных действий),
        # логиты после маскирования станут сплошным -inf -> NaN в Categorical.
        # Страховка: в этом случае временно разрешаем все действия.
        no_valid_action = mask.sum(dim=-1) == 0
        if no_valid_action.any():
            print(f"WARNING: empty action mask for {no_valid_action.sum().item()} batch element(s)")
            mask = mask.clone()
            mask[no_valid_action] = 1

        additive_mask = torch.where(mask == 1, 0.0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + additive_mask)


class FeaturesExtractor(BaseFeaturesExtractor):
    """Extracts the observation tensor from the dict obs and declares features_dim
    so SB3 builds the MLP with the right input size."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class PolicyPlayer(Player):
    policy: ActorCriticPolicy | None

    def __init__(
        self, policy: ActorCriticPolicy | None = None, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(
        self, battle: AbstractBattle
    ) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()
        obs = self.embed_battle(battle)
        obs 
        mask = np.array(SinglesEnv.get_action_mask(battle))
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(
                    obs, device=self.policy.device
                ).unsqueeze(0),
                "action_mask": torch.as_tensor(
                    mask, device=self.policy.device
                ).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict)
        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle):
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        type_chart = GenData.from_gen(battle.gen).type_chart
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                try:
                    moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=type_chart,
                    )
                except KeyError:
                    moves_dmg_multiplier[i] = 1.0

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        our_hp = (
            battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        )
        opp_hp = (
            battle.opponent_active_pokemon.current_hp_fraction
            if battle.opponent_active_pokemon
            else 0.0
        )

        our_status = _status_one_hot(
            battle.active_pokemon.status if battle.active_pokemon else None
        )
        opp_status = _status_one_hot(
            battle.opponent_active_pokemon.status
            if battle.opponent_active_pokemon
            else None
        )

        our_hazards = _hazards(battle.side_conditions)
        opp_hazards = _hazards(battle.opponent_side_conditions)

        our_switches = _switch_summary(battle.team)
        opp_switches = _switch_summary(battle.opponent_team)

        obs = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                [our_hp, opp_hp],
                our_status,
                opp_status,
                our_hazards,
                opp_hazards,
                our_switches,
                opp_switches,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs


class ExampleEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @classmethod
    def create_env(cls) -> Monitor:
        env = cls(battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None)
        candidates = [
            RandomPlayer(start_listening=False),
            MaxBasePowerPlayer(start_listening=False),
            SimpleHeuristicsPlayer(start_listening=False),
        ]
        self_play_opp = _make_self_play_opponents()
        candidates += self_play_opp
        opponent = random.choice(candidates)
        return Monitor(SingleAgentWrapper(env, opponent))

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        return PolicyPlayer.embed_battle(battle)


def train():
    num_envs = 2
    total_timesteps = 2_000_000
    phase_size = 200_000  # раз в столько шагов обновляем self-play снапшот

    env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=3072 // num_envs,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,
        device="cpu",
        tensorboard_log="./tb_logs/",
    )

    steps_done = 0
    counter = 0
    while steps_done < total_timesteps:
        ppo.learn(phase_size, reset_num_timesteps=False)
        steps_done += phase_size

        ppo.save(SELF_PLAY_PATH + "_" + str(counter))  # текущий снапшот -> оппонент для следующей фазы
        counter+=1
        env.close()
        env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        ppo.set_env(env)

    ppo.save("models/ppo_policy_final")
    env.close()

    # evaluate
    agent = PolicyPlayer(
        policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=10
    )
    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))
    print("--- Win rates vs bots ---")
    for opp in opponents:
        win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
        print(f"{opp.username}: {win_rate}%")


if __name__ == "__main__":
    train()