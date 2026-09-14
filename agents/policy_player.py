import argparse
import os
import time
from functools import partial

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from agents.training import collect_or_load_dataset 
from agents import config  # было: from . import config
from agents.config import BATTLE_FORMAT, MIN_WINRATE_TO_QUALIFY, QUALIFIED_PREFIX, SELF_PLAY_PATH, VECNORM_PATH
from agents.env import ExampleEnv
from agents.policy import MaskedActorCriticPolicy
from agents.players import PolicyPlayer
from agents.training import (
    LRSchedule,
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


def run(
    resume_from: str | None = None,
    total_timesteps: int = 2_000_000,
    num_envs: int = 8,
    phase_size: int = 200_000,
    norm_reward: bool = False,
    pretrain_battles: int = 0,
    ent_coef: float | None = None,
    epochs: int = 5,
    dataset_path: str = "models/heuristic_dataset.npz",
    force_recollect: bool = False,
    no_normalize_bc:bool = False
):
    run_name = f"{'retrain' if resume_from else 'train'}_{time.strftime('%Y%m%d_%H%M%S')}"

    if resume_from:
        ppo = PPO.load(resume_from, device="cpu")
        steps_done_holder = {"value": ppo.num_timesteps}
        if ent_coef is not None:
            print(f"Переопределяю ent_coef: {ppo.ent_coef} -> {ent_coef}")
            ppo.ent_coef = ent_coef
        env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        if no_normalize_bc is False:
            if os.path.isfile(VECNORM_PATH):
                env = VecNormalize.load(VECNORM_PATH, env)
            else:
                env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
    else:
        steps_done_holder = {"value": 0}
        env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        if no_normalize_bc is False:
            env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
        ppo = PPO(
            MaskedActorCriticPolicy,
            env,
            ent_coef=ent_coef if ent_coef is not None else 0.01,
            learning_rate=2e-4,
            n_steps=3072 // num_envs,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cpu",
            tensorboard_log="./tb_logs/",
        )
        if pretrain_battles > 0:
            print(f"Собираю датасет на {pretrain_battles} боях SimpleHeuristicsPlayer...")
            dataset = collect_or_load_dataset(
                n_battles=pretrain_battles, path=dataset_path, force_recollect=force_recollect
            )
            print("Претрейн через behavioral cloning...")
            pretrain_policy_bc(ppo, dataset, epochs=epochs,normalize=not no_normalize_bc)
    counter_callback  = StepCounterCallback(steps_done_holder,num_envs)
    schedule = make_lr_schedule(2e-4, total_timesteps, steps_done_holder)
    ppo.lr_schedule = schedule
    env.training = True
    env.norm_reward = norm_reward
    ppo.set_env(env)

    counter = _next_snapshot_index()
    current_weights = None

    while steps_done_holder["value"] < total_timesteps:
        ppo.learn(phase_size, callback=counter_callback, reset_num_timesteps=False, tb_log_name=run_name)
        
        ppo.save(f"{SELF_PLAY_PATH}_{counter}")

        win_rates = evaluate_win_rates(ppo, n_battles=60)
        heuristics_rate = win_rates.get("SimpleHeuristicsPlayer", 0)
        if heuristics_rate >= MIN_WINRATE_TO_QUALIFY:
            ppo.save(f"models/{QUALIFIED_PREFIX}{counter}")
            print(f"[phase {counter}] снапшот прошёл порог ({heuristics_rate}%) -> добавлен в self-play пул")
        else:
            print(f"[phase {counter}] снапшот НЕ прошёл порог ({heuristics_rate}% < {MIN_WINRATE_TO_QUALIFY}%) -> пропущен")

        _update_opponent_weights(win_rates)
        current_weights = dict(zip(win_rates.keys(), _get_opponent_weights(list(win_rates.keys()))))

        for name, rate in win_rates.items():
            ppo.logger.record(f"eval/winrate_{name}", rate)
        ppo.logger.dump(steps_done_holder["value"])
        print(f"[phase {counter}] {win_rates}")

        counter += 1
        if no_normalize_bc is False:
            env.save(VECNORM_PATH)
            env.close()
            raw_env = SubprocVecEnv(
                [partial(ExampleEnv.create_env, opponent_weights=current_weights) for _ in range(num_envs)]
            )
            env = VecNormalize.load(VECNORM_PATH, raw_env)
            env.training = True
            env.norm_reward = norm_reward
            ppo.set_env(env)
            
    ppo.save("models/ppo_policy_final")
    env.close()

    agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    opponents = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))
    print("--- Win rates vs bots ---")
    for opp in opponents:
        win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
        print(f"{opp.username}: {win_rate}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--phase-size", type=int, default=200_000)
    parser.add_argument("--norm-reward", action="store_true")
    parser.add_argument("--no-normalize-bc", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--pretrain-battles", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset-path", type=str, default="models/heuristic_dataset.npz")
    parser.add_argument("--force-recollect", action="store_true", help="Пересобрать датасет заново, игнорируя кэш")
    args = parser.parse_args()

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
        no_normalize_bc=args.no_normalize_bc
    )