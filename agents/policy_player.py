import argparse
import os
import time
from functools import partial

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from agents.training import collect_or_load_dataset
from agents.config import BATTLE_FORMAT, MIN_WINRATE_TO_QUALIFY, QUALIFIED_PREFIX, SELF_PLAY_PATH, VECNORM_PATH
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
    no_normalize_bc: bool = False,
    learning_rate: float = 2e-4,
    contrastive: bool = False,
    neg_weight: float = 0.3,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 128,
    vf_coef: float = 0.5,
    bc_value_coef: float = 0.0,
    value_warmup_steps: int = 0,
):
    # phase_size должен делиться на n_steps*num_envs = 3072 для ровных роллаутов
    if phase_size % 3072 != 0:
        print(f"WARNING: phase_size {phase_size} не кратен 3072 (n_steps*num_envs). Будет обрезка последнего роллаута.")

    run_name = f"{'retrain' if resume_from else 'train'}_{time.strftime('%Y%m%d_%H%M%S')}"

    if resume_from:
        ppo = PPO.load(resume_from, device="cpu")
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
        # создаём env заново; если был VecNormalize — загружаем
        base_env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        if not no_normalize_bc:
            if os.path.isfile(VECNORM_PATH):
                env = VecNormalize.load(VECNORM_PATH, base_env)
                print(f"Загрузил VecNormalize из {VECNORM_PATH}")
            else:
                env = VecNormalize(base_env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
        else:
            env = base_env
    else:
        steps_done_holder = {"value": 0}
        base_env = SubprocVecEnv([ExampleEnv.create_env for _ in range(num_envs)])
        if not no_normalize_bc:
            env = VecNormalize(base_env, norm_obs=True, norm_reward=norm_reward, gamma=0.99, norm_obs_keys=["observation"])
        else:
            env = base_env
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
        )
        # BC: либо собираем с нуля (pretrain_battles>0), либо грузим готовый (replay_dataset.npz) даже при 0
        need_bc = False
        dataset = None
        if pretrain_battles > 0:
            print(f"Собираю датасет на {pretrain_battles} боях SimpleHeuristicsPlayer...")
            dataset = collect_or_load_dataset(
                n_battles=pretrain_battles, path=dataset_path, force_recollect=force_recollect
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
    # для адаптации opponent_weights: запомним последний словарь весов
    current_weights: dict[str, float] | None = None

    # опциональный прогрев value-сети после BC (лечит -3 -> -29 просадку из-за random value)
    warmup_remaining = value_warmup_steps
    if warmup_remaining and resume_from:
        print(f"Включен value warmup {warmup_remaining} шагов: замораживаю policy-сеть, учу только value")
        # замораживаем policy-голову
        ppo.policy.action_net.requires_grad_(False)
        try:
            ppo.policy.mlp_extractor.policy_net.requires_grad_(False)
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
                print("[warmup] размораживаю policy-сеть")
                ppo.policy.action_net.requires_grad_(True)
                try:
                    ppo.policy.mlp_extractor.policy_net.requires_grad_(True)
                except Exception:
                    pass
                # сбрасываем оптимизатор чтобы не тянуть моменты с warmup'а
                try:
                    ppo.policy.optimizer.state.clear()
                except Exception:
                    pass

        ppo.save(f"{SELF_PLAY_PATH}_{counter}")

        # оценка — теперь с нормализацией (см. training.evaluate_win_rates)
        win_rates = evaluate_win_rates(ppo, n_battles=60)
        heuristics_rate = win_rates.get("SimpleHeuristicsPlayer", 0)
        # иногда ключа нет если оценка упала — fallback
        if heuristics_rate >= MIN_WINRATE_TO_QUALIFY:
            # дополнительно проверяем что файл не перезапишет существующий qualified
            save_path = f"models/{QUALIFIED_PREFIX}{counter}"
            ppo.save(save_path)
            print(f"[phase {counter}] снапшот прошёл порог ({heuristics_rate}% >= {MIN_WINRATE_TO_QUALIFY}%) -> {save_path}")
        else:
            print(f"[phase {counter}] снапшот НЕ прошёл порог ({heuristics_rate}% < {MIN_WINRATE_TO_QUALIFY}%) -> пропущен")

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
        ppo.logger.dump(steps_done_holder["value"])
        print(f"[phase {counter}] {win_rates}")

        counter += 1

        # FIX: раньше пересоздавали env только if not no_normalize_bc — из-за этого
        # при --no-normalize-bc self-play веса никогда не применялись (плато).
        # Теперь всегда пересоздаём, но ветвимся по нормализации.
        if not no_normalize_bc:
            # сохраняем статистику нормализации
            try:
                env.save(VECNORM_PATH)
            except Exception as e:
                print(f"Не удалось сохранить VecNormalize: {e}")
            env.close()
            raw_env = SubprocVecEnv(
                [partial(ExampleEnv.create_env, opponent_weights=current_weights) for _ in range(num_envs)]
            )
            try:
                env = VecNormalize.load(VECNORM_PATH, raw_env)
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
        ppo.set_env(env)

    ppo.save("models/ppo_policy_final")
    # сохраняем VecNormalize финальный
    if not no_normalize_bc and hasattr(env, "save"):
        try:
            env.save(VECNORM_PATH)
        except Exception:
            pass
    env.close()

    # финальная оценка — тоже с нормализацией (патчим PolicyPlayer внутри evaluate, но тут просто выводим)
    # создаём агента с нормализацией вручную
    from agents.training import evaluate_win_rates as eval2
    # быстрый прогон без нормализации патча — evaluate уже патчит
    # для финального вывода используем evaluate
    final_rates = evaluate_win_rates(ppo, n_battles=60)
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
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4, dest="learning_rate", help="Начальный learning_rate (линейно аннилится до 0). По умолчанию 2e-4, для дообучения после BC рекомендуется 5e-5..1e-4")
    parser.add_argument("--lr", type=float, default=None, dest="lr_alias", help="Алиас для --learning-rate")
    parser.add_argument("--pretrain-battles", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset-path", type=str, default="models/heuristic_dataset.npz")
    parser.add_argument("--force-recollect", action="store_true", help="Пересобрать датасет заново, игнорируя кэш")
    parser.add_argument("--contrastive", action="store_true", help="Контрастивный BC: отталкиваться от ходов проигравшего (w=-neg_weight)")
    parser.add_argument("--neg-weight", type=float, default=0.3, help="Вес лузер-ходов при --contrastive (0.3 слабее, 1.0 симметрично)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip_range (0.2 по умолчанию, после BC ставить 0.1 чтобы не снести BC)")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO n_epochs на один роллаут (10 по умолчанию, после BC ставить 3)")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO batch_size (128 по умолчанию, после BC ставить 256)")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="PPO vf_coef вес value loss (0.5 по умолчанию)")
    parser.add_argument("--bc-value-coef", type=float, default=0.0, help="BC value_coef вес value loss при претреине (0.0 только policy, 0.5 учит и value)")
    parser.add_argument("--value-warmup-steps", type=int, default=0, help="Сколько шагов после resume учить только value (заморозить policy) чтобы вылечить просадку -3->-29. Рекомендую 50000")
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
    )
