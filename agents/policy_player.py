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
import torch
import torch.nn as nn

def _migrate_ppo_713_to_715(ppp_path: str):
    """Надёжная миграция 713->715 через паддинг весов в zip. Не зависит от глобального N_FEATURES."""
    import torch
    import tempfile
    import os
    OLD_N = 713
    NEW_N = 715
    print(f"  Миграция 713->715 для {ppp_path}...")
    try:
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
        for key in list(policy_state.keys()):
            tensor = policy_state[key]
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 2 and tensor.shape[1] == OLD_N and tensor.shape[0] == 512:
                if "features_extractor" in key and "weight" in key:
                    new_tensor = torch.zeros((tensor.shape[0], NEW_N), dtype=tensor.dtype, device=tensor.device)
                    new_tensor[:, :OLD_N] = tensor
                    # последние 2 колонки — нули (is_tera флаги)
                    policy_state[key] = new_tensor
                    padded += 1
                    print(f"    паддинг {key} {list(tensor.shape)} -> {list(new_tensor.shape)}")
        if padded == 0:
            print("    WARN: не нашёл весов [512,713] для паддинга — возможно уже 715 или другая архитектура")

        # обновляем observation_space в data если есть
        try:
            if isinstance(data, dict) and "observation_space" in data:
                obs_space = data["observation_space"]
                if hasattr(obs_space, "spaces") and "observation" in obs_space.spaces:
                    old_shape = obs_space.spaces["observation"].shape
                    if old_shape == (OLD_N,):
                        am_space = obs_space.spaces.get("action_mask", Box(0, 1, shape=(9,), dtype=bool))
                        new_obs_space = Dict({"observation": Box(-1, 4, shape=(NEW_N,), dtype="float32"), "action_mask": am_space})
                        data["observation_space"] = new_obs_space
                        print(f"    обновил data['observation_space'] {old_shape} -> {(NEW_N,)}")
        except Exception as e:
            print(f"    не удалось обновить observation_space в data: {e}")

        # сбрасываем optimizer state чтобы не тянуть 713 моменты
        if pytorch_variables is not None:
            print("    сбрасываю optimizer state (713->715) — будет новый оптимизатор")
            pytorch_variables = None

        # сохраняем пропатченный чекпоинт во временный файл и грузим как обычный PPO
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_to_zip_file(tmp_path, data=data, params=params, pytorch_variables=pytorch_variables)
            print(f"  Сохраняю пропатченный чекпоинт во временный файл {tmp_path}")
            ppo_new = PPO.load(tmp_path, device="cpu")
            print(f"  Успешно загрузил мигрированный PPO")
            # критично: сбрасываем Adam моменты (713) — иначе exp_avg 713 vs grad 715 -> RuntimeError
            try:
                if hasattr(ppo_new, "policy") and hasattr(ppo_new.policy, "optimizer") and ppo_new.policy.optimizer is not None:
                    ppo_new.policy.optimizer.state.clear()
                    print("    сбросил optimizer.state (713->715)")
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
        print(f"  load_from_zip_file миграция не удалась: {e}, пробую fallback через создание нового PPO и копирование весов")
        import traceback
        traceback.print_exc()
        # fallback: создаём новый PPO с 715 и копируем веса напрямую
        try:
            # создаём dummy env с 715
            from agents.env import ExampleEnv
            from stable_baselines3.common.vec_env import SubprocVecEnv
            dummy_env = SubprocVecEnv([ExampleEnv.create_env for _ in range(1)])
            # пробуем загрузить старый PPO через временный 713 конфиг если предыдущий способ упал
            # последний шанс: пробуем просто загрузить с strict=False через низкоуровневый torch
            # создаём новый PPO
            from agents.policy import MaskedActorCriticPolicy
            ppo_new = PPO(MaskedActorCriticPolicy, dummy_env, device="cpu", verbose=0)
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
                    if isinstance(tt, torch.Tensor) and tt.dim()==2 and tt.shape[1]==OLD_N and tt.shape[0]==512 and "features_extractor" in kk:
                        nt = torch.zeros((512, NEW_N), dtype=tt.dtype, device=tt.device)
                        nt[:, :OLD_N] = tt
                        policy_state2[kk] = nt
                # загружаем в ppo_new
                try:
                    ppo_new.policy.load_state_dict(policy_state2, strict=False)
                    print("  Fallback: загрузил падденный state_dict напрямую в новый PPO (strict=False)")
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

def _migrate_vecnormalize_713_to_715(vec_path: str, base_env):
    """Грузит VecNormalize с 713 статистикой и расширяет obs_rms до 715 (mean 0, var 1 для новых 2 признаков)."""
    try:
        vec = VecNormalize.load(vec_path, base_env)
        # проверяем размер
        try:
            mean = vec.obs_rms["observation"].mean
            if len(mean) == 713 and base_env.observation_spaces[base_env.possible_agents[0]].shape[0] == 715:
                import numpy as np
                print(f"  Мигрирую VecNormalize 713->715 (mean {mean.shape} -> 715)")
                new_mean = np.zeros(715, dtype=mean.dtype)
                new_mean[:713] = mean
                # new_var по умолчанию 1 для новых признаков (ненормализованные)
                old_var = vec.obs_rms["observation"].var
                new_var = np.ones(715, dtype=old_var.dtype)
                new_var[:713] = old_var
                vec.obs_rms["observation"].mean = new_mean
                vec.obs_rms["observation"].var = new_var
                # count остаётся прежним
        except Exception as e:
            print(f"  VecNormalize миграция 713->715 не удалась: {e}")
        return vec
    except Exception as e:
        msg = str(e)
        if "713" in msg or "715" in msg or "shape" in msg.lower():
            print(f"  VecNormalize.load упал из-за 713->715, создаю новый VecNormalize (статистика сброшена): {e}")
            return VecNormalize(base_env, norm_obs=True, norm_reward=False, gamma=0.99, norm_obs_keys=["observation"])
        raise



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
    vf_coef: float = 0.5,
    bc_value_coef: float = 0.0,
    value_warmup_steps: int = 0,
    min_winrate: int = 25,
    reset_schedules: bool = False,
    eval_battles: int = 20,
    skip_eval: bool = False,
):
    # phase_size должен делиться на фактический размер роллаута n_steps*num_envs (с учётом целочисленного деления)
    rollout_size = (3072 // num_envs) * num_envs
    if phase_size % rollout_size != 0:
        print(f"WARNING: phase_size {phase_size} не кратен фактическому размеру роллаута {rollout_size} (n_steps {3072 // num_envs} * num_envs {num_envs}). Будет обрезка последнего роллаута.")

    run_name = f"{'retrain' if resume_from else 'train'}_{time.strftime('%Y%m%d_%H%M%S')}"

    if resume_from:
        try:
            ppo = PPO.load(resume_from, device="cpu")
        except RuntimeError as e:
            if "713" in str(e) and "715" in str(e):
                print(f"Старый снапшот {resume_from} с 713 признаками — мигрирую на 715...")
                ppo = _migrate_ppo_713_to_715(resume_from)
            else:
                raise
        except Exception as e:
            # SB3 иногда оборачивает RuntimeError
            if "713" in str(e) and "715" in str(e):
                print(f"Старый снапшот {resume_from} с 713 признаками — мигрирую на 715...")
                ppo = _migrate_ppo_713_to_715(resume_from)
            else:
                raise
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
        # иногда ключа нет если оценка упала — fallback
        if heuristics_rate >= min_winrate:
            # дополнительно проверяем что файл не перезапишет существующий qualified
            save_path = f"models/{QUALIFIED_PREFIX}{counter}"
            ppo.save(save_path)
            print(f"[phase {counter}] снапшот прошёл порог ({heuristics_rate}% >= {min_winrate}%) -> {save_path}")
        else:
            print(f"[phase {counter}] снапшот НЕ прошёл порог ({heuristics_rate}% < {min_winrate}%) -> пропущен")

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
    )
