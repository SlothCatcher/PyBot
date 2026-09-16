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
    """Считает timesteps и одновременно обновляет ent_coef по расписанию (если передано)."""
    def __init__(self, steps_holder: dict, num_envs: int, ent_schedule=None, ppo_ref=None):
        self.steps_holder = steps_holder
        self.num_envs = num_envs
        self.ent_schedule = ent_schedule
        self.ppo_ref = ppo_ref  # ссылка на PPO чтобы менять ent_coef на лету

    def __call__(self, _locals, _globals) -> bool:
        self.steps_holder["value"] += self.num_envs
        if self.ent_schedule is not None and self.ppo_ref is not None:
            # SB3 передает progress_remaining, но у нас свой счётчик
            # ent_schedule ожидает steps_holder, так что просто вызываем
            try:
                new_ent = self.ent_schedule(1.0)  # аргумент не важен, внутри считается по holder
                self.ppo_ref.ent_coef = new_ent
            except Exception:
                pass
        return True


def make_lr_schedule(initial_lr: float, total_timesteps: int, steps_holder: dict):
    # --total-timesteps 0 используется для "только BC, без RL" (pretrain-battles>0, total 0)
    if total_timesteps is None or total_timesteps <= 0:
        return lambda progress_remaining: initial_lr
    def lr_schedule(progress_remaining: float) -> float:
        # progress_remaining от SB3 игнорируем, считаем по holder для консистентности resume
        global_progress = max(1.0 - (steps_holder["value"] / total_timesteps), 0.0)
        return initial_lr * global_progress
    return lr_schedule

def make_ent_schedule(total_timesteps: int, steps_holder: dict):
    if total_timesteps is None or total_timesteps <= 0:
        return lambda progress_remaining: 0.01
    def ent_schedule(progress_remaining: float) -> float:
        current_step = steps_holder["value"]
        progress = min(current_step / total_timesteps, 1.0)
        # FIX: 0.05 было слишком агрессивно — после BC политика сразу размывалась и ep_rew 10→-10.
        # Новый мягкий график: старт 0.01 (как дефолт PPO) → 0.005 → 0.001
        if progress <= 0.2:
            return 0.01
        elif progress <= 0.7:
            phase_progress = (progress - 0.2) / 0.5
            return 0.01 - phase_progress * (0.01 - 0.005)
        else:
            phase_progress = (progress - 0.7) / 0.3
            return 0.005 - phase_progress * (0.005 - 0.001)
    return ent_schedule


def save_dataset(dataset: list, path: str):
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    mask_arr = np.stack([d[1] for d in dataset]).astype(np.int8)
    action_arr = np.array([d[2] for d in dataset], dtype=np.int64)
    # dataset может содержать ret (BC с return) или нет (старый формат)
    if len(dataset[0]) == 4:
        return_arr = np.array([d[3] for d in dataset], dtype=np.float32)
        np.savez_compressed(path, obs=obs_arr, mask=mask_arr, action=action_arr, ret=return_arr)
    else:
        np.savez_compressed(path, obs=obs_arr, mask=mask_arr, action=action_arr)
    print(f"Датасет сохранён: {path} ({len(dataset)} примеров)")

def load_dataset(path: str) -> list:
    data = np.load(path)
    if "ret" in data:
        dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
    else:
        # старый датасет без ret — вернём тройки, BC потом пересчитает или упадёт с понятным сообщением
        dataset = list(zip(data["obs"], data["mask"], data["action"]))
        print(f"WARNING: датасет {path} без поля ret (старый формат). BC без ret невозможен — нужен пересбор.")
    print(f"Датасет загружен: {path} ({len(dataset)} примеров)")
    return dataset

def collect_or_load_dataset(n_battles: int, path: str, force_recollect: bool = False) -> list:
    # если датасет есть и размерность совпадает — грузим, иначе пробуем пересобрать из сырого кэша без боёв
    if os.path.exists(path) and not force_recollect:
        try:
            data = np.load(path)
            obs_dim = data["obs"].shape[1] if "obs" in data else None
            from .config import N_FEATURES
            if obs_dim is not None and obs_dim != N_FEATURES:
                print(f"Датасет {path} dim {obs_dim} != N_FEATURES {N_FEATURES} — пересобираю из сырого кэша без новых боёв")
                # пробуем пересобрать из сырого кэша
                try:
                    raw_cached, battles_cached = _load_heuristic_raw_cache()
                    if raw_cached is not None and len(battles_cached) >= n_battles:
                        recomputed = _recompute_dataset_from_raw(raw_cached, battles_cached)
                        # фильтруем до n_battles
                        from collections import defaultdict
                        grouped = defaultdict(list)
                        for obs, mask, action, tag in recomputed:
                            grouped[tag].append((obs, mask, action))
                        tags = list(grouped.keys())[:n_battles]
                        filtered = []
                        for tag in tags:
                            for obs, mask, action in grouped[tag]:
                                filtered.append((obs, mask, action, tag))
                        filtered_battles = {tag: battles_cached[tag] for tag in tags if tag in battles_cached}
                        dataset = _compute_bc_returns(filtered, filtered_battles)
                        save_dataset(dataset, path)
                        return dataset
                except Exception as e:
                    print(f"Пересбор из кэша не удался: {e}, пересобираю боями...")
            else:
                return load_dataset(path)
        except Exception as e:
            print(f"Не удалось загрузить датасет {path}: {e}, пересобираю...")
    dataset = collect_heuristic_dataset(n_battles=n_battles, force_recollect=force_recollect)
    save_dataset(dataset, path)
    return dataset

def _next_snapshot_index() -> int:
    from os import listdir
    from .config import SELF_PLAY_PATH
    prefix = SELF_PLAY_PATH.split("/")[-1] + "_"
    try:
        existing = [f for f in listdir("models/") if prefix in f]
    except FileNotFoundError:
        return 0
    nums = []
    for f in existing:
        suffix = f.split("_")[-1].split(".")[0]
        if suffix.isdigit():
            nums.append(int(suffix))
    return max(nums, default=-1) + 1

def _compute_bc_returns(raw_dataset: list, battles: dict, gamma: float = 0.99, victory_value: float = 30.0) -> list:
    from collections import defaultdict
    grouped = defaultdict(list)
    for entry in raw_dataset:
        # entry is (obs, mask, action, tag)
        if len(entry) != 4:
            continue
        obs, mask, action, tag = entry
        grouped[tag].append((obs, mask, action))
    final = []
    for tag, transitions in grouped.items():
        battle = battles.get(tag)
        if battle is None or battle.won is None:
            continue
        outcome = victory_value if battle.won else -victory_value
        n = len(transitions)
        for i, (obs, mask, action) in enumerate(transitions):
            steps_remaining = n - i - 1
            ret = outcome * (gamma ** steps_remaining)
            final.append((obs, mask, action, ret))
    return final

HEURISTIC_RAW_CACHE = "models/heuristic_raw_cache.pkl"

def _save_heuristic_raw_cache(raw_dataset: list, battles: dict, n_battles: int):
    try:
        import pickle
        os.makedirs(os.path.dirname(HEURISTIC_RAW_CACHE), exist_ok=True)
        with open(HEURISTIC_RAW_CACHE, "wb") as f:
            pickle.dump({"raw_dataset": raw_dataset, "battles": battles, "n_battles": n_battles}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Сырой кэш эвристики сохранён: {HEURISTIC_RAW_CACHE} ({len(raw_dataset)} переходов, {len(battles)} боёв)")
    except Exception as e:
        print(f"Не удалось сохранить сырой кэш: {e}")

def _load_heuristic_raw_cache():
    try:
        import pickle
        if not os.path.exists(HEURISTIC_RAW_CACHE):
            return None, None
        with open(HEURISTIC_RAW_CACHE, "rb") as f:
            data = pickle.load(f)
        return data.get("raw_dataset"), data.get("battles")
    except Exception as e:
        print(f"Не удалось загрузить сырой кэш: {e}")
        return None, None

def _recompute_dataset_from_raw(raw_dataset: list, battles: dict) -> list:
    """Пересобирает (obs, mask, action, tag) из сырого кэша с текущими признаками (N_FEATURES)."""
    from .features import embed_battle_with_fusion
    recomputed = []
    for entry in raw_dataset:
        # raw_dataset entries: (battle_copy, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect)
        if len(entry) == 8:
            battle_copy, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect = entry
        elif len(entry) == 4:
            # старый кэш без raw (только obs) — не можем пересчитать, пропускаем
            continue
        else:
            continue
        try:
            obs = embed_battle_with_fusion(battle_copy, our_fusion, opp_fusion, our_protected_last_turn=our_protect, opp_protected_last_turn=opp_protect)
            recomputed.append((obs, mask, action, tag))
        except Exception as e:
            continue
    return recomputed

def collect_heuristic_dataset(n_battles: int = 200, force_recollect: bool = False, use_cache: bool = True) -> list:
    # 1) пробуем взять из сырого кэша (пересчёт без новых боёв при смене признаков)
    if use_cache and not force_recollect:
        raw_cached, battles_cached = _load_heuristic_raw_cache()
        if raw_cached is not None and battles_cached is not None:
            # сколько боёв в кэше?
            n_cached_battles = len(battles_cached)
            if n_cached_battles >= n_battles:
                print(f"Кэш хит: {n_cached_battles} боёв в {HEURISTIC_RAW_CACHE} >= {n_battles} запрошено — пересобираю obs без новых боёв")
                recomputed = _recompute_dataset_from_raw(raw_cached, battles_cached)
                # обрезаем до n_battles по тегам (группируем по tag, берём первые n_battles тегов)
                from collections import defaultdict
                grouped = defaultdict(list)
                for obs, mask, action, tag in recomputed:
                    grouped[tag].append((obs, mask, action))
                # берём первые n_battles тегов
                tags = list(grouped.keys())[:n_battles]
                filtered_recomputed = []
                for tag in tags:
                    for obs, mask, action in grouped[tag]:
                        filtered_recomputed.append((obs, mask, action, tag))
                # battles для return тоже фильтруем
                filtered_battles = {tag: battles_cached[tag] for tag in tags if tag in battles_cached}
                final_dataset = _compute_bc_returns(filtered_recomputed, filtered_battles)
                print(f"Собрано {len(final_dataset)} примеров (с return) из кэша {n_battles} боёв, исходно {len(filtered_recomputed)} переходов (без новых боёв)")
                if len(final_dataset) > 0:
                    return final_dataset
                print("Кэш дал 0 примеров, пересобираю боями...")
            else:
                print(f"Кэш: {n_cached_battles}/{n_battles} боёв — доберу {n_battles - n_cached_battles} новых боёв")
                # доберём недостающие бои и объединим
                dataset: list = []
                raw_dataset: list = []
                recorder = HeuristicRecorder(dataset=dataset, raw_dataset=raw_dataset, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
                opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
                need = n_battles - n_cached_battles
                asyncio.run(recorder.battle_against(opponent, n_battles=need))
                battles_new = getattr(recorder, "battles", None) or getattr(recorder, "_battles", {})
                # recomputed старые
                recomputed = _recompute_dataset_from_raw(raw_cached, battles_cached)
                # новые уже с obs
                # объединяем raw
                combined_raw = raw_cached + raw_dataset
                combined_battles = {**battles_cached, **battles_new}
                # сохраняем обновлённый кэш
                _save_heuristic_raw_cache(combined_raw, combined_battles, n_battles)
                # пересобираем финальный датасет из всех raw (чтобы obs были консистентны с новыми признаками)
                all_recomputed = _recompute_dataset_from_raw(combined_raw, combined_battles)
                final_dataset = _compute_bc_returns(all_recomputed, combined_battles)
                print(f"Собрано {len(final_dataset)} примеров (с return) из {n_battles} боёв (кэш+добор), исходно {len(all_recomputed)} переходов")
                return final_dataset
    # 2) обычный путь: новые бои
    dataset: list = []
    raw_dataset: list = []
    recorder = HeuristicRecorder(dataset=dataset, raw_dataset=raw_dataset, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    asyncio.run(recorder.battle_against(opponent, n_battles=n_battles))
    battles = getattr(recorder, "battles", None) or getattr(recorder, "_battles", {})
    # сохраняем сырой кэш
    if use_cache:
        _save_heuristic_raw_cache(raw_dataset, battles, n_battles)
    final_dataset = _compute_bc_returns(dataset, battles)
    print(f"Собрано {len(final_dataset)} примеров (с return) из {n_battles} боёв, исходно {len(dataset)} переходов")
    if len(final_dataset) == 0 and len(dataset) > 0:
        print("WARNING: все переходы отфильтрованы (battle.won is None). Проверьте версию poke_env и логику сбора.")
    return final_dataset

_win_rate_ema: dict[str, float] = {}
_EMA_ALPHA = 0.3
_MIN_WEIGHT = 0.10
_MAX_WEIGHT = 0.45

def warm_up_vec_normalize(vec_normalize, dataset):
    # dataset: list of (obs, mask, action, ret) — берём только obs
    if len(dataset) == 0:
        return
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    # VecNormalize хранит RunningMeanStd в dict по ключу "observation"
    if "observation" in vec_normalize.obs_rms:
        vec_normalize.obs_rms["observation"].update(obs_arr)

def _update_opponent_weights(win_rates: dict[str, float]):
    for name, rate in win_rates.items():
        rate_frac = rate / 100.0
        prev = _win_rate_ema.get(name, rate_frac)
        _win_rate_ema[name] = _EMA_ALPHA * rate_frac + (1 - _EMA_ALPHA) * prev
        # debug
    # если self_play впервые появился, EMA уже записана

def _get_opponent_weights(names: list[str]) -> list[float]:
    # names — список категорий, например ["RandomPlayer","MaxBasePowerPlayer","SimpleHeuristicsPlayer","self_play"]
    # Для self_play используем EMA по ключу "self_play" (если его нет — 0.5 нейтрально)
    raw = []
    for n in names:
        ema = _win_rate_ema.get(n, 0.5)
        # чем ниже винрейт против оппонента, тем больше веса (хотим тренироваться против сильных)
        # для self_play логика такая же: если много проигрываем self_play — больше игр против него
        w = max(1.0 - ema, 0.05)
        raw.append(w)
    total = sum(raw)
    if total == 0:
        return [1.0/len(names)]*len(names)
    weights = [w / total for w in raw]
    # клиппинг чтобы не было экстремальных распределений (генетический дрейф)
    weights = [min(max(w, _MIN_WEIGHT), _MAX_WEIGHT) for w in weights]
    total2 = sum(weights)
    return [w / total2 for w in weights]

def evaluate_win_rates(ppo, n_battles: int = 180) -> dict[str, float]:
    """
    Оценивает винрейт против 3 эвристик + self_play (если есть).
    ВАЖНО: если ppo обучен с VecNormalize, наблюдение нужно нормализовать так же,
    иначе оценка занижена на 10-20% (одна из причин плато 30-40%).
    """
    vec_norm = ppo.get_vec_normalize_env() if hasattr(ppo, "get_vec_normalize_env") else None
    # пытаемся достать VecNormalize даже если обёрнут
    # ppo.get_vec_normalize_env возвращает VecNormalize или None
    # Для оценки создаём агента и патчим embed чтобы нормализовать
    base_agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
    orig_embed = base_agent.embed_battle
    if vec_norm is not None:
        try:
            # проверяем что VecNormalize действительно нормализует observation
            def norm_embed(battle):
                raw = orig_embed(battle)
                # VecNormalize.normalize_obs ожидает batch dim
                normed = vec_norm.normalize_obs({"observation": raw[None, :]})["observation"][0]
                return normed
            base_agent.embed_battle = norm_embed  # type: ignore
        except Exception as e:
            print(f"Не удалось патчить нормализацию для eval: {e}")

    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    # Пробуем добавить self_play как отдельного оппонента для оценки
    # Но self_play снапшоты грузятся через env._make_self_play_opponents, а тут делаем напрямую
    try:
        from .env import _make_self_play_opponents
        sp_opps = _make_self_play_opponents()
        if sp_opps:
            # берём сильнейшего последнего как репрезент self_play
            opponents.append(sp_opps[-1])
            # переименуем чтобы в словаре было "self_play" а не класс
            # Костыль: меняем __class__.__name__ через обёртку
            # проще — после battle_against заменим ключ
    except Exception:
        sp_opps = []

    asyncio.run(base_agent.battle_against(*opponents, n_battles=n_battles))
    rates: dict[str, float] = {}
    for idx, opp in enumerate(opponents):
        # последние sp_opps считаем как self_play
        if idx >= len(opponents) - len(sp_opps) and sp_opps:
            key = "self_play"
        else:
            key = opp.__class__.__name__
        if opp.n_finished_battles == 0:
            rates[key] = 0.0
        else:
            rates[key] = round(100 * opp.n_lost_battles / opp.n_finished_battles, 1)
    # если self_play был, но мы добавили только одного, ключ один
    return rates

def pretrain_policy_bc(
    ppo: PPO, dataset: list, epochs: int = 50, batch_size: int = 256,
    normalize: bool = False, value_coef: float = 0.0, val_frac: float = 0.1,
    patience: int = 5,
    contrastive: bool = False, neg_weight: float = 0.3,
):
    """
    Behavioral Cloning на датасете эвристики.
    Исправления vs оригинал:
    - value_coef 0.0 по умолчанию (было 0.5→0.25) — value от BC масштаба ±30 конфликтует с
      PPO+VecNormalize(norm_reward) где return нормируется к ~1, из-за этого первый PPO
      апдейт давал advantage ~15 и policy коллапсировала 10→-10
    - градиенты клиппятся по норме 0.5 (иначе взрыв из-за большой value loss)
    - проверка размерности obs vs N_FEATURES
    - normalize теперь корректно warm-up'ит VecNormalize
    - contrastive=True: учится НЕ делать как проигравший — policy_loss = -(w*logProb).mean(),
      w=+1 для ret>0 (победитель), w=-neg_weight для ret<0 (проигравший). По умолчанию
      выключено (чистый BC как раньше), включи --contrastive чтобы использовать оба исхода.
    """
    if len(dataset) == 0:
        print("BC: пустой датасет, пропускаю")
        return
    # проверка формата
    if len(dataset[0]) == 3:
        raise ValueError("Датасет без ret: соберите заново с _compute_bc_returns (нужен victory_value).")
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    mask_arr = np.stack([d[1] for d in dataset]).astype(np.float32)
    action_arr = np.array([d[2] for d in dataset], dtype=np.int64)
    return_arr = np.array([d[3] for d in dataset], dtype=np.float32)

    # проверка размерности
    from .config import N_FEATURES
    if obs_arr.shape[1] != N_FEATURES:
        raise ValueError(f"BC obs dim {obs_arr.shape[1]} != N_FEATURES {N_FEATURES}. Пересоберите датасет или обновите config.")

    if normalize:
        vec_normalize = ppo.get_vec_normalize_env()
        if vec_normalize is not None:
            warm_up_vec_normalize(vec_normalize, dataset)
            obs_arr = vec_normalize.normalize_obs({"observation": obs_arr})["observation"]
            print(f"BC: нормализовал {len(obs_arr)} obs через VecNormalize (mean {vec_normalize.obs_rms['observation'].mean[:3]})")
        else:
            print("BC: normalize=True но VecNormalize не найден — обучаю на сырых obs")

    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = ppo.policy.device
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(epochs):
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
            if contrastive:
                # стабильный контрастив: победитель -logProb, проигравший -log(1 - prob)
                # вместо w*logProb (который уходит в -inf), используем ограниченный -log(1-prob) >=0
                log_prob = distribution.log_prob(action_batch)
                prob = log_prob.exp().clamp(1e-6, 1-1e-6)
                win_mask = return_batch > 0
                lose_mask = return_batch < 0
                win_loss = -log_prob[win_mask].mean() if win_mask.any() else torch.tensor(0.0, device=device)
                if lose_mask.any():
                    # -log(1 - prob): 0 когда prob->0, +inf когда prob->1 → минимизация толкает prob к 0
                    lose_loss = -torch.log(1 - prob[lose_mask] + 1e-8).mean()
                    policy_loss = win_loss + neg_weight * lose_loss
                else:
                    policy_loss = win_loss
            else:
                policy_loss = -distribution.log_prob(action_batch).mean()
            values = ppo.policy.value_net(latent_vf).flatten()
            # value loss может быть большой (scale 30), поэтому клип и coef 0.25
            value_loss = torch.nn.functional.mse_loss(values, return_batch)
            loss = policy_loss + value_coef * value_loss

            ppo.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.policy.parameters(), 0.5)
            ppo.policy.optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

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
            if contrastive:
                log_prob = distribution.log_prob(action_batch)
                prob = log_prob.exp().clamp(1e-6, 1-1e-6)
                win_mask = return_batch > 0
                lose_mask = return_batch < 0
                win_loss = -log_prob[win_mask].mean().item() if win_mask.any() else 0.0
                if lose_mask.any():
                    lose_loss = -torch.log(1 - prob[lose_mask] + 1e-8).mean().item()
                    val_policy_loss = win_loss + neg_weight * lose_loss
                else:
                    val_policy_loss = win_loss
            else:
                val_policy_loss = -distribution.log_prob(action_batch).mean().item()
            values = ppo.policy.value_net(latent_vf).flatten()
            val_value_loss = torch.nn.functional.mse_loss(values, return_batch).item()
            val_loss = val_policy_loss + value_coef * val_value_loss

        print(
            f"[BC epoch {epoch}] train_policy={total_policy_loss/n_batches:.4f} "
            f"train_value={total_value_loss/n_batches:.4f} "
            f"val_policy={val_policy_loss:.4f} val_value={val_value_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in ppo.policy.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Ранняя остановка на эпохе {epoch} (val loss не улучшается {patience} эпох)")
                break

    if best_state is not None:
        ppo.policy.load_state_dict(best_state)
        print("Восстановлены веса с лучшей val_loss")
        # сбрасываем оптимизатор после BC чтобы не нести импульс в RL
        # ВАЖНО: нельзя делать optimizer.state = {} (теряется defaultdict -> KeyError в Adam),
        # нужно clear() чтобы сохранить тип defaultdict
        try:
            ppo.policy.optimizer.state.clear()
        except Exception:
            try:
                # fallback: пересоздать как defaultdict если кто-то уже заменил на dict
                from collections import defaultdict
                ppo.policy.optimizer.state = defaultdict(dict)
            except Exception:
                pass
