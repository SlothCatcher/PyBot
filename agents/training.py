import asyncio
import os
import gc
import pickle
import glob
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
            try:
                new_ent = self.ent_schedule(1.0)
                self.ppo_ref.ent_coef = new_ent
            except Exception:
                pass
        return True


def make_lr_schedule(initial_lr: float, total_timesteps: int, steps_holder: dict):
    if total_timesteps is None or total_timesteps <= 0:
        return lambda progress_remaining: initial_lr
    def lr_schedule(progress_remaining: float) -> float:
        global_progress = max(1.0 - (steps_holder["value"] / total_timesteps), 0.0)
        return initial_lr * global_progress
    return lr_schedule

def make_ent_schedule(total_timesteps: int, steps_holder: dict):
    if total_timesteps is None or total_timesteps <= 0:
        return lambda progress_remaining: 0.01
    def ent_schedule(progress_remaining: float) -> float:
        current_step = steps_holder["value"]
        progress = min(current_step / total_timesteps, 1.0)
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
    if len(dataset) == 0:
        print(f"WARNING: save_dataset {path} пустой список, пропускаю")
        return
    obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    mask_arr = np.stack([d[1] for d in dataset]).astype(np.int8)
    action_arr = np.array([d[2] for d in dataset], dtype=np.int64)
    if len(dataset[0]) == 4:
        return_arr = np.array([d[3] for d in dataset], dtype=np.float32)
        np.savez_compressed(path, obs=obs_arr, mask=mask_arr, action=action_arr, ret=return_arr)
    else:
        np.savez_compressed(path, obs=obs_arr, mask=mask_arr, action=action_arr)
    print(f"Датасет сохранён: {path} ({len(dataset)} примеров)")

def load_dataset(path: str) -> list:
    # Лёгкая обёртка для совместимости — для больших файлов лучше использовать load_dataset_arrays (mmap)
    data = np.load(path, mmap_mode='r') if os.path.getsize(path) > 500_000_000 else np.load(path)
    if "ret" in data:
        # для больших файлов не делаем list(zip) — это дублирует память (1.2M туплов ~10GB), возвращаем ленивый вид
        # но для совместимости со старым кодом пока делаем list только для маленьких файлов
        if data["obs"].shape[0] > 200_000:
            print(f"Датасет {path} большой ({data['obs'].shape[0]} примеров), возвращаю mmap-вид (без list) — используйте pretrain_policy_bc с путём")
            # вернём специальный объект-обёртку, который pretrain поймёт
            return _DatasetView(data)
        dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
    else:
        dataset = list(zip(data["obs"], data["mask"], data["action"]))
        print(f"WARNING: датасет {path} без поля ret (старый формат). BC без ret невозможен — нужен пересбор.")
    print(f"Датасет загружен: {path} ({len(dataset)} примеров)")
    return dataset

class _DatasetView:
    """Лёгкий view на npz без копирования — для больших датасетов, чтобы проверка кэша не ела RAM."""
    def __init__(self, npz):
        self.npz = npz
        self.obs = npz["obs"]
        self.mask = npz["mask"]
        self.action = npz["action"]
        self.ret = npz["ret"] if "ret" in npz else None
    def __len__(self):
        return int(self.obs.shape[0])
    def __getitem__(self, idx):
        if self.ret is not None:
            return (self.obs[idx], self.mask[idx], int(self.action[idx]), float(self.ret[idx]))
        return (self.obs[idx], self.mask[idx], int(self.action[idx]))

# ---------------- Chunked / streaming helpers (fix 61GB swap on 50k) ----------------

HEURISTIC_RAW_CACHE = "models/heuristic_raw_cache.pkl"
HEURISTIC_RAW_CACHE_DIR = "models/heuristic_raw_chunks"
HEURISTIC_DATASET_TMP_DIR = "models/heuristic_dataset_tmp"
DEFAULT_CHUNK_SIZE = 1000

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _sanitize_for_pickle(obj):
    """Пробует pickle, если падает из-за _thread.lock — заменяет непиклибельные battles на SimpleNamespace(won)."""
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return obj
    except Exception as e:
        if "cannot pickle" not in str(e) and "_thread.lock" not in str(e) and "lock" not in str(e).lower():
            # неизвестная ошибка — всё равно пробуем санитизацию
            pass
        # пытаемся санитизировать battles/raw_dataset
        try:
            import types, copy
            if isinstance(obj, dict) and "battles" in obj and "raw_dataset" in obj:
                battles = obj.get("battles", {})
                sanitized_battles = {}
                for k, v in list(battles.items())[:10000]:
                    try:
                        pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                        sanitized_battles[k] = v
                    except Exception:
                        try:
                            ns = types.SimpleNamespace()
                            ns.won = getattr(v, "won", None)
                            ns.battle_tag = getattr(v, "battle_tag", k)
                            sanitized_battles[k] = ns
                        except Exception:
                            continue
                # raw_dataset: каждый entry[0] может быть battle_copy
                raw = obj.get("raw_dataset", [])
                sanitized_raw = []
                for entry in raw:
                    if len(entry) == 8:
                        bc, mask, act, tag, of, opf, opr, oppr = entry
                        try:
                            pickle.dumps(bc, protocol=pickle.HIGHEST_PROTOCOL)
                            sanitized_raw.append(entry)
                        except Exception:
                            try:
                                # stripped stub уже должен быть пиклибелен — если нет, пропускаем bc
                                import types as _t
                                stub = _t.SimpleNamespace()
                                for attr in ["battle_tag","gen","weather","fields","side_conditions","opponent_side_conditions","available_moves","team","opponent_team","active_pokemon","opponent_active_pokemon","player_role"]:
                                    if hasattr(bc, attr):
                                        try:
                                            stub.__dict__[attr] = getattr(bc, attr)
                                        except Exception:
                                            pass
                                stub.battle_tag = getattr(bc, "battle_tag", tag)
                                sanitized_raw.append((stub, mask, act, tag, of, opf, opr, oppr))
                            except Exception:
                                sanitized_raw.append((None, mask, act, tag, of, opf, opr, oppr))
                    else:
                        sanitized_raw.append(entry)
                return {"raw_dataset": sanitized_raw, "battles": sanitized_battles, "chunk_idx": obj.get("chunk_idx")}
            else:
                return obj
        except Exception:
            return obj
    return obj

def _count_raw_chunk_battles() -> int:
    if not os.path.isdir(HEURISTIC_RAW_CACHE_DIR):
        return 0
    # быстрый путь: читаем .meta.json без загрузки тяжёлых pickle (каждый pickle 20-100MB с battle_copy)
    total = 0
    meta_files = glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "raw_chunk_*.meta.json"))
    if meta_files:
        for mf in meta_files:
            try:
                import json
                with open(mf, "r") as f:
                    meta = json.load(f)
                    total += int(meta.get("battles", 0))
            except Exception:
                pass
        pkl_files = glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "raw_chunk_*.pkl"))
        if len(meta_files) == len(pkl_files):
            return total
    total = 0
    for fn in glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "raw_chunk_*.pkl")):
        meta_fn = fn.replace(".pkl", ".meta.json")
        if os.path.exists(meta_fn):
            try:
                import json
                with open(meta_fn, "r") as f:
                    meta = json.load(f)
                    total += int(meta.get("battles", 0))
                    continue
            except Exception:
                pass
        try:
            with open(fn, "rb") as f:
                data = pickle.load(f)
                total += len(data.get("battles", {}))
        except Exception:
            pass
    return total

def _list_raw_chunk_files():
    if not os.path.isdir(HEURISTIC_RAW_CACHE_DIR):
        return []
    return sorted(glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "raw_chunk_*.pkl")))

def _list_dataset_chunk_files(tmp_dir: str = HEURISTIC_DATASET_TMP_DIR):
    if not os.path.isdir(tmp_dir):
        return []
    return sorted(glob.glob(os.path.join(tmp_dir, "dataset_chunk_*.npz")))

def _save_raw_chunk(raw_dataset: list, battles: dict, chunk_idx: int):
    _ensure_dir(HEURISTIC_RAW_CACHE_DIR)
    path = os.path.join(HEURISTIC_RAW_CACHE_DIR, f"raw_chunk_{chunk_idx:04d}.pkl")
    tmp = path + ".tmp"
    payload = {"raw_dataset": raw_dataset, "battles": battles, "chunk_idx": chunk_idx}
    # защита от cannot pickle '_thread.lock' — санитизируем если нужно
    payload = _sanitize_for_pickle(payload)
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    try:
        import json
        meta_path = path.replace(".pkl", ".meta.json")
        with open(meta_path, "w") as mf:
            json.dump({"battles": len(battles), "transitions": len(raw_dataset), "chunk_idx": chunk_idx}, mf)
    except Exception:
        pass
    print(f"  Сырой чанк {chunk_idx} сохранён: {path} ({len(raw_dataset)} переходов, {len(battles)} боёв)")

def _save_dataset_chunk(dataset_with_ret: list, chunk_idx: int, tmp_dir: str = HEURISTIC_DATASET_TMP_DIR):
    _ensure_dir(tmp_dir)
    path = os.path.join(tmp_dir, f"dataset_chunk_{chunk_idx:04d}.npz")
    # np.savez_compressed добавляет .npz если путь не заканчивается на .npz, поэтому используем .tmp.npz
    tmp = path.replace(".npz", ".tmp.npz") if path.endswith(".npz") else path + ".tmp.npz"
    # dataset_with_ret is list of (obs, mask, action, ret)
    save_dataset(dataset_with_ret, tmp)
    # save_dataset мог не создать файл если список пустой
    if os.path.exists(tmp):
        os.replace(tmp, path)
        print(f"  Датасет чанк {chunk_idx} сохранён: {path} ({len(dataset_with_ret)} примеров)")
    else:
        print(f"  Датасет чанк {chunk_idx} пустой, пропускаю сохранение")

def _merge_dataset_chunks(chunk_files: list, final_path: str):
    if not chunk_files:
        print("WARNING: нет чанков для мержа")
        return
    print(f"Мержу {len(chunk_files)} чанков в {final_path} ...")
    # First pass: count total and get dims
    total = 0
    obs_dim = None
    mask_dim = None
    has_ret = None
    for cf in chunk_files:
        try:
            d = np.load(cf)
            n = len(d["obs"])
            total += n
            if obs_dim is None:
                obs_dim = d["obs"].shape[1]
                mask_dim = d["mask"].shape[1] if d["mask"].ndim > 1 else 1
                has_ret = "ret" in d
        except Exception as e:
            print(f"  пропуск битого чанка {cf}: {e}")
    if total == 0:
        print("WARNING: все чанки пустые")
        return
    print(f"  Всего {total} примеров, obs_dim={obs_dim}, mask_dim={mask_dim}, has_ret={has_ret}")
    # Preallocate arrays (peak ~3.6GB for 50k, vs 61GB swap before)
    obs_arr = np.empty((total, obs_dim), dtype=np.float32)
    # mask may be 2D (N, 9) or 1D? Check features: action mask size is 9? Actually depends on SinglesEnv
    # We'll handle both
    sample_mask = np.load(chunk_files[0])["mask"]
    if sample_mask.ndim == 2:
        mask_arr = np.empty((total, sample_mask.shape[1]), dtype=np.int8)
    else:
        mask_arr = np.empty((total,), dtype=np.int8)
    action_arr = np.empty((total,), dtype=np.int64)
    ret_arr = np.empty((total,), dtype=np.float32) if has_ret else None

    offset = 0
    for cf in chunk_files:
        d = np.load(cf)
        n = len(d["obs"])
        obs_arr[offset:offset+n] = d["obs"]
        mask_arr[offset:offset+n] = d["mask"]
        action_arr[offset:offset+n] = d["action"]
        if has_ret and "ret" in d:
            ret_arr[offset:offset+n] = d["ret"]
        offset += n
        del d
        gc.collect()
    _ensure_dir(os.path.dirname(final_path) or ".")
    # атомарная запись: временный файл должен заканчиваться на .npz чтобы np.savez не добавил суффикс
    if final_path.endswith(".npz"):
        tmp = final_path.replace(".npz", ".tmp.npz")
    else:
        tmp = final_path + ".tmp.npz"
    if has_ret:
        np.savez_compressed(tmp, obs=obs_arr, mask=mask_arr, action=action_arr, ret=ret_arr)
    else:
        np.savez_compressed(tmp, obs=obs_arr, mask=mask_arr, action=action_arr)
    os.replace(tmp, final_path)
    print(f"  Финальный датасет сохранён: {final_path} ({total} примеров)")
    # free
    del obs_arr, mask_arr, action_arr
    if ret_arr is not None:
        del ret_arr
    gc.collect()

def _recompute_from_chunked_cache(n_battles: int) -> list:
    """Пересобирает датасет из chunked raw cache без новых боёв, но стримингово (по чанкам) чтобы не держать 1.2M battle_copy в памяти."""
    from .features import embed_battle_with_fusion
    from collections import defaultdict
    chunk_files = _list_raw_chunk_files()
    if not chunk_files:
        return None
    print(f"Кэш хит (chunked): {len(chunk_files)} чанков, пересобираю obs без новых боёв для {n_battles} боёв")
    # Need to collect up to n_battles battles worth of tags
    # First, iterate chunks to collect tags until we have n_battles
    needed_tags = []
    battles_collected = {}
    raw_entries_needed = []  # will be streamed, but we need to limit
    # We will stream recompute per chunk and write to tmp dataset chunks, then merge
    _ensure_dir(HEURISTIC_DATASET_TMP_DIR)
    # clear tmp dataset chunks for recompute?
    # Use separate tmp for recompute
    recompute_tmp = HEURISTIC_DATASET_TMP_DIR + "_recompute"
    _ensure_dir(recompute_tmp)
    # clear recompute tmp
    for f in glob.glob(os.path.join(recompute_tmp, "*.npz")):
        try:
            os.remove(f)
        except:
            pass
    tag_count = 0
    chunk_idx_out = 0
    for cf in chunk_files:
        if tag_count >= n_battles:
            break
        try:
            with open(cf, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  пропуск битого raw чанка {cf}: {e}")
            continue
        raw_dataset = data.get("raw_dataset", [])
        battles = data.get("battles", {})
        # How many new battles does this chunk add?
        remaining = n_battles - tag_count
        # If battles count > remaining, we need to slice this chunk's battles/tags
        # We need to group raw_dataset by tag to slice
        from collections import defaultdict as dd
        grouped = dd(list)
        for entry in raw_dataset:
            if len(entry) == 8:
                tag = entry[3]
                grouped[tag].append(entry)
            elif len(entry) == 4:
                continue
        tags_in_chunk = list(grouped.keys())
        # If we would exceed, truncate
        if len(tags_in_chunk) > remaining:
            tags_in_chunk = tags_in_chunk[:remaining]
            # filter raw_dataset
            filtered_raw = []
            for tag in tags_in_chunk:
                filtered_raw.extend(grouped[tag])
            raw_dataset = filtered_raw
            battles = {k: v for k, v in battles.items() if k in tags_in_chunk}
        else:
            # keep as is
            pass
        # Now recompute obs for this chunk's raw_dataset
        recomputed = []
        for entry in raw_dataset:
            if len(entry) != 8:
                continue
            battle_copy, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect = entry
            try:
                obs = embed_battle_with_fusion(battle_copy, our_fusion, opp_fusion, our_protected_last_turn=our_protect, opp_protected_last_turn=opp_protect)
                recomputed.append((obs, mask, action, tag))
            except Exception:
                continue
        # compute returns
        grouped2 = defaultdict(list)
        for obs, mask, action, tag in recomputed:
            grouped2[tag].append((obs, mask, action))
        # also filter battles to those with won not None
        final_chunk = []
        for tag, transitions in grouped2.items():
            battle = battles.get(tag)
            if battle is None or getattr(battle, "won", None) is None:
                continue
            outcome = 30.0 if battle.won else -30.0
            n = len(transitions)
            for i, (obs, mask, action) in enumerate(transitions):
                ret = outcome * (0.99 ** (n - i - 1))
                final_chunk.append((obs, mask, action, ret))
        if final_chunk:
            _save_dataset_chunk(final_chunk, chunk_idx_out, tmp_dir=recompute_tmp)
            chunk_idx_out += 1
        tag_count += len(tags_in_chunk)
        # free
        del raw_dataset, battles, recomputed, final_chunk
        gc.collect()
        print(f"  Recompute chunk {cf} -> {len(tags_in_chunk)} боёв, total {tag_count}/{n_battles}")
    # Now merge recomputed chunks into memory list? For return we need list, but we can merge into final array and then load as list via streaming merge without holding all raw
    # Instead of merging to single file, we will merge recomputed chunks into final dataset file in recompute_tmp and then load
    chunk_files_out = _list_dataset_chunk_files(recompute_tmp)
    if not chunk_files_out:
        print("Recompute: нет данных после пересчёта")
        return []
    # Merge into single array in memory and return list (peak still 3.6GB but not 61GB)
    # We can directly load merged via _merge_dataset_chunks to a temp final path and then load
    merged_path = os.path.join(recompute_tmp, "_merged.npz")
    _merge_dataset_chunks(chunk_files_out, merged_path)
    # Load as list
    data = np.load(merged_path)
    if "ret" in data:
        dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
    else:
        dataset = list(zip(data["obs"], data["mask"], data["action"]))
    print(f"Собрано {len(dataset)} примеров (с return) из chunked кэша {tag_count} боёв")
    # cleanup recompute tmp? Keep for debug
    return dataset

def _get_npz_obs_dim(path: str):
    """Быстро достаёт obs_dim из .npz без загрузки всего массива (3.6GB для 50k)."""
    try:
        import zipfile
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('obs.npy') as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                if len(shape) >= 2:
                    return int(shape[1])
                return None
    except Exception:
        pass
    try:
        data = np.load(path, mmap_mode='r')
        if "obs" in data:
            return int(data["obs"].shape[1])
    except Exception:
        pass
    return None

def _get_npz_n_transitions(path: str):
    """Быстро достаёт количество переходов (shape[0]) без загрузки массива."""
    try:
        import zipfile
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('obs.npy') as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                if len(shape) >= 1:
                    return int(shape[0])
                return None
    except Exception:
        pass
    try:
        data = np.load(path, mmap_mode='r')
        if "obs" in data:
            return int(data["obs"].shape[0])
    except Exception:
        pass
    return None

def _get_legacy_battle_count_fast() -> int:
    """Быстро узнаёт количество боёв в legacy pkl без загрузки raw_dataset (через .meta или только battles)."""
    meta_path = HEURISTIC_RAW_CACHE + ".meta.json"
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                return int(json.load(f).get("battles", 0))
        except Exception:
            pass
    # fallback: пробуем загрузить только battles без raw_dataset через pickle streaming
    # pickle не поддерживает частичную загрузку, поэтому читаем файл и ищем количество через быстрый парсинг
    # но для 100 боёв это быстро, для 50k — всё равно тяжело, поэтому лучше сразу вернуть 0 и идти в chunked
    if os.path.exists(HEURISTIC_RAW_CACHE):
        try:
            # пробуем быстро: читаем первые 1MB и ищем b'battles'
            # но надёжнее просто загрузить с mmap: всё равно для 100 боёв это 100MB, не критично
            # для больших legacy (50k) — лучше не грузить, а считать что 0 и идти в chunked
            size = os.path.getsize(HEURISTIC_RAW_CACHE)
            if size > 500_000_000:  # >500MB — считаем слишком большим для проверки кэша, пропускаем
                print(f"Legacy кэш слишком большой ({size//1024//1024}MB), пропускаю проверку (иди в chunked)")
                return 0
            with open(HEURISTIC_RAW_CACHE, "rb") as f:
                data = pickle.load(f)
                return len(data.get("battles", {}))
        except Exception:
            pass
    return 0

def collect_or_load_dataset(n_battles: int, path: str, force_recollect: bool = False) -> list:
    # если датасет есть, dim совпадает и хватает переходов — грузим, иначе доберём/допересоберём
    if os.path.exists(path) and not force_recollect:
        try:
            obs_dim = _get_npz_obs_dim(path)
            n_trans = _get_npz_n_transitions(path)
            from .config import N_FEATURES
            need_trans = n_battles * 8  # минимум 8 переходов на бой
            has_enough = n_trans is not None and n_trans >= need_trans
            dim_mismatch = obs_dim is not None and obs_dim != N_FEATURES
            if dim_mismatch:
                print(f"Датасет {path} dim {obs_dim} != {N_FEATURES} — пересобираю из сырого кэша без новых боёв")
                try:
                    if os.path.isdir(HEURISTIC_RAW_CACHE_DIR) and _count_raw_chunk_battles() >= n_battles:
                        dataset = _recompute_from_chunked_cache(n_battles)
                        if dataset is not None and len(dataset) > 0:
                            _ensure_dir(os.path.dirname(path) or ".")
                            import shutil
                            recompute_merged = os.path.join(HEURISTIC_DATASET_TMP_DIR + "_recompute", "_merged.npz")
                            if os.path.exists(recompute_merged):
                                shutil.copyfile(recompute_merged, path)
                                print(f"Пересобранный датасет скопирован в {path}")
                            else:
                                save_dataset(dataset, path)
                            return dataset
                    # legacy — только если не огромный (иначе chunked уже покрыл)
                    n_legacy = _get_legacy_battle_count_fast()
                    if n_legacy >= n_battles:
                        # размер guard уже внутри _get_legacy_battle_count_fast, но перепроверим
                        try:
                            if os.path.getsize(HEURISTIC_RAW_CACHE) <= 500_000_000:
                                raw_cached, battles_cached = _load_heuristic_raw_cache()
                                if raw_cached is not None and len(battles_cached) >= n_battles:
                                    recomputed = _recompute_dataset_from_raw(raw_cached, battles_cached)
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
                        except Exception as e2:
                            print(f"Legacy пересбор не удался: {e2}")
                except Exception as e:
                    print(f"Пересбор из кэша не удался: {e}, пересобираю боями...")
                    import traceback
                    traceback.print_exc()
                # если пересбор не удался — падаем в collect
                raise ValueError(f"dim mismatch {obs_dim} != {N_FEATURES}")
            if not has_enough and n_trans is not None:
                print(f"Датасет {path} имеет {n_trans} переходов, нужно ~{need_trans} для {n_battles} боёв — доберу")
                raise ValueError(f"not enough transitions {n_trans} < {need_trans}")
            # dim ok и хватает данных — возвращаем
            return load_dataset(path)
        except ValueError as ve:
            # not enough или dim mismatch — идём в сбор, не считаем ошибкой
            print(f"  -> добор через collect_heuristic_dataset: {ve}")
        except Exception as e:
            print(f"Не удалось загрузить датасет {path}: {e}, пересобираю...")
            import traceback
            traceback.print_exc()
    dataset = collect_heuristic_dataset(n_battles=n_battles, force_recollect=force_recollect)
    # collect_heuristic_dataset in chunked mode already saved merged file to path? Check if path exists and dataset is None?
    # If collect returned list, save it
    if isinstance(dataset, list) and len(dataset) > 0:
        # If path already exists from chunked merge, don't overwrite if same
        # But if we are in chunked mode, collect already merged to some tmp and we need to save to path
        # Check if path exists and is recent (merged from chunks)
        # For simplicity, if dataset is list and path not exists or force, save
        if not os.path.exists(path) or force_recollect:
            # For large dataset, use chunked merge path if available
            # If chunk files exist, merge them directly to path to avoid double save
            chunk_files = _list_dataset_chunk_files()
            if len(chunk_files) > 1 and len(dataset) > 10000:
                # prefer merge from chunks (more memory efficient than save_dataset which stacks again)
                print(f"Сохраняю датасет через мерж чанков в {path}")
                _merge_dataset_chunks(chunk_files, path)
                # reload dataset from file to ensure consistency? Keep returned list
            else:
                save_dataset(dataset, path)
        else:
            # path exists, maybe already merged
            pass
    elif isinstance(dataset, list):
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

def _save_heuristic_raw_cache(raw_dataset: list, battles: dict, n_battles: int):
    try:
        import pickle, json
        os.makedirs(os.path.dirname(HEURISTIC_RAW_CACHE), exist_ok=True)
        payload = {"raw_dataset": raw_dataset, "battles": battles, "n_battles": n_battles}
        payload = _sanitize_for_pickle(payload)
        with open(HEURISTIC_RAW_CACHE, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        # лёгкий meta для быстрой проверки без загрузки pickle
        try:
            with open(HEURISTIC_RAW_CACHE + ".meta.json", "w") as mf:
                json.dump({"battles": len(battles), "transitions": len(raw_dataset), "n_battles": n_battles}, mf)
        except Exception:
            pass
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
        if len(entry) == 8:
            battle_copy, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect = entry
        elif len(entry) == 4:
            continue
        else:
            continue
        try:
            obs = embed_battle_with_fusion(battle_copy, our_fusion, opp_fusion, our_protected_last_turn=our_protect, opp_protected_last_turn=opp_protect)
            recomputed.append((obs, mask, action, tag))
        except Exception as e:
            continue
    return recomputed

def collect_heuristic_dataset(n_battles: int = 200, force_recollect: bool = False, use_cache: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE, resume: bool = True) -> list:
    """
    Собирает датасет боями SimpleHeuristics vs SimpleHeuristics.
    Для больших n_battles (>chunk_size) пишет на диск пачками по chunk_size боёв,
    не держа всё в памяти (фикс 61GB swap на 50k). При краше можно возобновить — уже готовые чанки пропускаются.
    """
    # 1) пробуем взять из кэша (chunked или legacy) без новых боёв
    if use_cache and not force_recollect:
        # chunked cache hit
        n_cached_chunked = _count_raw_chunk_battles()
        if n_cached_chunked >= n_battles:
            print(f"Кэш хит (chunked): {n_cached_chunked} боёв в {HEURISTIC_RAW_CACHE_DIR} >= {n_battles} запрошено — пересобираю obs без новых боёв (стриминг)")
            ds = _recompute_from_chunked_cache(n_battles)
            if ds is not None and len(ds) > 0:
                return ds
            print("Chunked кэш дал 0 примеров, пробую legacy...")
        # legacy single — сначала быстрая проверка без загрузки 5GB pickle
        n_cached_battles_fast = _get_legacy_battle_count_fast()
        # только если быстро нашли что достаточно и legacy не огромный — грузим
        if n_cached_battles_fast > 0:
            # проверяем размер прежде чем грузить
            try:
                if os.path.getsize(HEURISTIC_RAW_CACHE) > 500_000_000:
                    print(f"Legacy кэш большой, пропускаю legacy-путь (иди в chunked добор)")
                    n_cached_battles_fast = 0
                    raw_cached = battles_cached = None
                else:
                    raw_cached, battles_cached = _load_heuristic_raw_cache()
                    if raw_cached is None or battles_cached is None:
                        n_cached_battles_fast = 0
                    else:
                        n_cached_battles = len(battles_cached)
                        # переопределим fast для дальнейшего elif
                        n_cached_battles_fast = n_cached_battles
            except Exception:
                raw_cached = battles_cached = None
                n_cached_battles_fast = 0
        else:
            raw_cached = battles_cached = None
        if raw_cached is not None and battles_cached is not None:
            n_cached_battles = len(battles_cached)
            if n_cached_battles >= n_battles:
                print(f"Кэш хит: {n_cached_battles} боёв в {HEURISTIC_RAW_CACHE} >= {n_battles} запрошено — пересобираю obs без новых боёв")
                recomputed = _recompute_dataset_from_raw(raw_cached, battles_cached)
                from collections import defaultdict
                grouped = defaultdict(list)
                for obs, mask, action, tag in recomputed:
                    grouped[tag].append((obs, mask, action))
                tags = list(grouped.keys())[:n_battles]
                filtered_recomputed = []
                for tag in tags:
                    for obs, mask, action in grouped[tag]:
                        filtered_recomputed.append((obs, mask, action, tag))
                filtered_battles = {tag: battles_cached[tag] for tag in tags if tag in battles_cached}
                final_dataset = _compute_bc_returns(filtered_recomputed, filtered_battles)
                print(f"Собрано {len(final_dataset)} примеров (с return) из кэша {n_battles} боёв, исходно {len(filtered_recomputed)} переходов (без новых боёв)")
                if len(final_dataset) > 0:
                    return final_dataset
                print("Кэш дал 0 примеров, пересобираю боями...")
            elif n_cached_battles > 0:
                # если chunked уже есть — не мигрируем legacy повторно (избегаем дублей после прерванной миграции)
                if _count_raw_chunk_battles() > 0:
                    print(f"Кэш: {n_cached_battles}/{n_battles} боёв legacy, но chunked уже содержит {_count_raw_chunk_battles()} боёв — пропускаю миграцию (избегаю дублей)")
                    # проверяем, что для каждого raw чанка есть dataset чанк, иначе пересобираем
                    raw_files = _list_raw_chunk_files()
                    dataset_files = set(_list_dataset_chunk_files())
                    missing = []
                    for rf in raw_files:
                        try:
                            idx = int(os.path.basename(rf).split("_")[-1].split(".")[0])
                        except:
                            continue
                        df = os.path.join(HEURISTIC_DATASET_TMP_DIR, f"dataset_chunk_{idx:04d}.npz")
                        if df not in dataset_files and not os.path.exists(df):
                            missing.append((rf, idx))
                    if missing:
                        print(f"  Найдены raw чанки без dataset ({len(missing)}), пересобираю...")
                        for rf, idx in missing:
                            try:
                                with open(rf, "rb") as f:
                                    data = pickle.load(f)
                                raw_dataset = data.get("raw_dataset", [])
                                battles = data.get("battles", {})
                                recomputed = _recompute_dataset_from_raw(raw_dataset, battles)
                                dataset_for_chunk = [(obs, mask, action, tag) for obs, mask, action, tag in recomputed]
                                final_for_chunk = _compute_bc_returns(dataset_for_chunk, battles)
                                _save_dataset_chunk(final_for_chunk, idx)
                                del raw_dataset, battles, recomputed, dataset_for_chunk, final_for_chunk
                                gc.collect()
                                # удаляем битые .tmp если остались
                                for leftover in glob.glob(os.path.join(HEURISTIC_DATASET_TMP_DIR, "*.tmp.npz")) + glob.glob(os.path.join(HEURISTIC_DATASET_TMP_DIR, "*.tmp")):
                                    try:
                                        os.remove(leftover)
                                    except:
                                        pass
                            except Exception as e:
                                print(f"  Не удалось пересобрать dataset для raw {rf}: {e}")
                    # чистим остатки .tmp.npz от прошлого падения
                    for leftover in glob.glob(os.path.join(HEURISTIC_DATASET_TMP_DIR, "*.tmp.npz")) + glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "*.tmp")):
                        try:
                            os.remove(leftover)
                        except:
                            pass
                    n_cached_chunked = _count_raw_chunk_battles()
                    print(f"  Chunked теперь {n_cached_chunked} боёв, продолжаю добор пачками...")
                else:
                    print(f"Кэш: {n_cached_battles}/{n_battles} боёв — доберу {n_battles - n_cached_battles} новых боёв (legacy, без чанков)")
                    # Переходим в chunked добор: сохраним legacy в chunked формат и продолжим чанками
                    print("  Мигрирую legacy кэш в chunked формат для добора пачками...")
                    _ensure_dir(HEURISTIC_RAW_CACHE_DIR)
                    _ensure_dir(HEURISTIC_DATASET_TMP_DIR)
                    from collections import defaultdict as dd
                    grouped_raw = dd(list)
                    for entry in raw_cached:
                        if len(entry) == 8:
                            grouped_raw[entry[3]].append(entry)
                    tags = list(grouped_raw.keys())
                    for idx in range(0, len(tags), chunk_size):
                        chunk_tags = tags[idx: idx+chunk_size]
                        chunk_raw = []
                        chunk_battles = {}
                        for t in chunk_tags:
                            chunk_raw.extend(grouped_raw[t])
                            if t in battles_cached:
                                chunk_battles[t] = battles_cached[t]
                        existing = _list_raw_chunk_files()
                        next_idx = len(existing)
                        _save_raw_chunk(chunk_raw, chunk_battles, next_idx)
                        recomputed = _recompute_dataset_from_raw(chunk_raw, chunk_battles)
                        dataset_for_chunk = [(obs, mask, action, tag) for obs, mask, action, tag in recomputed]
                        final_for_chunk = _compute_bc_returns(dataset_for_chunk, chunk_battles)
                        _save_dataset_chunk(final_for_chunk, next_idx)
                        del chunk_raw, chunk_battles, recomputed, final_for_chunk
                        gc.collect()
                    n_cached_chunked = _count_raw_chunk_battles()
                    print(f"  Миграция завершена, chunked теперь {n_cached_chunked} боёв")
                # продолжим добор как chunked (ниже)
                # не возвращаем, падаем в chunked сбор
    # 2) Если n_battles маленький и нет chunked кэша — старый быстрый путь без чанков (совместимость)
    if n_battles <= chunk_size and not os.path.isdir(HEURISTIC_RAW_CACHE_DIR):
        # обычный путь: новые бои одним батчем (для 200 боёв быстрее и проще)
        dataset: list = []
        raw_dataset: list = []
        recorder = HeuristicRecorder(dataset=dataset, raw_dataset=raw_dataset, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        asyncio.run(recorder.battle_against(opponent, n_battles=n_battles))
        battles = getattr(recorder, "battles", None) or getattr(recorder, "_battles", {})
        if use_cache:
            _save_heuristic_raw_cache(raw_dataset, battles, n_battles)
        final_dataset = _compute_bc_returns(dataset, battles)
        print(f"Собрано {len(final_dataset)} примеров (с return) из {n_battles} боёв, исходно {len(dataset)} переходов")
        if len(final_dataset) == 0 and len(dataset) > 0:
            print("WARNING: все переходы отфильтрованы (battle.won is None). Проверьте версию poke_env и логику сбора.")
        return final_dataset

    # 3) Chunked путь для больших n_battles (50k) — пачками по 1000, с записью на диск и resume
    _ensure_dir(HEURISTIC_RAW_CACHE_DIR)
    _ensure_dir(HEURISTIC_DATASET_TMP_DIR)
    # чистим битые .tmp от прошлого падения (Windows np.savez добавлял .npz)
    for leftover in glob.glob(os.path.join(HEURISTIC_DATASET_TMP_DIR, "*.tmp")) + glob.glob(os.path.join(HEURISTIC_DATASET_TMP_DIR, "*.tmp.npz")) + glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "*.tmp")):
        try:
            os.remove(leftover)
        except:
            pass
    if force_recollect:
        print(f"force_recollect: очищаю чанки {HEURISTIC_RAW_CACHE_DIR} и {HEURISTIC_DATASET_TMP_DIR}")
        for f in glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "raw_chunk_*.pkl")) + glob.glob(os.path.join(HEURISTIC_RAW_CACHE_DIR, "raw_chunk_*.meta.json")):
            try:
                os.remove(f)
            except:
                pass
        for f in glob.glob(os.path.join(HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_*.npz")):
            try:
                os.remove(f)
            except:
                pass
        gc.collect()

    # Определяем сколько уже собрано в chunked кэше (для resume / добора)
    n_cached_chunked = _count_raw_chunk_battles()
    # Определяем стартовый индекс чанка
    existing_raw_chunks = _list_raw_chunk_files()
    existing_dataset_chunks = _list_dataset_chunk_files()
    # Самый простой способ resume: смотрим сколько чанков уже есть, и сколько боёв в них
    # num_existing_chunks = len(existing_raw_chunks)
    # Но если n_cached_chunked >= n_battles, мы уже вышли выше (cache hit). Значит n_cached < n_battles
    # Нужно добрать remaining = n_battles - n_cached_chunked
    remaining_total = n_battles - n_cached_chunked
    if remaining_total <= 0:
        # всё уже есть, просто мержим и возвращаем
        print(f"Chunked кэш уже содержит {n_cached_chunked} боёв >= {n_battles}, мержу чанки")
        chunk_files = _list_dataset_chunk_files()
        # Обрезаем до нужного кол-ва боёв если есть лишние (редко)
        # Для простоты: если есть 50 чанков по 1000 и нужно 50000, берём все
        # Если нужно меньше, пересобираем через _recompute
        if n_cached_chunked > n_battles:
            return _recompute_from_chunked_cache(n_battles)
        # Merge and return
        # Use recompute merge helper to avoid double memory?
        # We already have dataset chunks, just merge
        tmp_merged = os.path.join(HEURISTIC_DATASET_TMP_DIR, "_merged_final.npz")
        _merge_dataset_chunks(chunk_files, tmp_merged)
        data = np.load(tmp_merged)
        if "ret" in data:
            dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
        else:
            dataset = list(zip(data["obs"], data["mask"], data["action"]))
        print(f"Собрано {len(dataset)} примеров из chunked кэша (resume, без новых боёв)")
        return dataset

    # Иначе надо добрать remaining_total боёв пачками
    num_existing_chunks = len(existing_raw_chunks)
    # Вычисляем сколько чанков нужно добрать
    n_chunks_needed = (remaining_total + chunk_size - 1) // chunk_size
    print(f"Chunked сбор: нужно добрать {remaining_total} боёв ({n_chunks_needed} чанков по {chunk_size}), уже есть {n_cached_chunked} боёв в {num_existing_chunks} чанках")
    # Для каждого нового чанка
    for i in range(n_chunks_needed):
        chunk_idx = num_existing_chunks + i
        need = min(chunk_size, remaining_total - i * chunk_size)
        raw_chunk_path = os.path.join(HEURISTIC_RAW_CACHE_DIR, f"raw_chunk_{chunk_idx:04d}.pkl")
        dataset_chunk_path = os.path.join(HEURISTIC_DATASET_TMP_DIR, f"dataset_chunk_{chunk_idx:04d}.npz")
        if resume and os.path.exists(raw_chunk_path) and os.path.exists(dataset_chunk_path):
            # Проверка что чанк полный (кол-во боёв совпадает)
            try:
                with open(raw_chunk_path, "rb") as f:
                    d = pickle.load(f)
                    n_in_chunk = len(d.get("battles", {}))
                if n_in_chunk >= need * 0.9:  # допуск 90% (иногда бои не завершаются)
                    print(f"Чанк {chunk_idx} уже существует ({n_in_chunk} боёв), пропускаю")
                    continue
                else:
                    print(f"Чанк {chunk_idx} неполный ({n_in_chunk}/{need}), пересобираю")
            except Exception as e:
                print(f"Чанк {chunk_idx} битый ({e}), пересобираю")
        print(f"--- Чанк {chunk_idx+1}/{num_existing_chunks + n_chunks_needed}: собираю {need} боёв ---")
        dataset_chunk: list = []
        raw_dataset_chunk: list = []
        recorder = HeuristicRecorder(dataset=dataset_chunk, raw_dataset=raw_dataset_chunk, battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        try:
            asyncio.run(recorder.battle_against(opponent, n_battles=need))
        except Exception as e:
            print(f"Ошибка в чанке {chunk_idx}: {e}")
            import traceback
            traceback.print_exc()
        battles_chunk = getattr(recorder, "battles", None) or getattr(recorder, "_battles", {})
        n_finished = len(battles_chunk)
        print(f"  Чанк {chunk_idx} завершён: {n_finished}/{need} боёв, {len(dataset_chunk)} переходов")
        # Сохраняем сырой чанк сразу на диск
        try:
            _save_raw_chunk(raw_dataset_chunk, battles_chunk, chunk_idx)
        except Exception as e:
            print(f"  Не удалось сохранить сырой чанк {chunk_idx}: {e}")
        # Считаем returns и сохраняем датасет чанк
        try:
            final_chunk = _compute_bc_returns(dataset_chunk, battles_chunk)
            _save_dataset_chunk(final_chunk, chunk_idx)
            print(f"  Чанк {chunk_idx} датасет: {len(final_chunk)} примеров с return")
        except Exception as e:
            print(f"  Не удалось сохранить датасет чанк {chunk_idx}: {e}")
            import traceback
            traceback.print_exc()
        # Освобождаем память
        del dataset_chunk, raw_dataset_chunk, battles_chunk
        if 'final_chunk' in locals():
            del final_chunk
        del recorder, opponent
        gc.collect()
        # Принудительно чистим poke_env внутренние кэши? Нет
        print(f"  Память после чанка {chunk_idx}: освобождена, осталось {n_chunks_needed - i -1} чанков")

    # После всех чанков — мержим датасет чанки в один список для возврата
    chunk_files = _list_dataset_chunk_files()
    if not chunk_files:
        print("ERROR: после chunked сбора нет датасет чанков")
        return []
    # Если n_battles не кратно chunk_size, последний чанк уже правильный, мержим все
    # Но если мы добрали и теперь всего больше чем нужно (из-за округления), обрежем через recompute? Пока просто мержим всё
    # Для точного n_battles используем _recompute_from_chunked_cache если нужно обрезать, иначе мержим напрямую
    total_battles_after = _count_raw_chunk_battles()
    print(f"Все чанки собраны: {total_battles_after} боёв в {len(chunk_files)} чанках (запрошено {n_battles})")
    if total_battles_after > n_battles:
        print(f"  Больше чем нужно ({total_battles_after}>{n_battles}), обрезаю через recompute")
        return _recompute_from_chunked_cache(n_battles)
    # Обычный мерж
    tmp_merged = os.path.join(HEURISTIC_DATASET_TMP_DIR, "_merged_final.npz")
    _merge_dataset_chunks(chunk_files, tmp_merged)
    data = np.load(tmp_merged)
    if "ret" in data:
        dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
    else:
        dataset = list(zip(data["obs"], data["mask"], data["action"]))
    print(f"Собрано {len(dataset)} примеров (с return) из {total_battles_after} боёв (chunked, по {chunk_size} на чанк, исходно ~{len(dataset)} переходов)")
    # Также обновим legacy single cache для совместимости? Не нужно, но можем сохранить мерж raw в single для старых скриптов (опционально, но это снова OOM для 50k)
    # Не сохраняем single для больших n
    return dataset

_win_rate_ema: dict[str, float] = {}
_EMA_ALPHA = 0.3
_MIN_WEIGHT = 0.10
_MAX_WEIGHT = 0.45

def warm_up_vec_normalize(vec_normalize, dataset):
    if len(dataset) == 0:
        return
    # поддержка _DatasetView: у него obs уже массив (mmap)
    if isinstance(dataset, _DatasetView):
        obs_arr = np.asarray(dataset.obs, dtype=np.float32)
    elif isinstance(dataset, str) and os.path.exists(dataset):
        data = np.load(dataset, mmap_mode='r')
        obs_arr = np.asarray(data["obs"], dtype=np.float32)
    else:
        # list path — для больших датасетов делаем батчевую оценку чтобы не stack 3.6GB сразу
        if len(dataset) > 200_000:
            # считаем среднее по батчам 10k
            batch = 10000
            # возьмём первые 50000 для warmup чтобы не грузить всё
            sample_n = min(len(dataset), 50000)
            obs_sample = np.stack([dataset[i][0] for i in range(sample_n)]).astype(np.float32)
            if "observation" in vec_normalize.obs_rms:
                vec_normalize.obs_rms["observation"].update(obs_sample)
            return
        obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
    if "observation" in vec_normalize.obs_rms:
        vec_normalize.obs_rms["observation"].update(obs_arr)

def _update_opponent_weights(win_rates: dict[str, float]):
    for name, rate in win_rates.items():
        rate_frac = rate / 100.0
        prev = _win_rate_ema.get(name, rate_frac)
        _win_rate_ema[name] = _EMA_ALPHA * rate_frac + (1 - _EMA_ALPHA) * prev

def _get_opponent_weights(names: list[str]) -> list[float]:
    raw = []
    for n in names:
        ema = _win_rate_ema.get(n, 0.5)
        w = max(1.0 - ema, 0.05)
        raw.append(w)
    total = sum(raw)
    if total == 0:
        return [1.0/len(names)]*len(names)
    weights = [w / total for w in raw]
    weights = [min(max(w, _MIN_WEIGHT), _MAX_WEIGHT) for w in weights]
    total2 = sum(weights)
    return [w / total2 for w in weights]

def evaluate_win_rates(ppo, n_battles: int = 180) -> dict[str, float]:
    vec_norm = ppo.get_vec_normalize_env() if hasattr(ppo, "get_vec_normalize_env") else None
    base_agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
    orig_embed = base_agent.embed_battle
    if vec_norm is not None:
        try:
            def norm_embed(battle):
                raw = orig_embed(battle)
                normed = vec_norm.normalize_obs({"observation": raw[None, :]})["observation"][0]
                return normed
            base_agent.embed_battle = norm_embed  # type: ignore
        except Exception as e:
            print(f"Не удалось патчить нормализацию для eval: {e}")

    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    n_sp_appended = 0
    sp_opps: list = []
    try:
        from .env import _make_self_play_opponents
        sp_opps = _make_self_play_opponents()
        if sp_opps:
            # _make_self_play_opponents() возвращает до 3 снапшотов, но добавляем ровно 1
            # и создаём свежий PolicyPlayer без start_listening=False (battle_against нужен слушающий сокет)
            # иначе battle_against может зависнуть с 0 finished battles
            base_sp = sp_opps[-1]
            try:
                eval_sp = PolicyPlayer(policy=getattr(base_sp, "policy", None), battle_format=BATTLE_FORMAT, max_concurrent_battles=30)
                opponents.append(eval_sp)
            except Exception:
                opponents.append(base_sp)
            n_sp_appended = 1
    except Exception:
        sp_opps = []
        n_sp_appended = 0

    asyncio.run(base_agent.battle_against(*opponents, n_battles=n_battles))
    rates: dict[str, float] = {}
    for idx, opp in enumerate(opponents):
        if n_sp_appended and idx >= len(opponents) - n_sp_appended:
            key = "self_play"
        else:
            key = opp.__class__.__name__
        if opp.n_finished_battles == 0:
            rates[key] = 0.0
        else:
            rates[key] = round(100 * opp.n_lost_battles / opp.n_finished_battles, 1)
    return rates

def pretrain_policy_bc(
    ppo: PPO, dataset, epochs: int = 50, batch_size: int = 256,
    normalize: bool = False, value_coef: float = 0.0, val_frac: float = 0.1,
    patience: int = 5,
    contrastive: bool = False, neg_weight: float = 0.3,
):
    # dataset может быть list, _DatasetView (mmap) или путь к .npz
    if isinstance(dataset, str) and os.path.exists(dataset):
        # путь — грузим mmap
        print(f"BC: гружу датасет по пути {dataset} (mmap)")
        data = np.load(dataset, mmap_mode='r') if os.path.getsize(dataset) > 200_000_000 else np.load(dataset)
        if "ret" not in data:
            raise ValueError("Датасет без ret: соберите заново с _compute_bc_returns (нужен victory_value).")
        # используем mmap напрямую без list(zip)
        obs_arr = data["obs"]
        mask_arr = data["mask"]
        action_arr = data["action"]
        return_arr = data["ret"]
        # для совместимости создаём view
        dataset_len = int(obs_arr.shape[0])
        # проверим dim
        from .config import N_FEATURES
        if obs_arr.shape[1] != N_FEATURES:
            raise ValueError(f"BC obs dim {obs_arr.shape[1]} != N_FEATURES {N_FEATURES}. Пересоберите датасет или обновите config.")
        # normalize
        n = dataset_len
        # делаем пермутацию без копирования всего массива в RAM? используем индексы
        # для экономии RAM не делаем np.stack — уже массивы
        is_mmap = True
    else:
        if len(dataset) == 0:
            print("BC: пустой датасет, пропускаю")
            return
        # проверка на _DatasetView
        if isinstance(dataset, _DatasetView):
            if dataset.ret is None:
                raise ValueError("Датасет без ret: соберите заново с _compute_bc_returns (нужен victory_value).")
            obs_arr = dataset.obs
            mask_arr = dataset.mask
            action_arr = dataset.action
            return_arr = dataset.ret
            dataset_len = len(dataset)
            is_mmap = True
            from .config import N_FEATURES
            if obs_arr.shape[1] != N_FEATURES:
                raise ValueError(f"BC obs dim {obs_arr.shape[1]} != N_FEATURES {N_FEATURES}. Пересоберите датасет или обновите config.")
            n = dataset_len
        else:
            # обычный list
            if len(dataset[0]) == 3:
                raise ValueError("Датасет без ret: соберите заново с _compute_bc_returns (нужен victory_value).")
            # для больших list (>200k) не делаем один большой stack — это 3.6GB, делаем через mmap view если можно
            if len(dataset) > 200_000:
                print(f"BC: датасет большой ({len(dataset)}), делаю временный stack по частям")
            obs_arr = np.stack([d[0] for d in dataset]).astype(np.float32)
            mask_arr = np.stack([d[1] for d in dataset]).astype(np.float32)
            action_arr = np.array([d[2] for d in dataset], dtype=np.int64)
            return_arr = np.array([d[3] for d in dataset], dtype=np.float32)
            from .config import N_FEATURES
            if obs_arr.shape[1] != N_FEATURES:
                raise ValueError(f"BC obs dim {obs_arr.shape[1]} != N_FEATURES {N_FEATURES}. Пересоберите датасет или обновите config.")
            n = len(dataset)
            is_mmap = False
            dataset_len = n

    # дальнейшая логика общая — используем obs_arr/mask_arr/action_arr/return_arr как массивы
    # для mmap это уже np.memmap, для list — обычные ndarray
    # dim уже проверен выше, повторная проверка не нужна

    if normalize:
        vec_normalize = ppo.get_vec_normalize_env()
        if vec_normalize is not None:
            # warm_up: для пути dataset это data view, для _DatasetView — сам объект
            warm_arg = dataset
            if isinstance(dataset, str) and 'data' in locals():
                # создаём view для warm_up
                try:
                    warm_arg = _DatasetView(data)
                except Exception:
                    warm_arg = dataset
            warm_up_vec_normalize(vec_normalize, warm_arg)
            if is_mmap and obs_arr.shape[0] > 200_000:
                print(f"BC: нормализую большой mmap ({obs_arr.shape[0]}) по частям")
                normed = np.empty(obs_arr.shape, dtype=np.float32)
                chunk = 50000
                for s in range(0, obs_arr.shape[0], chunk):
                    e = min(s+chunk, obs_arr.shape[0])
                    normed[s:e] = vec_normalize.normalize_obs({"observation": np.asarray(obs_arr[s:e], dtype=np.float32)})["observation"]
                obs_arr = normed
                print(f"BC: нормализовал {len(obs_arr)} obs через VecNormalize (mean {vec_normalize.obs_rms['observation'].mean[:3]})")
            else:
                obs_arr = vec_normalize.normalize_obs({"observation": np.asarray(obs_arr, dtype=np.float32)})["observation"]
                print(f"BC: нормализовал {len(obs_arr)} obs через VecNormalize (mean {vec_normalize.obs_rms['observation'].mean[:3]})")
        else:
            print("BC: normalize=True но VecNormalize не найден — обучаю на сырых obs")

    # n уже определён выше (dataset_len), не переопределяем len(dataset) для пути-строки
    # если вдруг n не определён (старый путь), fallback
    if 'n' not in locals() or n is None:
        try:
            n = len(dataset)
        except Exception:
            n = int(obs_arr.shape[0])
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
                log_prob = distribution.log_prob(action_batch)
                prob = log_prob.exp().clamp(1e-6, 1-1e-6)
                win_mask = return_batch > 0
                lose_mask = return_batch < 0
                win_loss = -log_prob[win_mask].mean() if win_mask.any() else torch.tensor(0.0, device=device)
                if lose_mask.any():
                    lose_loss = -torch.log(1 - prob[lose_mask] + 1e-8).mean()
                    policy_loss = win_loss + neg_weight * lose_loss
                else:
                    policy_loss = win_loss
            else:
                policy_loss = -distribution.log_prob(action_batch).mean()
            values = ppo.policy.value_net(latent_vf).flatten()
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
        try:
            ppo.policy.optimizer.state.clear()
        except Exception:
            try:
                from collections import defaultdict
                ppo.policy.optimizer.state = defaultdict(dict)
            except Exception:
                pass
