import asyncio
import os
import gc
import pickle
import glob
import numpy as np
import torch
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from .config import BATTLE_FORMAT, N_FEATURES
from .players import HeuristicRecorder, PolicyPlayer

class LRSchedule:
    def __call__(self, progress): return 3e-5
class ENTSchedule:
    def __call__(self, progress): return 0.03

class StepCounterCallback:
    """Считает timesteps, обновляет ent_coef по расписанию и печатает состав действий.

    Состав действий — прямая проверка жалобы «модель только спамит атаками»: SB3 передаёт
    в callback локальные переменные шага, там есть массив выбранных действий (по одному
    на env). Раскладка poke-env SinglesEnv: 0..5 свитч, 6..9 приём, 10..21 мега/z/динамакс
    (в этом формате не используются), 22..25 тера.
    """
    def __init__(self, steps_holder: dict, num_envs: int, ent_schedule=None, ppo_ref=None,
                 mix_every: int = 20_000):
        self.steps_holder = steps_holder
        self.num_envs = num_envs
        self.ent_schedule = ent_schedule
        self.ppo_ref = ppo_ref  # ссылка на PPO чтобы менять ent_coef на лету
        self.mix_every = max(int(mix_every), 1)
        # Разделяем свитчи, иначе метрика врёт (жалоба: «[mix] 60% свитчей, а в боях модель
        # не свитчила ни разу»):
        #   switch        — свитч, когда приёмы БЫЛИ доступны: собственный выбор политики;
        #   switch_forced — свитч, когда приёмов нет вообще (наш покемон упал, маска = только
        #                   свитчи): в бою это видно как свитч, но выбора «атака или свитч» нет;
        #   single        — шагов с ровно одним разрешённым действием (подмножество
        #                   switch_forced: остался один живой покемон). Диагностика wait-шагов.
        #   choice        — шагов, где у агента был настоящий выбор (>=2 легальных действий и
        #                   есть приёмы). Всё остальное — сервер заставил (фейнт/ожидание).
        self._mix = {"switch": 0, "switch_forced": 0, "move": 0, "tera": 0,
                     "gimmick": 0, "other": 0, "single": 0, "choice": 0}
        self._mix_next = self.mix_every
        self._mask_shape_warned = False
        # Curiosity (--icm): wrapper кладёт r_int/beta/r_int_episode в infos. Собираем здесь,
        # чтобы видеть в логе/TB, не перевешивает ли intrinsic плотный shaping
        # (SHAPING_EPISODE_CAP=12 вводился именно против такого перевеса).
        self._curiosity = {"n": 0, "r_int_sum": 0.0, "beta": None, "ep_max": 0.0, "episodes": 0}

    @staticmethod
    def classify_action(action: int) -> str:
        """Класс действия по конвенции poke-env SinglesEnv (см. ExampleEnv.action_to_order)."""
        a = int(action)
        if 0 <= a < 6:
            return "switch"
        if 22 <= a <= 25:
            return "tera"
        if 10 <= a <= 21:
            return "gimmick"  # мега/z/динамакс: в gen9-фьюжне недоступны
        if 6 <= a <= 9:
            return "move"
        return "other"

    def action_mix(self) -> dict:
        sw_all = self._mix["switch"] + self._mix["switch_forced"]
        total = sum(self._mix[k] for k in ("switch", "switch_forced", "move", "tera", "gimmick", "other"))
        out = dict(self._mix)
        out["_total"] = total            # все шаги, что видел SB3 (= столько же действий ушло в бой)
        out["_decided"] = self._mix["choice"]   # шаги, где был настоящий выбор (>1 варианта и есть приёмы)
        if total:
            out["_switch_share"] = round(sw_all / total, 4)                # сколько ВСЕХ свитчей
            out["_switch_own_share"] = round(self._mix["switch"] / total, 4)      # свитч как выбор
            out["_switch_forced_share"] = round(self._mix["switch_forced"] / total, 4)  # свитч по фейнту
            out["_move_share"] = round(self._mix["move"] / total, 4)
            out["_tera_share"] = round(self._mix["tera"] / total, 4)
            out["_single_share"] = round(self._mix["single"] / total, 4)
            out["_choice_share"] = round(self._mix["choice"] / total, 4)
        return out

    @staticmethod
    def _mask_rows(mask) -> "tuple | None":
        """(single, no_moves) по строкам env-ов из маски действий.

        single   — разрешено ровно одно действие: состояние ожидания соперника (`battle._wait`
                   даёт маску `[1, 0, 0, ...]`) или принудительный свитч с одним живым покемоном.
                   В обоих случаях политика обязана выбрать действие 0, и это индекс СВИТЧА.
        no_moves — приёмов нет вообще (mask[6:] пусто): наш покемон упал, сервер требует свитч.
        """
        try:
            m = mask
            if hasattr(m, "detach"):
                m = m.detach()
            import numpy as _np
            arr = m.cpu().numpy() if hasattr(m, "cpu") else _np.asarray(m)
            arr = _np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return (arr.sum(axis=-1) == 1, arr[..., 6:].sum(axis=-1) == 0)
        except Exception:
            return None

    def __call__(self, _locals, _globals) -> bool:
        self.steps_holder["value"] += self.num_envs
        if self.ent_schedule is not None and self.ppo_ref is not None:
            try:
                new_ent = self.ent_schedule(1.0)
                self.ppo_ref.ent_coef = new_ent
            except Exception:
                pass
        # диагностика: какие действия выбирает политика (свитч/приём/тера).
        # Классифицируем действие ВМЕСТЕ с маской из того же шага, иначе свитчи не отличить от
        # вынужденных (см. комментарий к self._mix).
        try:
            actions = _locals.get("actions") if isinstance(_locals, dict) else None
            if actions is not None:
                acts = np.asarray(actions).reshape(-1)
                rows = None
                obs_t = _locals.get("obs_tensor") if isinstance(_locals, dict) else None
                if isinstance(obs_t, dict) and "action_mask" in obs_t:
                    rows = self._mask_rows(obs_t["action_mask"])
                if rows is None and self.ppo_ref is not None:
                    rows = self._mask_rows(getattr(self.ppo_ref.policy, "_mask", None))
                if rows is not None and len(rows[0]) != acts.size:
                    if not getattr(self, "_mask_shape_warned", False):
                        self._mask_shape_warned = True
                        print(f"WARNING: маска ({len(rows[0])}) не совпала с числом действий ({acts.size}) — "
                              f"[mix] не сможет отделить вынужденные свитчи")
                    rows = None
                single, no_moves = rows if rows is not None else (None, None)
                # curiosity-метрики из infos (есть только при --icm)
                infos = _locals.get("infos") if isinstance(_locals, dict) else None
                if isinstance(infos, (list, tuple)) and infos:
                    cur = self._curiosity
                    for info in infos:
                        if not isinstance(info, dict):
                            continue
                        if "r_int" in info:
                            cur["n"] += 1
                            cur["r_int_sum"] += float(info["r_int"])
                            if "beta" in info:
                                cur["beta"] = float(info["beta"])
                        if "r_int_episode" in info:
                            cur["episodes"] += 1
                            cur["ep_max"] = max(cur["ep_max"], float(info["r_int_episode"]))

                for i, a in enumerate(acts):
                    kind = self.classify_action(int(a))
                    if kind == "switch" and no_moves is not None:
                        # свитч без доступных приёмов = покемон упал; это не выбор «атака/свитч»
                        key = "switch_forced" if bool(no_moves[i]) else "switch"
                    else:
                        key = kind
                    self._mix[key] += 1
                    if single is not None and bool(single[i]):
                        self._mix["single"] += 1
                    if single is not None and no_moves is not None \
                            and not bool(single[i]) and not bool(no_moves[i]):
                        self._mix["choice"] += 1   # был выбор: >=2 действий и доступны приёмы
                total = sum(self._mix[k] for k in
                            ("switch", "switch_forced", "move", "tera", "gimmick", "other"))
                if total >= self._mix_next:
                    self._mix_next = total + self.mix_every
                    self._log_mix(total)
        except Exception:
            pass
        return True

    def _log_mix(self, total: int) -> None:
        mix = self.action_mix()
        n_all = max(mix.get("_total", total), 1)
        own = mix["switch"]
        forced = mix["switch_forced"]
        msg = (f"[mix] решений {mix.get('_total', total)}: "
               f"приём {mix.get('_move_share', 0.0) * 100:.1f}%, "
               f"тера {mix.get('_tera_share', 0.0) * 100:.1f}%, "
               f"свитч {mix.get('_switch_share', 0.0) * 100:.1f}% "
               f"(своих {own} = {own / n_all * 100:.1f}%, вынужденных после фейнта {forced} = {forced / n_all * 100:.1f}%)")
        if mix["gimmick"] or mix["other"]:
            msg += f", прочее {(mix['gimmick'] + mix['other']) / n_all * 100:.1f}%"
        msg += (f"; шагов без выбора {mix['single']} ({mix.get('_single_share', 0.0) * 100:.1f}%), "
                f"своих решений (был выбор) {mix.get('_decided', 0)} "
                f"({mix.get('_choice_share', 0.0) * 100:.1f}%)")
        cur = self._curiosity
        if cur["n"]:
            r_mean = cur["r_int_sum"] / cur["n"]
            beta = cur["beta"] if cur["beta"] is not None else 0.0
            msg += (f" | curiosity: r_int(сред) {r_mean:.4f}, beta {beta:.4f}, "
                    f"вклад за эпизод (макс) {cur['ep_max']:.2f} из {max(cur['episodes'], 1)} эпизодов")
        print(msg)
        try:
            logger = getattr(self.ppo_ref, "logger", None)
            if logger is not None:
                logger.record("mix/switch_share", float(mix.get("_switch_share", 0.0)))
                logger.record("mix/switch_own_share", float(mix.get("_switch_own_share", 0.0)))
                logger.record("mix/switch_forced_share", float(mix.get("_switch_forced_share", 0.0)))
                logger.record("mix/move_share", float(mix.get("_move_share", 0.0)))
                logger.record("mix/tera_share", float(mix.get("_tera_share", 0.0)))
                logger.record("mix/single_share", float(mix.get("_single_share", 0.0)))
                logger.record("mix/choice_share", float(mix.get("_choice_share", 0.0)))
                cur = self._curiosity
                if cur["n"]:
                    logger.record("mix/r_int_mean", float(cur["r_int_sum"] / cur["n"]))
                    logger.record("mix/r_int_episode_max", float(cur["ep_max"]))
                    if cur["beta"] is not None:
                        logger.record("mix/beta", float(cur["beta"]))
        except Exception:
            pass


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
                    if len(entry) in (8, 9):
                        bc, mask, act, tag, of, opf, opr, oppr = entry[:8]
                        tail = tuple(entry[8:])
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
                                sanitized_raw.append((stub, mask, act, tag, of, opf, opr, oppr) + tail)
                            except Exception:
                                sanitized_raw.append((None, mask, act, tag, of, opf, opr, oppr) + tail)
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

def _npz_array_shape(path: str, key: str = "obs"):
    """Форма массива из .npz по заголовку (без загрузки данных). None, если не прочитать."""
    try:
        import zipfile
        with zipfile.ZipFile(path, "r") as z:
            with z.open(f"{key}.npy") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
                else:
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                return tuple(int(x) for x in shape)
    except Exception:
        return None


def _npz_keys(path: str) -> list:
    """Имена массивов внутри .npz (по оглавлению zip)."""
    try:
        import zipfile
        with zipfile.ZipFile(path, "r") as z:
            return [n[:-4] for n in z.namelist() if n.endswith(".npy")]
    except Exception:
        return []


def _dataset_chunk_dims(chunk_files: list) -> dict:
    """Сводка по размерностям чанков датасета: {(obs_dim, mask_w): количество}."""
    dims: dict = {}
    for cf in chunk_files:
        obs_shape = _npz_array_shape(cf, "obs")
        if not obs_shape:
            continue
        mask_shape = _npz_array_shape(cf, "mask")
        mask_w = int(mask_shape[1]) if mask_shape and len(mask_shape) > 1 else 1
        key = (int(obs_shape[1]) if len(obs_shape) > 1 else 0, mask_w)
        dims[key] = dims.get(key, 0) + 1
    return dims


def _wrong_dim_dataset_chunks(chunk_files: list, target_dim: int) -> dict:
    """{obs_dim: количество чанков} для чанков, чья размерность obs != target_dim.

    Нужно, чтобы не подмешивать в датасет чанки, посчитанные прежним набором признаков:
    раскладка менялась не только в конец (713 -> 715 вставил [our_is_tera, opp_is_tera]
    перед tera_type), поэтому «добить нулями справа» — не универсальное решение.
    """
    bad: dict = {}
    for cf in chunk_files:
        shape = _npz_array_shape(cf, "obs")
        if not shape or len(shape) < 2:
            continue
        dim = int(shape[1])
        if dim != int(target_dim):
            bad[dim] = bad.get(dim, 0) + 1
    return bad


def _list_dataset_chunk_files(tmp_dir: str | None = None):
    """Чанки датасета из каталога (по умолчанию HEURISTIC_DATASET_TMP_DIR).

    Каталог читается в момент вызова, а не при импорте: иначе константу нельзя ни
    переопределить, ни подменить в тестах (default-значение связывается один раз).
    """
    tmp_dir = tmp_dir or HEURISTIC_DATASET_TMP_DIR
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
    # Первый проход: количества и размерности (по заголовкам, без загрузки данных)
    total = 0
    obs_dims: set = set()
    mask_dims: set = set()
    ret_counts = {"ret": 0, "no_ret": 0}
    usable = []
    for cf in chunk_files:
        try:
            shape = _npz_array_shape(cf, "obs")
            if not shape or len(shape) < 2 or int(shape[0]) == 0:
                continue
            mshape = _npz_array_shape(cf, "mask")
            mw = int(mshape[1]) if mshape and len(mshape) > 1 else 1
            has_ret_chunk = "ret" in _npz_keys(cf)
            total += int(shape[0])
            obs_dims.add(int(shape[1]))
            mask_dims.add(mw)
            ret_counts["ret" if has_ret_chunk else "no_ret"] += 1
            usable.append(cf)
        except Exception as e:
            print(f"  пропуск битого чанка {cf}: {e}")
    if total == 0:
        print("WARNING: все чанки пустые")
        return
    # Разные размерности = чанки от разных версий признаков/экшн-спейса. Молча падать в
    # середине записи («could not broadcast ... into shape») или добивать нулями нельзя:
    # раскладка признаков менялась и в середину (713 -> 715 вставил два tera-признака перед
    # tera_type), поэтому паддинг справа сдвинул бы колонки и испортил данные.
    if len(obs_dims) > 1 or len(mask_dims) > 1:
        obs_part = ", ".join(f"{d}: {sum(1 for c in usable if (_npz_array_shape(c, 'obs') or (0, 0))[1] == d)} чанк(ов)"
                             for d in sorted(obs_dims))
        mask_part = ", ".join(str(d) for d in sorted(mask_dims))
        raise ValueError(
            f"чанки собраны разными версиями кода и их нельзя слить: obs_dim ({obs_part}), mask_dim ({mask_part}). "
            "Пересоберите датасет из сырого кэша текущим кодом — удалите ТОЛЬКО чанки датасета "
            f"({os.path.join(HEURISTIC_DATASET_TMP_DIR, 'dataset_chunk_*.npz')}) и запустите обучение снова: "
            "obs пересчитается из сырого кэша. Сырой кэш и --force-recollect для этого не нужны "
            "(--force-recollect ещё и стирает сырые чанки, из-за чего бои придётся собирать заново)."
        )
    obs_dim = obs_dims.pop()
    mask_dim = mask_dims.pop()
    has_ret = ret_counts["ret"] > 0
    if ret_counts["no_ret"]:
        print(f"  WARNING: в {ret_counts['no_ret']} чанк(ах) нет ret — эти примеры получат return=0")
    print(f"  Всего {total} примеров, obs_dim={obs_dim}, mask_dim={mask_dim}, has_ret={has_ret}")
    # Preallocate arrays (peak ~3.6GB for 50k, vs 61GB swap before)
    obs_arr = np.empty((total, obs_dim), dtype=np.float32)
    mask_1d = mask_dim == 1        # 1D-маску сохраняем плоской, как в исходных чанках
    mask_arr = np.empty((total,) if mask_1d else (total, mask_dim), dtype=np.int8)
    action_arr = np.empty((total,), dtype=np.int64)
    ret_arr = np.zeros((total,), dtype=np.float32) if has_ret else None

    offset = 0
    for cf in usable:
        d = np.load(cf)
        n = int(d["obs"].shape[0])
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

def _features_fingerprint() -> str:
    """Хеш кода, который считает признаки (features.py + damage.py + fusion_types.py +
    type_utils.py + config.py).

    Нужен, чтобы кэш пересчёта (`models/heuristic_dataset_tmp_recompute/_merged.npz`) не был
    использован после правок, меняющих ЗНАЧЕНИЯ признаков при той же размерности: obs_dim
    в такой ситуации совпадает, а раскладка/значения колонок уже другие.
    """
    import hashlib
    h = hashlib.md5()
    base = os.path.dirname(os.path.abspath(__file__))
    # fusion_types/type_utils тоже считают значения признаков (фьюжн-типы и безопасные
    # множители типа) — без них кэш датасета и сайдкар статистики остались бы «валидными»
    # после правки типов при той же размерности obs
    for name in ("features.py", "damage.py", "fusion_types.py", "type_utils.py", "config.py"):
        try:
            with open(os.path.join(base, name), "rb") as f:
                h.update(f.read())
        except Exception:
            h.update(name.encode())
    return h.hexdigest()


def _raw_chunks_fingerprint(chunk_files: list) -> list:
    """Отпечаток набора сырых чанков: имя + размер + mtime (меняется при доборе боёв)."""
    out = []
    for f in chunk_files:
        try:
            st = os.stat(f)
            out.append([os.path.basename(f), int(st.st_size), int(st.st_mtime_ns)])
        except Exception:
            out.append([os.path.basename(f), -1, -1])
    return out


def _store_dataset(dataset: list, path: str) -> str:
    """Сохраняет УЖЕ СОБРАННЫЙ в памяти датасет в `path`, не подмешивая чужие чанки с диска.

    Раньше здесь предпочитался мерж `models/heuristic_dataset_tmp/dataset_chunk_*.npz`, если их
    больше одного. Это ломается, когда чанки остались от прошлого прогона: в памяти лежит
    свежий (текущий N_FEATURES) датасет, а из чанков подмешиваются признаки прежних версий —
    отсюда `could not broadcast input array from shape (N,715) into shape (N,713)`, а если бы
    размерности совпали, на диск молча ушли бы устаревшие данные.

    Порядок: свежий recompute-мерж (копирование файла вместо повторного stack на ~3 ГБ),
    иначе — save_dataset из памяти.
    """
    _ensure_dir(os.path.dirname(path) or ".")
    merged = os.path.join(HEURISTIC_DATASET_TMP_DIR + "_recompute", "_merged.npz")
    try:
        if os.path.exists(merged) and len(dataset) > 0:
            dim = _get_npz_obs_dim(merged)
            n = _get_npz_n_transitions(merged)
            mem_dim = int(np.asarray(dataset[0][0]).shape[0])
            if dim == mem_dim and n == len(dataset):
                import shutil
                shutil.copyfile(merged, path)
                print(f"Датасет скопирован из recompute-мержа: {path} ({n} примеров, obs {dim})")
                return path
            print(f"  recompute-мерж не совпадает с датасетом в памяти "
                  f"(obs {dim} vs {mem_dim}, примеров {n} vs {len(dataset)}) — пишу из памяти")
    except Exception as e:
        print(f"  не удалось использовать recompute-мерж: {e}")
    save_dataset(dataset, path)
    return path


def _recompute_from_chunked_cache(n_battles: int) -> list:
    """Пересобирает датасет из chunked raw cache без новых боёв, но стримингово (по чанкам) чтобы не держать 1.2M battle_copy в памяти."""
    from .features import embed_battle_with_fusion
    from .config import N_FEATURES
    from collections import defaultdict
    chunk_files = _list_raw_chunk_files()
    if not chunk_files:
        return None
    merged_path = os.path.join(HEURISTIC_DATASET_TMP_DIR + "_recompute", "_merged.npz")
    meta_path = merged_path + ".meta.json"
    fingerprint = _raw_chunks_fingerprint(chunk_files)
    feature_hash = _features_fingerprint()
    # Готовый мерж: то же число боёв, тот же сырой кэш, тот же код признаков.
    # Иначе пересчёт 800k+ переходов повторяется с нуля (десятки минут) на каждом запуске.
    try:
        if os.path.exists(merged_path) and os.path.exists(meta_path):
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (int(meta.get("obs_dim", -1)) == int(N_FEATURES)
                    and int(meta.get("battles_requested", -1)) == int(n_battles)
                    and meta.get("chunks") == fingerprint
                    and meta.get("features_hash") == feature_hash):
                data = np.load(merged_path)
                if "ret" in data:
                    dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
                else:
                    dataset = list(zip(data["obs"], data["mask"], data["action"]))
                print(f"Recompute: готовый мерж {merged_path} подходит "
                      f"(obs {meta['obs_dim']}, {len(dataset)} примеров, код признаков тот же) — "
                      f"пересчёт не нужен")
                return dataset
            print("Recompute: готовый мерж не подходит (другие бои/код признаков) — пересобираю")
        elif os.path.exists(merged_path):
            # Сайдкара ещё нет (мерж от прежней версии кода). Принимаем его только если он
            # НОВЕЕ кода признаков: значит пересчёт делался уже текущими features/damage/config.
            # Так первый запуск после этого обновления не пересчитывает 800k переходов заново,
            # а любая правка кода признаков (или git pull) автоматически инвалидирует кэш.
            try:
                code_files = [os.path.join(os.path.dirname(os.path.abspath(__file__)), n)
                              for n in ("features.py", "damage.py", "config.py")]
                code_mtime = max(os.path.getmtime(f) for f in code_files if os.path.exists(f))
                if int(_get_npz_obs_dim(merged_path) or -1) == int(N_FEATURES) \
                        and os.path.getmtime(merged_path) > code_mtime:
                    data = np.load(merged_path)
                    if "ret" in data:
                        dataset = list(zip(data["obs"], data["mask"], data["action"], data["ret"]))
                    else:
                        dataset = list(zip(data["obs"], data["mask"], data["action"]))
                    print(f"Recompute: использую мерж {merged_path} без сайдкара "
                          f"(obs {N_FEATURES}, {len(dataset)} примеров, файл новее кода признаков)")
                    try:
                        import json
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump({"obs_dim": int(N_FEATURES), "battles_requested": int(n_battles),
                                       "chunks": fingerprint, "features_hash": feature_hash,
                                       "examples": len(dataset), "adopted_without_sidecar": True},
                                      f, ensure_ascii=False)
                    except Exception:
                        pass
                    return dataset
                print("Recompute: мерж без сайдкара старше кода признаков — пересобираю")
            except Exception as e:
                print(f"Recompute: мерж без сайдкара не удалось использовать ({e}) — пересобираю")
    except Exception as e:
        print(f"Recompute: готовый мерж не удалось использовать ({e}) — пересобираю")
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
            if len(entry) in (8, 9):
                grouped[entry[3]].append(entry)
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
            fields = _raw_entry_fields(entry)
            if fields is None:
                continue
            (battle_copy, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect,
             our_team_fusions, opp_team_fusions) = fields
            try:
                obs = embed_battle_with_fusion(
                    battle_copy, our_fusion, opp_fusion,
                    our_protected_last_turn=our_protect, opp_protected_last_turn=opp_protect,
                    our_team_fusions=our_team_fusions, opp_team_fusions=opp_team_fusions,
                )
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
    # сайдкар: по нему следующий запуск поймёт, что пересчёт можно не повторять
    try:
        import json
        with open(merged_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "obs_dim": int(N_FEATURES),
                "battles_requested": int(n_battles),
                "battles_processed": int(tag_count),
                "chunks": fingerprint,
                "features_hash": feature_hash,
                "examples": int(_get_npz_n_transitions(merged_path) or 0),
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"  не удалось записать метаданные пересчёта: {e}")
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
                            _store_dataset(dataset, path)
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
            # датасет уже собран в памяти — пишем его, а не чанки неизвестного происхождения
            _store_dataset(dataset, path)
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

def _raw_entry_fields(entry) -> tuple | None:
    """Разбирает запись сырого кэша: (battle_copy, mask, action, tag, our_fusion, opp_fusion,
    our_protect, opp_protect, our_team_fusions, opp_team_fusions).

    Записи бывают 8-элементные (до добавления командных карт) и 9-элементные; у 8-элементных
    командные карты = (None, None), то есть прежнее поведение. None — если запись не подходит.
    """
    if len(entry) < 8:
        return None
    bc, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect = entry[:8]
    teams = entry[8] if len(entry) > 8 else None
    if isinstance(teams, (tuple, list)) and len(teams) >= 2:
        our_team_fusions, opp_team_fusions = teams[0], teams[1]
    else:
        our_team_fusions, opp_team_fusions = None, None
    return bc, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect, \
        our_team_fusions, opp_team_fusions


def _recompute_dataset_from_raw(raw_dataset: list, battles: dict) -> list:
    """Пересобирает (obs, mask, action, tag) из сырого кэша с текущими признаками (N_FEATURES)."""
    from .features import embed_battle_with_fusion
    recomputed = []
    for entry in raw_dataset:
        fields = _raw_entry_fields(entry)
        if fields is None:
            continue
        (battle_copy, mask, action, tag, our_fusion, opp_fusion, our_protect, opp_protect,
         our_team_fusions, opp_team_fusions) = fields
        try:
            obs = embed_battle_with_fusion(
                battle_copy, our_fusion, opp_fusion,
                our_protected_last_turn=our_protect, opp_protected_last_turn=opp_protect,
                our_team_fusions=our_team_fusions, opp_team_fusions=opp_team_fusions,
            )
            recomputed.append((obs, mask, action, tag))
        except Exception:
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
                        if len(entry) in (8, 9):
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
    from .config import N_FEATURES as _N_FEATURES_CHECK
    N_FEATURES = _N_FEATURES_CHECK
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
        # чанки могли остаться от прежней версии признаков: такие данные нельзя отдавать в
        # обучение (колонки сдвинуты), поэтому пересчитываем obs из сырого кэша текущим кодом
        bad_dims = _wrong_dim_dataset_chunks(chunk_files, N_FEATURES)
        if bad_dims:
            print(f"  Датасет чанки старой размерности {bad_dims} (текущая {N_FEATURES}) — "
                  f"пересобираю из сырого кэша")
            return _recompute_from_chunked_cache(n_battles)
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
    elif isinstance(dataset, np.ndarray) and dataset.ndim == 2:
        # уже готовый массив obs (например выровненный по размерности признаков)
        obs_arr = np.asarray(dataset, dtype=np.float32)
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
        rms = vec_normalize.obs_rms["observation"]
        try:
            rms_dim = int(np.asarray(rms.mean).size)
            if obs_arr.shape[1] != rms_dim:
                # размерности должны совпадать: иначе RunningMeanStd.update падает на broadcast
                # (obs 418 против статистики 870). Приводим тем же правилом, что и сам BC.
                obs_arr = _pad_obs_to_features(obs_arr, rms_dim, label="датасет для warm-up статистики")
        except ValueError:
            raise
        except Exception as e:
            print(f"BC: warm-up статистики пропущен ({e})")
            return
        rms.update(np.asarray(obs_arr, dtype=np.float32))

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

def eval_normalizer_for(ppo):
    """Нормализатор obs для боёв за винрейт: РОВНО та статистика, что видит обучение.

    Вынесено из evaluate_win_rates отдельной функцией, чтобы это проверялось тестом без
    сервера. Регрессия: здесь стоял `target_dim=N_FEATURES` без импорта — NameError глотался
    `except Exception`, нормализация молча выключалась, и модель в боях оценки играла на
    СЫРЫХ признаках (другие условия, чем обучение): винрейт и поведение (свитчи/тера) не
    сопоставимы с `[mix]`. Теперь про сбой пишется явно.
    """
    vec_norm = ppo.get_vec_normalize_env() if hasattr(ppo, "get_vec_normalize_env") else None
    if vec_norm is not None:
        try:
            from .vecnorm_utils import LiveVecNormalizeAdapter
            normalizer = LiveVecNormalizeAdapter(vec_norm, target_dim=N_FEATURES)
        except Exception as e:
            normalizer = None
            print(f"НЕ УДАЛОСЬ включить нормализацию для eval ({type(e).__name__}: {e}) — "
                  f"пробую статистику с диска")
        if normalizer is not None:
            why = ""
            try:
                why = f" ({normalizer.describe()})"
            except Exception:
                pass
            print(f"eval: нормализация obs из живого VecNormalize (как в обучении){why}")
            return normalizer

    if os.environ.get("PYBOT_SELF_PLAY_NORM", "1") == "0":
        print("eval: нормализация obs отключена (PYBOT_SELF_PLAY_NORM=0)")
        return None
    # запасной путь: живой VecNormalize недоступен (например, оценка идёт из другого места) —
    # берём статистику с диска. НЕ применяем, если обучение идёт без нормализации
    # (--no-normalize-bc): иначе eval окажется в других условиях, чем обучение.
    try:
        from .config import VECNORM_PATH
        from .vecnorm_utils import load_vecnorm_stats
        if os.path.isfile(VECNORM_PATH):
            normalizer = load_vecnorm_stats(VECNORM_PATH, N_FEATURES)
            if normalizer is not None:
                try:
                    why = f" ({normalizer.describe()})"
                except Exception:
                    why = ""
                print(f"eval: нормализация obs из {VECNORM_PATH}{why}")
                return normalizer
    except Exception as e:
        print(f"eval: статистику с диска взять не удалось ({type(e).__name__}: {e})")
    print("ВНИМАНИЕ: eval играет БЕЗ нормализации obs — если обучение шло с нормализацией, "
          "винрейт и поведение модели в боях НЕ сопоставимы с [mix]")
    return None


def evaluate_win_rates(ppo, n_battles: int = 180) -> dict[str, float]:
    # Нормализация obs: ровно те же статистики, что видит обучение (живой VecNormalize),
    # через штатный параметр PolicyPlayer — см. eval_normalizer_for().
    normalizer = eval_normalizer_for(ppo)
    base_agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT,
                              max_concurrent_battles=30, obs_normalizer=normalizer)
    # считаем реальные решения модели в оценочных боях (без wait-шагов, как в [mix])
    base_agent.action_counter = {"switch": 0, "switch_forced": 0, "move": 0, "tera": 0, "other": 0}

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
                eval_sp = PolicyPlayer(policy=getattr(base_sp, "policy", None), battle_format=BATTLE_FORMAT,
                                       max_concurrent_battles=30, obs_normalizer=normalizer)
                opponents.append(eval_sp)
            except Exception:
                opponents.append(base_sp)
            n_sp_appended = 1
    except Exception:
        sp_opps = []
        n_sp_appended = 0

    asyncio.run(base_agent.battle_against(*opponents, n_battles=n_battles))
    try:
        cnt = base_agent.action_counter or {}
        n_dec = sum(int(v) for v in cnt.values())
        if n_dec:
            own = int(cnt.get("switch", 0))
            forced = int(cnt.get("switch_forced", 0))
            print(f"eval-микс (решения модели в оценочных боях): "
                  f"приём {cnt.get('move', 0) / n_dec * 100:.1f}%, "
                  f"тера {cnt.get('tera', 0) / n_dec * 100:.1f}%, "
                  f"свитч {(own + forced) / n_dec * 100:.1f}% "
                  f"(своих {own} = {own / n_dec * 100:.1f}%, вынужденных после фейнта {forced} = "
                  f"{forced / n_dec * 100:.1f}%) из {n_dec} решений")
    except Exception:
        pass
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

# Датасет, собранный на этой или более новой раскладке, добивается нулями корректно: все
# изменения признаков начиная с 715 добавлялись ТОЛЬКО В ХВОСТ (блок урона DAMAGE_BLOCK_SIZE
# идёт последним, см. features._damage_block). А вот 713 -> 715 вставил [our_is_tera,
# opp_is_tera] ПЕРЕД tera_type, то есть в СЕРЕДИНУ: у датасета на 713 и старее колонки после
# места вставки — уже другие признаки, и добивание нулями молча сдвинуло бы их все.
MIN_PREFIX_OBS_DIM = 715


def dataset_layout_error(obs_dim: int, target_dim: int, *, label: str = "датасет") -> str:
    """Текст ошибки для датасета, который нельзя добить нулями (раскладка менялась в середине)."""
    return (
        f"{label} собран на старой раскладке признаков: obs {obs_dim}, а сейчас {target_dim}.\n"
        f"Добить его нулями нельзя: раскладка менялась не только в хвост — при переходе 713 -> 715\n"
        f"два признака террорализации были вставлены в СЕРЕДИНУ (перед tera_type), поэтому\n"
        f"старые колонки — уже другие признаки, и модель училась бы на чужих значениях.\n"
        f"Что делать: пересобрать датасет текущим кодом (obs пересчитываются из сырых боёв\n"
        f"автоматически) — например прогоном сборщика датасета (build_heuristic_dataset.py /\n"
        f"collect_heuristic.py), либо обучением без --dataset-path (--pretrain-battles N).\n"
        f"Проверить свой файл: python -c \"import numpy as np; d = np.load('PATH');\n"
        f"print(d['obs'].shape, 'ret' in d.files)\""
    )


def validate_bc_dataset(path: str, target_dim: int) -> dict:
    """Быстрая проверка датасета для BC (до создания env): размерность, наличие ret.

    Возвращает {"obs_dim", "has_ret", "examples"} или бросает SystemExit с человеческим текстом.
    Раньше отсутствие ret и старая раскладка выяснялись уже внутри BC — после того как
    поднимались 8 env-процессов, а старая раскладка вообще приводила к тихой порче колонок.
    """
    import numpy as _np

    try:
        with _np.load(path, mmap_mode="r") as data:
            keys = set(data.files)
            obs_dim = int(data["obs"].shape[1])
            examples = int(data["obs"].shape[0])
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Не удалось прочитать датасет {path}: {e}")
    if "ret" not in keys:
        raise SystemExit(
            f"В датасете {path} нет 'ret' (возвратов) — BC по нему невозможен.\n"
            f"Датасет нужно пересобрать сборщиком репозитория: он считает возвраты\n"
            f"(_compute_bc_returns) и кладёт их в тот же .npz. Найденные ключи: {sorted(keys)}"
        )
    if obs_dim != int(target_dim) and obs_dim < MIN_PREFIX_OBS_DIM:
        raise SystemExit(dataset_layout_error(obs_dim, target_dim, label=f"датасет {path}"))
    return {"obs_dim": obs_dim, "has_ret": True, "examples": examples}


def _pad_obs_to_features(obs_arr, target_dim: int, *, label: str = "датасет",
                         memmap_threshold: int = 200_000):
    """Добивает obs старого датасета нулями до target_dim (новые признаки = «нет информации»).

    Допустимо только для раскладок, где все изменения шли в хвост (>= MIN_PREFIX_OBS_DIM):
    датасеты на 713 и старее добивать нельзя (см. dataset_layout_error).
    """
    import numpy as _np

    old_dim = int(obs_arr.shape[1])
    if old_dim == int(target_dim):
        return obs_arr
    if old_dim > int(target_dim):
        print(f"BC: {label} шире текущих признаков ({old_dim} > {target_dim}) — обрезаю хвост")
        return _np.ascontiguousarray(obs_arr[:, :int(target_dim)])
    if old_dim < MIN_PREFIX_OBS_DIM:
        raise ValueError(dataset_layout_error(old_dim, int(target_dim), label=label))
    n = int(obs_arr.shape[0])
    print(f"BC: {label} старой размерности ({old_dim} < {target_dim}) — добиваю нулями "
          f"({int(target_dim) - old_dim} новых признаков)")
    if n > int(memmap_threshold):
        import tempfile as _tempfile
        tmp = _tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        tmp.close()
        out = _np.lib.format.open_memmap(tmp.name, mode="w+", dtype=_np.float32,
                                         shape=(n, int(target_dim)))
        chunk = 50_000
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out[s:e] = 0.0
            out[s:e, :old_dim] = _np.asarray(obs_arr[s:e], dtype=_np.float32)
        out.flush()
        return out
    out = _np.zeros((n, int(target_dim)), dtype=_np.float32)
    out[:, :old_dim] = _np.asarray(obs_arr, dtype=_np.float32)
    return out


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
        obs_arr = _pad_obs_to_features(obs_arr, N_FEATURES)
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
            obs_arr = _pad_obs_to_features(obs_arr, N_FEATURES)
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
            obs_arr = _pad_obs_to_features(obs_arr, N_FEATURES)
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
            # warm-up должен видеть ровно те obs, на которых пойдёт BC (obs_arr уже
            # выровнен по размерности признаков), иначе статистика окажется чужой размерности
            warm_up_vec_normalize(vec_normalize, obs_arr if obs_arr is not None else warm_arg)
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
