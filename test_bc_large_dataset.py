#!/usr/bin/env python3
"""BC на большом датасете: не падать по памяти и не терять прогресс.

Повод: прогон `--dataset-path models/mixed.npz` (3 271 527 x 991) обрывался БЕЗ traceback
сразу после создания BC-оптимизатора. Причина — две аллокации на весь датасет:
  * `warm_up_vec_normalize` звал SB3-шный `RunningMeanStd.update(arr)`, который начинается
    с `arr.astype(np.float64)` = ~26 ГБ на таком датасете;
  * нормализация делала `np.empty(obs_arr.shape, float32)` = ещё ~13 ГБ.
На Windows такой пик упирается в коммит-лимит, и процесс убивается без исключения.

Что проверяем (без сервера, только torch/SB3/numpy):
  A) `warm_up_vec_normalize` считает статистику ЧАНКАМИ: в `RunningMeanStd.update` никогда не
     попадает больше `chunk_rows` строк, а результат совпадает с «одним куском»;
  B) режим `on_the_fly`: нормализованный датасет НЕ материализуется (запрет аллокаций > 100 МБ),
     obs остаются memmap, а нормализация батча даёт ровно тот же результат, что in_memory;
  C) `auto` выбирает режим по порогу `BC_MATERIALIZE_LIMIT_BYTES`;
  D) `--bc-max-examples` реально урезает число батчей;
  E) прогресс печатается с flush и содержит ETA;
  F) чекпоинт после каждой эпохи создаётся и грузится;
  G) MemoryError / KeyboardInterrupt внутри цикла не убивают прогон: сообщение, применение
     лучших весов, `interrupted` вместо исключения.

Запуск: PYTHONPATH=. python test_bc_large_dataset.py
"""
import io
import os
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from agents.config import N_FEATURES

OK = 0
FAIL: list = []


def check(name, cond, extra=""):
    global OK
    if cond:
        OK += 1
        print(f"OK   {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL.append(name)
        print(f"FAIL {name}" + (f": {extra}" if extra else ""))


# --------------------------------------------------------------- общие хелперы ---
def make_vec_env(action_dim=26):
    """DummyVecEnv (gymnasium) + VecNormalize: как в run() перед BC."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gymnasium import spaces

    class _Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(action_dim,), dtype=np.int8),
            })
            self.action_space = spaces.Discrete(action_dim)

        def reset(self, **kwargs):
            return {"observation": np.zeros(N_FEATURES, np.float32),
                    "action_mask": np.ones(action_dim, np.int8)}, {}

        def step(self, action):
            return {"observation": np.zeros(N_FEATURES, np.float32),
                    "action_mask": np.ones(action_dim, np.int8)}, 0.0, True, False, {}

    venv = VecNormalize(DummyVecEnv([_Env]), norm_obs=True, norm_reward=False,
                        clip_obs=10.0, training=True, norm_obs_keys=["observation"])
    return venv


def make_ppo(vec_env, features_dim=64):
    from stable_baselines3 import PPO
    from agents.policy import MaskedActorCriticPolicy
    from agents.optim import SplitAdam

    return PPO(MaskedActorCriticPolicy, vec_env, device="cpu", n_steps=8, batch_size=8,
               n_epochs=1, verbose=0,
               policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=int(features_dim)),
                                  net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                  optimizer_class=SplitAdam))


def write_dataset(path: str, n: int, seed: int = 0, compressed: bool = False):
    """Датасет нужной формы: > 200 МБ файла, чтобы pretrain пошёл по mmap-пути."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    mask = np.ones((n, 26), dtype=np.int8)
    action = rng.integers(0, 26, size=n).astype(np.int64)
    ret = rng.normal(size=n).astype(np.float32)
    (np.savez_compressed if compressed else np.savez)(path, obs=obs, mask=mask,
                                                      action=action, ret=ret)
    return path


# ------------------------------------------------------------------ A: warm-up ---
def part_a_warmup_chunked():
    print("=" * 78)
    print("A. warm_up_vec_normalize: статистика чанками, без float64-копии датасета")
    print("=" * 78)
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from agents.training import warm_up_vec_normalize

    n, dim = 20_000, 32
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    # ровно так датасет приходит в warm_up из pretrain_policy_bc: mmap-вид obs
    tmp_dir = tempfile.mkdtemp(prefix="bc_warm_")
    npz = os.path.join(tmp_dir, "obs.npz")
    np.savez(npz, obs=data)
    data_mmap = np.load(npz, mmap_mode="r")["obs"]

    class _FakeVecNorm:
        def __init__(self):
            self.obs_rms = {"observation": RunningMeanStd(shape=(dim,))}

    # эталон: один большой update
    ref = _FakeVecNorm()
    ref.obs_rms["observation"].update(np.asarray(data_mmap, dtype=np.float32))
    ref_mean = ref.obs_rms["observation"].mean.copy()
    ref_var = ref.obs_rms["observation"].var.copy()
    ref_count = ref.obs_rms["observation"].count

    # наш путь: чанками, фиксируем максимальный размер куска, попадающего в update
    got = _FakeVecNorm()
    seen_rows = []
    orig_update = RunningMeanStd.update

    def _spy(self, arr):
        seen_rows.append(int(arr.shape[0]))
        return orig_update(self, arr)

    RunningMeanStd.update = _spy
    try:
        warm_up_vec_normalize(got, data_mmap, chunk_rows=2_000, verbose=False)
    finally:
        RunningMeanStd.update = orig_update

    check("A: update вызывается ≤ chunk_rows строк (датасет не материализуется)",
          bool(seen_rows) and max(seen_rows) <= 2_000, f"макс. кусок {max(seen_rows)} строк")
    check("A: число кусков соответствует размеру датасета", len(seen_rows) == 10,
          f"{len(seen_rows)} кусков по 2000")
    check("A: count совпадает с одним куском", int(got.obs_rms["observation"].count) == int(ref_count),
          f"{got.obs_rms['observation'].count} vs {ref_count}")
    d_mean = float(np.abs(got.obs_rms["observation"].mean - ref_mean).max())
    d_var = float(np.abs(got.obs_rms["observation"].var - ref_var).max())
    # допуск на порядок: RunningMeanStd объединяет моменты по чанкам, это не бит-в-бит
    # (mean съезжает на ~3e-8, var на ~5e-6 при var~1 — на нормализацию не влияет)
    check("A: mean совпадает (max|Δ| < 1e-6)", d_mean < 1e-6, f"{d_mean:.2e}")
    check("A: var совпадает (max|Δ| < 1e-4)", d_var < 1e-4, f"{d_var:.2e}")


# --------------------------------------------------- B/C/D/E: настоящий прогон ---
def part_b_on_the_fly(td: str):
    print("=" * 78)
    print("B. on_the_fly: датасет не копируется в RAM, нормализация батчем")
    print("=" * 78)
    from agents import training as tr

    n = 60_000          # 238 МБ файла -> mmap-путь, obs 238 МБ
    ds = write_dataset(os.path.join(td, "big.npz"), n)
    check("B: файл датасета > 200 МБ (иначе путь не mmap)", os.path.getsize(ds) > 200_000_000,
          f"{os.path.getsize(ds) / 1e6:.0f} МБ")

    venv = make_vec_env()
    ppo = make_ppo(venv)

    # (1) страховка от исходного бага: запрещаем аллокации больше 100 МБ
    real_empty = np.empty
    big_allocs = []

    def _guard_empty(shape, *a, **kw):
        size = int(np.prod(shape)) * 4
        if size > 100_000_000:
            big_allocs.append((shape, size))
            raise AssertionError(f"датасет материализуется в RAM: np.empty({shape}) ~{size / 1e9:.2f} ГБ")
        return real_empty(shape, *a, **kw)

    tr.BC_MATERIALIZE_LIMIT_BYTES = 100_000_000      # auto -> on_the_fly для этого датасета
    np.empty = _guard_empty
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            tr.pretrain_policy_bc(ppo, ds, epochs=1, batch_size=512, normalize=True,
                                  reset_at_start=True, reset_optimizer=False,
                                  progress_every=50, verbose=True, patience=99)
    finally:
        np.empty = real_empty
        tr.BC_MATERIALIZE_LIMIT_BYTES = 1_500_000_000
    out = buf.getvalue()
    check("B: ни одной аллокации > 100 МБ (датасет не материализуется)", not big_allocs,
          str(big_allocs[:1]))
    check("B: в логе режим on_the_fly", "режим on_the_fly" in out,
          [l for l in out.splitlines() if "режим" in l][:1])
    check("B: в логе «нормализуются на лету»", "нормализуются на лету" in out)

    # (2) численная эквивалентность: чем нормализует BC (VecNormStats) == формула SB3
    from agents.training import bc_obs_normalizer
    data = np.asarray(np.load(ds, mmap_mode="r")["obs"][:1000], dtype=np.float32)
    on_fly = bc_obs_normalizer(venv).normalize(data)
    sb3 = venv.normalize_obs({"observation": data})["observation"]
    check("B: нормализация BC == VecNormalize.normalize_obs (max|Δ| = 0)",
          float(np.abs(on_fly - sb3).max()) == 0.0, f"{float(np.abs(on_fly - sb3).max()):.2e}")

    # (2b) устойчивость: VecNormalize с ключами по умолчанию (весь Dict) не роняет on_the_fly
    venv_keys = make_vec_env()
    venv_keys.norm_obs_keys = None
    try:
        norm2 = bc_obs_normalizer(venv_keys).normalize(data)
        check("B: нормализатор не зависит от norm_obs_keys (нет KeyError: action_mask)",
              norm2.shape == data.shape, f"{norm2.shape}")
    except Exception as e:                                  # noqa: BLE001
        check("B: нормализатор не зависит от norm_obs_keys (нет KeyError: action_mask)",
              False, f"{type(e).__name__}: {e}")

    # (3) auto при большом пороге -> in_memory (поведение как раньше)
    venv2 = make_vec_env()
    ppo2 = make_ppo(venv2)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        tr.pretrain_policy_bc(ppo2, os.path.join(td, "small.npz") if False else ds,
                              epochs=1, batch_size=4096, normalize=True,
                              reset_at_start=True, reset_optimizer=False,
                              progress_every=0, verbose=True, patience=99)
    out2 = buf2.getvalue()
    check("C: auto с порогом 1.5 ГБ выбирает in_memory для 0.24 ГБ датасета",
          "режим in_memory" in out2, [l for l in out2.splitlines() if "режим" in l][:1])

    # (4) явный режим on_the_fly для маленького датасета тоже работает
    venv3 = make_vec_env()
    ppo3 = make_ppo(venv3)
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        tr.pretrain_policy_bc(ppo3, ds, epochs=1, batch_size=8192, normalize=True,
                              normalize_mode="on_the_fly", reset_at_start=True,
                              reset_optimizer=False, progress_every=0, patience=99)
    check("C: явный --bc-normalize-mode on_the_fly уважается",
          "режим on_the_fly" in buf3.getvalue())

    # (5) max_examples урезает число батчей
    venv4 = make_vec_env()
    ppo4 = make_ppo(venv4)
    buf4 = io.StringIO()
    with redirect_stdout(buf4):
        tr.pretrain_policy_bc(ppo4, ds, epochs=1, batch_size=512, normalize=False,
                              max_examples=5_120, reset_at_start=True, reset_optimizer=False,
                              progress_every=0, patience=99)
    check("D: --bc-max-examples урезает выборку (5120/512 = 10 батчей)",
          "подвыборка" in buf4.getvalue(), [l for l in buf4.getvalue().splitlines()
                                            if "подвыборка" in l][:1])

    # (6) прогресс с ETA
    venv5 = make_vec_env()
    ppo5 = make_ppo(venv5)
    buf5 = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf5):
        tr.pretrain_policy_bc(ppo5, ds, epochs=1, batch_size=1024, normalize=False,
                              reset_at_start=True, reset_optimizer=False,
                              progress_every=10, patience=99)
    spent = time.time() - t0
    lines = [l for l in buf5.getvalue().splitlines() if l.startswith("BC epoch 1/1: батч")]
    check("E: прогресс печатается каждые N батчей", len(lines) >= 5, f"{len(lines)} строк, {spent:.1f} с")
    check("E: в прогрессе есть ETA и проценты",
          bool(lines) and "ETA эпохи" in lines[0] and "%" in lines[0], lines[0][:110] if lines else "")

    # (7) чекпоинт после каждой эпохи
    ckpt = os.path.join(td, "bc_latest.zip")
    venv6 = make_vec_env()
    ppo6 = make_ppo(venv6)
    with redirect_stdout(io.StringIO()):
        tr.pretrain_policy_bc(ppo6, ds, epochs=2, batch_size=8192, normalize=False,
                              reset_at_start=True, reset_optimizer=False, progress_every=0,
                              checkpoint_path=ckpt, patience=99)
    check("F: чекпоинт после эпохи создан", os.path.exists(ckpt),
          f"{os.path.getsize(ckpt) / 1e6:.1f} МБ" if os.path.exists(ckpt) else "нет файла")
    from stable_baselines3 import PPO as _PPO
    ok_load = False
    try:
        _PPO.load(ckpt, device="cpu")
        ok_load = True
    except Exception as e:
        ok_load = f"{type(e).__name__}: {e}"
    check("F: чекпоинт грузится SB3", ok_load is True, str(ok_load)[:80])


# --------------------------------------------- G: обрыв по памяти / Ctrl+C ---
def part_g_interrupts(td: str):
    print("=" * 78)
    print("G. MemoryError и Ctrl+C внутри цикла: сообщение вместо тихой смерти")
    print("=" * 78)
    from agents import training as tr

    ds = write_dataset(os.path.join(td, "small.npz"), 4_000, compressed=True)

    def run_with_raise(exc, at_batch=3):
        venv = make_vec_env()
        ppo = make_ppo(venv)
        calls = {"n": 0}
        real_extract = ppo.policy.extract_features

        def _boom(obs):
            calls["n"] += 1
            if calls["n"] >= at_batch:
                raise exc
            return real_extract(obs)

        ppo.policy.extract_features = _boom
        buf = io.StringIO()
        raised = None
        with redirect_stdout(buf):
            try:
                tr.pretrain_policy_bc(ppo, ds, epochs=3, batch_size=512, normalize=False,
                                      reset_at_start=True, reset_optimizer=False,
                                      progress_every=0, patience=99)
            except BaseException as e:                      # noqa: BLE001 - тест ловит всё
                raised = e
        # снимаем подмену, иначе она «утечёт» в последующие проверки
        try:
            del ppo.policy.extract_features
        except AttributeError:
            pass
        return buf.getvalue(), raised, ppo

    out, raised, ppo = run_with_raise(MemoryError("simulated oom"))
    check("G: MemoryError не вылетает наружу", raised is None, repr(raised))
    check("G: есть понятное сообщение про память",
          "не хватило памяти" in out and "--bc-max-examples" in out,
          [l for l in out.splitlines() if "не хватило памяти" in l][:1])
    check("G: при обрыве применяются лучшие веса / прогон доходит до конца",
          "веса" in out.lower() and len(ppo.policy.state_dict()) > 0)

    out2, raised2, ppo2 = run_with_raise(KeyboardInterrupt(), at_batch=2)
    check("G: Ctrl+C не вылетает наружу", raised2 is None, repr(raised2))
    check("G: есть сообщение «прервано пользователем»", "прервано пользователем" in out2,
          [l for l in out2.splitlines() if "прервано" in l][:1])
    check("G: после обрыва политика жива и считает forward",
          ppo2.policy({"observation": torch.zeros(1, N_FEATURES),
                       "action_mask": torch.ones(1, 26, dtype=torch.int8)}) is not None)


def anon_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def part_i_npz_without_ram(td: str):
    print("=" * 78)
    print("I. .npz не грузится в RAM целиком: заголовок, memmap-кэш, эквивалентность")
    print("=" * 78)
    from agents import training as tr

    # сжатый npz: член obs.npy занимает 238 МБ в RAM, а файл — в разы меньше
    n, dim = 60_000, N_FEATURES
    ds = os.path.join(td, "compressed.npz")
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(n, dim)).astype(np.float32)
    mask = np.ones((n, 26), np.int8)
    action = rng.integers(0, 26, size=n).astype(np.int64)
    ret = rng.normal(size=n).astype(np.float32)
    np.savez_compressed(ds, obs=obs, mask=mask, action=action, ret=ret)
    raw_bytes = n * dim * 4
    check("I: obs в датасете = 0.24 ГБ, файл меньше (numpy всё равно читает член целиком)",
          os.path.getsize(ds) <= raw_bytes, f"{os.path.getsize(ds) / 1e6:.0f} МБ файл, "
                                           f"{raw_bytes / 1e9:.2f} ГБ obs")

    # (1) метаданные читаются из заголовка, без подъёма массива
    before = anon_mb()
    info = tr.npz_member_info(ds, "obs.npy")
    after = anon_mb()
    check("I: npz_member_info отдаёт форму без чтения данных",
          info and tuple(info["shape"]) == (n, dim), str(info and info["shape"]))
    check("I: на метаданные потрачено < 20 МБ анонимной памяти (obs = 238 МБ)",
          (after - before) < 20, f"{after - before:.0f} МБ")

    # (2) validate_bc_dataset тоже не читает obs
    before = anon_mb()
    meta = tr.validate_bc_dataset(ds, dim)
    after = anon_mb()
    check("I: validate_bc_dataset берёт размеры из заголовка",
          meta["obs_dim"] == dim and meta["examples"] == n, str(meta))
    check("I: validate_bc_dataset не поднял obs в RAM (< 20 МБ)",
          (after - before) < 20, f"{after - before:.0f} МБ")

    # (3) потоковый кэш: пишется без больших аллокаций, читается как memmap, данные совпадают
    real_empty = np.empty
    big = []

    def _guard(shape, *a, **kw):
        sz = int(np.prod(shape)) * 4
        if sz > 100_000_000:
            big.append((shape, sz))
            raise AssertionError(f"аллокация {sz / 1e9:.2f} ГБ в RAM при выгрузке obs")
        return real_empty(shape, *a, **kw)

    before = anon_mb()
    np.empty = _guard
    try:
        cache = tr.ensure_obs_memmap(ds, verbose=False)
    finally:
        np.empty = real_empty
    after = anon_mb()
    check("I: кэш obs создан", bool(cache) and os.path.exists(cache), str(cache))
    check("I: выгрузка без аллокаций > 100 МБ (потоком)", not big, str(big[:1]))
    check("I: пик анонимной памяти при выгрузке < 60 МБ (obs 238 МБ)",
          (after - before) < 60, f"{after - before:.0f} МБ")
    mm = np.load(cache, mmap_mode="r")
    check("I: кэш читается как memmap", isinstance(mm, np.memmap), type(mm).__name__)
    check("I: данные кэша совпадают с npz бит-в-бит",
          bool(np.array_equal(np.asarray(mm[:500]), obs[:500])) and tuple(mm.shape) == (n, dim),
          f"{tuple(mm.shape)}")

    # (4) повторный вызов переиспользует кэш (по метаданным)
    mtime = os.path.getmtime(cache)
    cache2 = tr.ensure_obs_memmap(ds, verbose=False)
    check("I: повторный вызов переиспользует кэш (не перезаписывает)",
          cache2 == cache and os.path.getmtime(cache) == mtime, str(cache2))

    # (5) BC на сжатом npz идёт через memmap и даёт тот же loss, что путь «в RAM»
    def run_bc(limit_bytes, obs_cache="on"):
        # сид ДО создания политики: иначе случайная инициализация голов даст разные лоссы
        np.random.seed(0)
        torch.manual_seed(0)
        venv = make_vec_env()
        ppo = make_ppo(venv)
        tr.BC_NPZ_RAM_LIMIT_BYTES = limit_bytes
        buf = io.StringIO()
        np.random.seed(0)
        torch.manual_seed(0)          # dropout в экстракторе: без сида лоссы не сравнить
        try:
            with redirect_stdout(buf):
                tr.pretrain_policy_bc(ppo, ds, epochs=1, batch_size=4096, normalize=False,
                                      reset_at_start=True, reset_optimizer=False,
                                      progress_every=0, patience=99, obs_cache=obs_cache)
        finally:
            tr.BC_NPZ_RAM_LIMIT_BYTES = 1_500_000_000
        line = [l for l in buf.getvalue().splitlines() if l.startswith("[BC epoch 0]")]
        return (line[0] if line else ""), buf.getvalue()

    # с кэшем: obs — memmap (в логе видно «использую memmap-кэш»)
    loss_memmap, out_memmap = run_bc(limit_bytes=10_000_000, obs_cache="on")
    check("I: BC на большом obs идёт через memmap-кэш",
          "memmap" in out_memmap or "memmap-кэш" in out_memmap,
          [l for l in out_memmap.splitlines() if "memmap" in l][:1])
    # без кэша: obs читается в RAM (как раньше)
    loss_ram, out_ram = run_bc(limit_bytes=10 ** 12, obs_cache="off")
    check("I: путь 'в RAM' тоже работает", bool(loss_ram), loss_ram[:80])
    check("I: loss эпохи совпадает бит-в-бит (данные те же, путь разный)",
          loss_memmap == loss_ram, f"memmap={loss_memmap[:60]!r} ram={loss_ram[:60]!r}")


def part_h_cli_wiring():
    print("=" * 78)
    print("H. CLI: новые флаги существуют, парсятся и доезжают до run()")
    print("=" * 78)
    import inspect
    from agents import policy_player as pp

    args = pp.parse_args(["--dataset-path", "models/mixed.npz", "--epochs", "15",
                          "--total-timesteps", "0", "--skip-eval", "--save-as", "pretrained_v2",
                          "--bc-normalize-mode", "on_the_fly", "--bc-max-examples", "300000",
                          "--bc-progress-every", "50", "--no-bc-checkpoint"])
    check("H: --bc-normalize-mode распознан", args.bc_normalize_mode == "on_the_fly",
          args.bc_normalize_mode)
    check("H: --bc-max-examples распознан", int(args.bc_max_examples) == 300_000,
          str(args.bc_max_examples))
    check("H: --bc-progress-every распознан", int(args.bc_progress_every) == 50,
          str(args.bc_progress_every))
    check("H: --no-bc-checkpoint выключает страховку",
          args.bc_checkpoint_every_epoch is False)
    check("H: по умолчанию страховочный чекпоинт включён",
          pp.parse_args([]).bc_checkpoint_every_epoch is True)
    check("H: --bc-obs-cache off выключает memmap-кэш obs",
          pp.parse_args(["--bc-obs-cache", "off"]).bc_obs_cache == "off")

    sig = inspect.signature(pp.run)
    for name in ("bc_normalize_mode", "bc_obs_cache", "bc_max_examples", "bc_progress_every",
                 "bc_checkpoint_every_epoch", "bc_checkpoint_path"):
        check(f"H: run() принимает {name}", name in sig.parameters)

    # run() должен передавать эти параметры в pretrain_policy_bc
    src = inspect.getsource(pp.run)
    check("H: run() прокидывает режим нормализации в BC",
          "normalize_mode=bc_normalize_mode" in src and "max_examples=bc_max_examples" in src,
          "normalize_mode/max_examples")


def part_j_cli_path_load_dataset(td):
    """Клиентский маршрут (--dataset-path): load_dataset -> _DatasetView -> pretrain.

    Ровно здесь и терялась память: `load_dataset` проверял `data["obs"].shape[0]`, а numpy для
    .npz читает член ЦЕЛИКОМ — на датасете пользователя это 13 ГБ ещё до BC.
    """
    print("=" * 78)
    print("J. CLI-маршрут: load_dataset не читает obs, view с путём идёт через memmap")
    print("=" * 78)
    from agents import training as tr

    n, dim = 30_000, 991
    ds = os.path.join(td, "cli_route.npz")
    np.random.seed(3)
    obs = np.random.rand(n, dim).astype(np.float32)
    mask = np.ones((n, 26), np.int8)
    act = np.zeros(n, np.int64)
    ret = np.ones(n, np.float32)
    np.savez_compressed(ds, obs=obs, mask=mask, action=act, ret=ret)
    obs_bytes = n * dim * 4

    # (1) load_dataset на большом файле: ни obs, ни лишней памяти
    before = anon_mb()
    buf = io.StringIO()
    keep_rows = tr.VIEW_MIN_ROWS
    tr.VIEW_MIN_ROWS = 10_000          # 30k строк достаточно, чтобы view-путь включился
    try:
        with redirect_stdout(buf):
            loaded = tr.load_dataset(ds)
    finally:
        tr.VIEW_MIN_ROWS = keep_rows
    grew = anon_mb() - before
    check("J: load_dataset вернул ленивый view для большого датасета",
          isinstance(loaded, tr._DatasetView), type(loaded).__name__)
    check(f"J: load_dataset не поднял obs в RAM (obs {obs_bytes / 1e9:.2f} ГБ)",
          grew < 0.05 * obs_bytes / 1e6, f"+{grew:.0f} МБ")
    check("J: len(view) известен без чтения членов", len(loaded) == n, str(len(loaded)))
    check("J: view знает путь к источнику", loaded.path == ds, str(loaded.path))

    # (2) BC по view (как в run()) не читает obs.npz, а идёт через memmap-кэш
    # пороги опускаем, чтобы путь был ровно тот, что нужен большому датасету
    tr.BC_NPZ_RAM_LIMIT_BYTES = 1_000_000
    tr.BC_MATERIALIZE_LIMIT_BYTES = 1_000_000
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        venv = make_vec_env()
        ppo = make_ppo(venv)
        buf = io.StringIO()
        with redirect_stdout(buf):
            tr.pretrain_policy_bc(ppo, loaded, epochs=1, batch_size=4096, normalize=True,
                                  normalize_mode="auto", reset_at_start=True,
                                  reset_optimizer=False, progress_every=0, patience=99)
        log = buf.getvalue()
        cache = os.path.splitext(ds)[0] + "_bc_obs.npy"
        rms = ppo.get_vec_normalize_env().obs_rms["observation"]
        mm_now = isinstance(tr.bc_obs_array(loaded)[0], np.memmap)
    finally:
        tr.BC_NPZ_RAM_LIMIT_BYTES = 1_500_000_000
        tr.BC_MATERIALIZE_LIMIT_BYTES = 500_000_000
    check("J: BC по view выгрузил obs в memmap-кэш (а не в RAM)",
          os.path.exists(cache) and "memmap" in log,
          [l for l in log.splitlines() if "memmap" in l][:1])
    check("J: статистика нормализации посчитана по всему датасету (не по выборке)",
          int(getattr(rms, "count", 0)) >= n - 1, f"count={int(getattr(rms, 'count', 0))}")
    check("J: obs нормализуются на лету (датасет не копируется)",
          "on_the_fly" in log, [l for l in log.splitlines() if "нормализация obs" in l][:1])
    check("J: view после BC отдаёт те же obs (memmap-кэш, без чтения npz)",
          mm_now, type(tr.bc_obs_array(loaded)[0]).__name__)

    # (3) warm_up напрямую по пути: тоже без чтения obs в RAM
    np.random.seed(0)
    venv2 = make_vec_env()
    before = anon_mb()
    buf = io.StringIO()
    with redirect_stdout(buf):
        tr.warm_up_vec_normalize(venv2, ds, chunk_rows=2_000, verbose=False)
    grew = anon_mb() - before
    check(f"J: warm_up по пути не читает obs в RAM (obs {obs_bytes / 1e9:.2f} ГБ)",
          grew < 0.05 * obs_bytes / 1e6, f"+{grew:.0f} МБ")
    check("J: warm_up посчитал mean/var по всем строкам",
          float(venv2.obs_rms["observation"].count) >= n - 1,
          f"count={float(venv2.obs_rms['observation'].count):.0f}")

    # (4) фолбэк: без места под кэш флаг off оставляет прежний путь (obs в RAM), но это видно в логе
    buf = io.StringIO()
    tr.BC_NPZ_RAM_LIMIT_BYTES = 1_000_000
    try:
        with redirect_stdout(buf):
            obs_off, is_mm = tr.bc_obs_array(ds, obs_cache="off", verbose=False)
    finally:
        tr.BC_NPZ_RAM_LIMIT_BYTES = 1_500_000_000
    check("J: --bc-obs-cache off остаётся прежним путём (obs в RAM)",
          isinstance(obs_off, np.ndarray) and not isinstance(obs_off, np.memmap) and not is_mm,
          type(obs_off).__name__)

    # (5) мало места под кэш -> явное сообщение и старый путь (никаких тихих OOM)
    import shutil as _shutil
    real_usage = _shutil.disk_usage
    _shutil.disk_usage = lambda p: real_usage(p)._replace(free=1)
    tr.BC_NPZ_RAM_LIMIT_BYTES = 1_000_000
    try:
        if os.path.exists(cache):
            os.remove(cache)
            os.remove(cache + ".meta.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            obs_small, _mm = tr.bc_obs_array(ds, obs_cache="on", verbose=True)
    finally:
        _shutil.disk_usage = real_usage
        tr.BC_NPZ_RAM_LIMIT_BYTES = 1_500_000_000
    msg = buf.getvalue()
    check("J: нет места под кэш -> сказано, сколько нужно и сколько есть",
          "memmap-кэша obs нужно" in msg and "свободно" in msg, msg.strip().splitlines()[-1][:90])
    check("J: нет места под кэш -> работаем как раньше (obs в RAM), а не падаем",
          isinstance(obs_small, np.ndarray) and not isinstance(obs_small, np.memmap),
          type(obs_small).__name__)


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    part_a_warmup_chunked()
    print("-" * 78)
    td = tempfile.mkdtemp(prefix="bc_big_")
    try:
        part_b_on_the_fly(td)
        print("-" * 78)
        part_g_interrupts(td)
        print("-" * 78)
        part_h_cli_wiring()
        print("-" * 78)
        part_i_npz_without_ram(td)
        print("-" * 78)
        part_j_cli_path_load_dataset(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
