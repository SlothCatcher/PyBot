"""Проверка маршрута «BC-претрейн -> RL-дообучение» (то, что пользователь запускает руками).

Эталонный маршрут:
  1) python -m agents.policy_player --dataset-path models/mixed.npz --epochs 15 --contrastive \
       --neg-weight 0.3 --total-timesteps 0 --ent-coef 0.05 --lr 3e-4
  2) python -m agents.policy_player --resume models/pretrained_v2 --total-timesteps 10000000 \
       --lr 3e-5 --clip-range 0.1 --n-epochs 3 --batch-size 256 --vf-coef 0.5 --min-winrate 25 \
       --reset-schedules --icm --icm-anneal --eval-battles 60

Проверяем места, где маршрут ломался:
  * датасет старой раскладки (в репо лежат 418 и 411-мерные) молча добивался нулями в хвост —
    хотя 713 -> 715 вставил два признака В СЕРЕДИНУ; теперь такие датасеты отвергаются;
  * warm-up статистики нормализации падал на broadcast (obs 418 против статистики 870);
  * датасет без `ret` выяснялся только внутри BC (после подъёма env-процессов);
  * `--total-timesteps 0` всё равно поднимал 8 env-процессов и уходил в финальный сырой прогон
    на 300 боёв (висел без Showdown-сервера) — `--skip-eval` его не отключал;
  * имя финальной модели было жёстко `models/ppo_policy_final`, поэтому `--resume
    models/pretrained_v2` из шага 2 падал (теперь есть `--save-as`).

Все проверки идут БЕЗ Showdown-сервера и не трогают рабочую папку models/ (cwd — временный).

Запуск: PYTHONPATH=/home/user/PyBot /tmp/venv_pe/bin/python test_training_route.py
"""
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from agents import policy_player as pp  # noqa: E402
from agents.config import N_FEATURES, VECNORM_PATH  # noqa: E402
from agents.policy import MaskedActorCriticPolicy  # noqa: E402
from agents.training import (  # noqa: E402
    MIN_PREFIX_OBS_DIM,
    _pad_obs_to_features,
    pretrain_policy_bc,
    validate_bc_dataset,
    warm_up_vec_normalize,
)

OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


class Env870(gym.Env):
    """Env с наблюдением как у боевого (Dict: observation + action_mask)."""

    def __init__(self, dim: int = N_FEATURES):
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 4.0, shape=(dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(26,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(26)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def make_npz(path: str, dim: int, n: int = 60, with_ret: bool = True, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    payload = {
        "obs": rng.normal(size=(n, dim)).astype(np.float32),
        "mask": np.ones((n, 26), np.int8),
        "action": rng.integers(0, 26, n).astype(np.int64),
    }
    if with_ret:
        payload["ret"] = rng.normal(size=n).astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def expect_systemexit(fn):
    try:
        fn()
        return None
    except SystemExit as e:
        return str(e)
    except Exception as e:  # не тот тип исключения — тоже провал
        return f"__OTHER__{type(e).__name__}: {e}"


def test_validate_bc_dataset(td):
    ok = make_npz(os.path.join(td, "ok870.npz"), N_FEATURES)
    info = validate_bc_dataset(ok, N_FEATURES)
    check("валидатор: датасет 870 с ret принят",
          info["obs_dim"] == N_FEATURES and info["examples"] == 60, str(info))

    ok715 = make_npz(os.path.join(td, "ok715.npz"), 715)
    info715 = validate_bc_dataset(ok715, N_FEATURES)
    check("валидатор: датасет 715 (префикс раскладки) принят", info715["obs_dim"] == 715)

    for dim in (418, 411, 713):
        p = make_npz(os.path.join(td, f"old{dim}.npz"), dim)
        msg = expect_systemexit(lambda p=p: validate_bc_dataset(p, N_FEATURES))
        check(f"валидатор: датасет {dim} отвергнут с объяснением про середину раскладки",
              msg is not None and "__OTHER__" not in msg and "СЕРЕДИНУ" in msg,
              (msg or "исключения не было").splitlines()[0][:80])

    p_noret = make_npz(os.path.join(td, "noret.npz"), N_FEATURES, with_ret=False)
    msg = expect_systemexit(lambda: validate_bc_dataset(p_noret, N_FEATURES))
    check("валидатор: датасет без ret отвергнут до создания env",
          msg is not None and "__OTHER__" not in msg and "ret" in msg,
          (msg or "исключения не было").splitlines()[0][:80])


def test_padding_is_layout_aware():
    a418 = np.ones((3, 418), dtype=np.float32)
    for dim in (418, 713):
        arr = np.ones((3, dim), dtype=np.float32)
        try:
            _pad_obs_to_features(arr, N_FEATURES, label=f"датасет{dim}")
            check(f"паддинг {dim}: должен был упасть", False, "исключения не было")
        except ValueError as e:
            check(f"паддинг {dim}: ValueError с объяснением", "СЕРЕДИНУ" in str(e))
    out715 = _pad_obs_to_features(np.ones((3, 715), dtype=np.float32), N_FEATURES)
    check("паддинг 715 -> 870: хвост нулевой, префикс сохранён",
          out715.shape == (3, N_FEATURES) and bool(np.all(out715[:, :715] == 1.0))
          and bool(np.all(out715[:, 715:] == 0.0)))
    out802 = _pad_obs_to_features(np.ones((2, 802), dtype=np.float32), N_FEATURES)
    check("паддинг 802 -> 870", out802.shape == (2, N_FEATURES) and bool(np.all(out802[:, 802:] == 0.0)))
    out_wide = _pad_obs_to_features(np.ones((2, N_FEATURES + 30), dtype=np.float32), N_FEATURES)
    check("датасет шире признаков: хвост обрезан до N_FEATURES", out_wide.shape == (2, N_FEATURES))
    check("константа границы раскладки = 715", MIN_PREFIX_OBS_DIM == 715, str(MIN_PREFIX_OBS_DIM))


def test_warm_up_does_not_crash(td):
    venv = VecNormalize(DummyVecEnv([lambda: Env870()]), norm_obs=True, norm_reward=False,
                        gamma=0.99, norm_obs_keys=["observation"])
    rms = venv.obs_rms["observation"]
    check("свежая статистика: размерность 870", int(np.asarray(rms.mean).size) == N_FEATURES)

    # 1) уже выровненный ndarray (так передаёт pretrain_policy_bc) — раньше падало на stack
    arr870 = np.random.default_rng(1).normal(size=(60, N_FEATURES)).astype(np.float32)
    warm_up_vec_normalize(venv, arr870)
    check("warm-up на ndarray 870: размерность не поехала", int(np.asarray(rms.mean).size) == N_FEATURES)
    check("warm-up на ndarray 870: статистика обновилась по данным",
          float(rms.count) > 0 and abs(float(np.asarray(rms.mean)[0])) > 1e-6,
          f"count={rms.count}")

    # 2) датасет 715 приводится к размерности статистики, а не падает на broadcast
    venv2 = VecNormalize(DummyVecEnv([lambda: Env870()]), norm_obs=True, norm_reward=False,
                         gamma=0.99, norm_obs_keys=["observation"])
    arr715 = np.random.default_rng(2).normal(size=(40, 715)).astype(np.float32)
    warm_up_vec_normalize(venv2, arr715)
    check("warm-up на 715: приведён к 870 без падения",
          int(np.asarray(venv2.obs_rms["observation"].mean).size) == N_FEATURES)

    # 3) датасет старой раскладки — честная ошибка, а не порча статистики
    try:
        warm_up_vec_normalize(venv2, np.ones((10, 418), dtype=np.float32))
        check("warm-up на 418: должен был упасть", False, "исключения не было")
    except ValueError as e:
        check("warm-up на 418: ValueError с объяснением", "СЕРЕДИНУ" in str(e))
    venv.venv.close()
    venv2.venv.close()


def test_bc_on_715_dataset(td):
    """BC-нормализация + паддинг вместе: раньше падало на broadcast (418/715 против 870)."""
    path = make_npz(os.path.join(td, "ds715.npz"), 715, n=120)
    venv = VecNormalize(DummyVecEnv([lambda: Env870()]), norm_obs=True, norm_reward=False,
                        gamma=0.99, norm_obs_keys=["observation"])
    ppo = PPO(MaskedActorCriticPolicy, venv, device="cpu", n_steps=8, batch_size=8, verbose=0,
              policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=512)))
    try:
        pretrain_policy_bc(ppo, path, epochs=1, normalize=True, contrastive=True, neg_weight=0.3)
        check("BC на датасете 715 прошёл без ошибок (паддинг + нормализация)", True)
    except Exception as e:
        check("BC на датасете 715 прошёл без ошибок (паддинг + нормализация)", False,
              f"{type(e).__name__}: {e}")
    check("после BC статистика нормализации осталась 870",
          int(np.asarray(venv.obs_rms["observation"].mean).size) == N_FEATURES)
    check("после BC статистика не нулевая (warm-up учёл датасет)",
          float(np.max(np.abs(np.asarray(venv.obs_rms["observation"].mean)))) > 1e-6)
    venv.venv.close()


def test_skip_eval_gates_both_evals():
    """--skip-eval обязан отключать и eval, и сырой прогон на 300 боёв."""
    calls = {"eval": 0, "battles": 0}

    class _PlayerStub:
        def __init__(self, **kwargs):
            pass

        async def battle_against(self, *opps, n_battles=100):
            calls["battles"] += 1

    orig_eval, orig_player = pp.evaluate_win_rates, pp.PolicyPlayer
    pp.evaluate_win_rates = lambda ppo, n_battles=20: (calls.__setitem__("eval", calls["eval"] + 1), {})[1]
    pp.PolicyPlayer = _PlayerStub
    try:
        rates = pp._final_evals(None, 60, skip_eval=True)
        check("--skip-eval: результат помечен как пропущенный", rates == {"skipped": 0}, str(rates))
        check("--skip-eval: eval не запускался", calls["eval"] == 0)
        check("--skip-eval: сырой прогон на 300 боёв НЕ запускался", calls["battles"] == 0)

        pp._final_evals(None, 60, skip_eval=False, skip_raw=True)
        check("--skip-final-raw-eval: eval запускался", calls["eval"] == 1)
        check("--skip-final-raw-eval: сырой прогон пропущен", calls["battles"] == 0)
    finally:
        pp.evaluate_win_rates, pp.PolicyPlayer = orig_eval, orig_player


def test_run_bc_only_offline(td):
    """Полный прогон шага 1 маршрута: BC-only, без сервера, с --save-as."""
    cwd = os.getcwd()
    env_fns_seen = []

    def fake_subproc(fns):
        env_fns_seen.append(len(fns))
        return DummyVecEnv([lambda: Env870()])

    orig_subproc = pp.SubprocVecEnv
    orig_cwd = os.getcwd()
    pp.SubprocVecEnv = fake_subproc
    os.chdir(td)
    os.makedirs("models", exist_ok=True)
    ds = make_npz(os.path.join(td, "mixed_ok.npz"), N_FEATURES, n=80)
    try:
        pp.run(dataset_path=ds, epochs=1, contrastive=True, neg_weight=0.3, total_timesteps=0,
               ent_coef=0.05, learning_rate=3e-4, skip_eval=True, save_as="pretrained_v2",
               num_envs=8)
        check("шаг 1: модель сохранена как models/pretrained_v2.zip",
              os.path.isfile("models/pretrained_v2.zip"))
        check("шаг 1: VecNormalize сохранён с сайдкаром",
              os.path.isfile(VECNORM_PATH) and os.path.isfile(VECNORM_PATH + ".meta.json"))
        check("шаг 1: BC-only поднял 1 env, а не 8 (RL не запускается)",
              env_fns_seen == [1], str(env_fns_seen))
        import json
        meta = json.load(open(VECNORM_PATH + ".meta.json", encoding="utf-8"))
        check("шаг 1: сайдкар описывает текущую раскладку",
              int(meta.get("obs_dim", 0)) == N_FEATURES and bool(meta.get("features_hash")), str(meta))
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("шаг 1: прогон без сервера прошёл", False, f"{type(e).__name__}: {e}")
    finally:
        pp.SubprocVecEnv = orig_subproc
        os.chdir(orig_cwd)

    # шаг 2: resume по этому файлу находится и грузится (с расширением .zip)
    os.chdir(td)
    try:
        resolved = pp.resolve_checkpoint_path("models/pretrained_v2")
        check("шаг 2: --resume models/pretrained_v2.resolve находит .zip",
              resolved.endswith("pretrained_v2.zip"), resolved)
        from agents.checkpoint_utils import load_policy_compat
        ppo, info = load_policy_compat("models/pretrained_v2.zip", N_FEATURES)
        check("шаг 2: чекпоинт из шага 1 грузится без миграции",
              ppo is not None and info.get("migrated_from") is None, str(info))
        check("шаг 2: у чекпоинта 26 действий и obs N_FEATURES",
              tuple(ppo.policy.action_net.weight.shape)[0] == 26
              and tuple(ppo.observation_space["observation"].shape) == (N_FEATURES,))
    except Exception as e:
        check("шаг 2: чекпоинт загружается", False, f"{type(e).__name__}: {e}")
    finally:
        os.chdir(orig_cwd)


def test_cli_wiring():
    args = pp.parse_args(["--dataset-path", "models/mixed.npz", "--epochs", "15", "--contrastive",
                          "--neg-weight", "0.3", "--total-timesteps", "0", "--ent-coef", "0.05",
                          "--lr", "3e-4", "--save-as", "pretrained_v2"])
    check("CLI шага 1: --lr попал в lr_alias", abs(float(args.lr_alias) - 3e-4) < 1e-12, str(args.lr_alias))
    check("CLI шага 1: --save-as распознан", args.save_as == "pretrained_v2", str(args.save_as))
    check("CLI шага 1: --total-timesteps 0", int(args.total_timesteps) == 0)

    args2 = pp.parse_args(["--resume", "models/pretrained_v2", "--total-timesteps", "10000000",
                           "--lr", "3e-5", "--clip-range", "0.1", "--n-epochs", "3",
                           "--batch-size", "256", "--vf-coef", "0.5", "--min-winrate", "25",
                           "--reset-schedules", "--icm", "--icm-anneal", "--eval-battles", "60"])
    check("CLI шага 2: все флаги распознаны",
          args2.resume == "models/pretrained_v2" and int(args2.total_timesteps) == 10_000_000
          and bool(args2.reset_schedules) and bool(args2.icm) and bool(args2.icm_anneal)
          and int(args2.eval_battles) == 60 and int(args2.min_winrate) == 25,
          f"icm={args2.icm}, anneal={args2.icm_anneal}, eval={args2.eval_battles}")
    check("CLI: --skip-final-raw-eval существует",
          bool(pp.parse_args(["--skip-final-raw-eval"]).skip_final_raw_eval))


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")

    with tempfile.TemporaryDirectory() as td:
        test_validate_bc_dataset(td)
        print("-" * 74)
        test_padding_is_layout_aware()
        print("-" * 74)
        test_warm_up_does_not_crash(td)
        print("-" * 74)
        test_bc_on_715_dataset(td)
        print("-" * 74)
        test_skip_eval_gates_both_evals()
        print("-" * 74)
        test_cli_wiring()
        print("-" * 74)
        test_run_bc_only_offline(td)
    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
