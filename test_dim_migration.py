"""Регресс-тест авто-миграции чекпоинтов при изменении N_FEATURES.

Проверяем главный практический сценарий: снапшот обучен на старой размерности obs
(например 715), а в config.py уже новые признаки (802). Загрузка должна не падать,
а добить веса нулями (warm start) — так старые модели продолжают играть и доучиваться.

Запуск (нужны настоящие torch + stable_baselines3):
    PYTHONPATH=/home/user/PyBot /tmp/venv_pe/bin/python test_dim_migration.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

from agents.config import N_FEATURES
from agents.policy import MaskedActorCriticPolicy

OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


class FakeEnv(gym.Env):
    """Минимальное окружение с тем же observation_space, что у боевого env."""

    def __init__(self, dim: int):
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 4.0, shape=(dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(9,), dtype=bool),
        })
        self.action_space = spaces.Discrete(9)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def make_checkpoint(dim: int, path: str):
    """Создаёт настоящий SB3-чекпоинт с политикой сети из agents.policy на dim признаков."""
    env = FakeEnv(dim)
    try:
        ppo = PPO(MaskedActorCriticPolicy, env, device="cpu", n_steps=8, batch_size=8, verbose=0)
    finally:
        env.close()
    ppo.save(path)
    return path


def shrink_checkpoint(src: str, dst: str, keep_cols: int):
    """Из чекпоинта на N_FEATURES делает «старый» чекпоинт на keep_cols признаков.

    Это ровно то, что лежит у пользователя на диске: веса первого слоя уже урезаны,
    в метаданных observation_space старый.
    """
    data, params, _ = load_from_zip_file(src, device=torch.device("cpu"))
    new_params = {}
    for k, v in params.items():
        if isinstance(v, dict):
            new_v = {}
            for kk, tt in v.items():
                if isinstance(tt, torch.Tensor) and tt.dim() == 2 and "features_extractor" in kk \
                        and "weight" in kk and tt.shape[1] == N_FEATURES:
                    new_v[kk] = tt[:, :keep_cols].clone()
                else:
                    new_v[kk] = tt
            new_params[k] = new_v
        else:
            new_params[k] = v
    it = data["observation_space"].spaces["observation"]
    data["observation_space"] = spaces.Dict({
        "observation": spaces.Box(it.low[0], it.high[0], shape=(keep_cols,), dtype=np.float32),
        "action_mask": data["observation_space"].spaces["action_mask"],
    })
    # оптимизатор старой размерности не нужен — как при обычной миграции
    save_to_zip_file(dst, data=data, params=new_params, pytorch_variables=None)
    return dst


def main():
    from agents.policy_player import _checkpoint_obs_dim, _migrate_checkpoint_dim

    with tempfile.TemporaryDirectory() as td:
        full = make_checkpoint(N_FEATURES, os.path.join(td, "full.zip"))
        check("_checkpoint_obs_dim читает размерность",
              _checkpoint_obs_dim(full) == N_FEATURES,
              f"{_checkpoint_obs_dim(full)} vs {N_FEATURES}")

        # 1) миграция не нужна: тот же размер -> просто загрузка, без паддинга
        ppo_same = _migrate_checkpoint_dim(full, target_dim=N_FEATURES)
        w_same = ppo_same.policy.features_extractor.net[0].weight
        check("no-op миграция сохраняет размер весов", tuple(w_same.shape) == (512, N_FEATURES),
              str(tuple(w_same.shape)))

        # 2) основной сценарий: 715 -> N_FEATURES
        old_dim = 715
        if old_dim == N_FEATURES:
            old_dim = 700
        old_zip = shrink_checkpoint(full, os.path.join(td, "old715.zip"), old_dim)
        check("старый чекпоинт: размерность определена из метаданных",
              _checkpoint_obs_dim(old_zip) == old_dim, str(_checkpoint_obs_dim(old_zip)))

        ppo = _migrate_checkpoint_dim(old_zip, target_dim=N_FEATURES)
        w = ppo.policy.features_extractor.net[0].weight
        check("после миграции первый слой расширен до N_FEATURES",
              tuple(w.shape) == (512, N_FEATURES), str(tuple(w.shape)))
        check("веса не обнулились целиком (warm start, а не пустышка)",
              int(torch.count_nonzero(w[:, :old_dim])) > 0,
              f"ненулевых в старых колонках={int(torch.count_nonzero(w[:, :old_dim]))}")
        orig = load_from_zip_file(old_zip, device=torch.device("cpu"))[1]
        orig_w = None
        for k, v in orig.items():
            if isinstance(v, dict):
                for kk, tt in v.items():
                    if isinstance(tt, torch.Tensor) and tt.dim() == 2 and "features_extractor" in kk and "weight" in kk:
                        orig_w = tt
        check("первые old-колонок побитово равны исходным",
              orig_w is not None and bool(torch.equal(w[:, :old_dim], orig_w)))
        check("новые колонки заполнены нулями",
              bool(torch.count_nonzero(w[:, old_dim:]) == 0),
              f"ненулевых={int(torch.count_nonzero(w[:, old_dim:]))}")
        check("observation_space в загруженной модели обновлён",
              tuple(ppo.observation_space["observation"].shape) == (N_FEATURES,),
              str(tuple(ppo.observation_space["observation"].shape)))

        # 3) мигрированная модель реально считает forward на новой размерности
        obs = {
            "observation": torch.zeros((2, N_FEATURES), dtype=torch.float32),
            "action_mask": torch.ones((2, 9), dtype=torch.bool),
        }
        with torch.no_grad():
            actions, values, log_prob = ppo.policy(obs)
        check("мигрированная политика делает forward без ошибок",
              tuple(actions.shape) == (2,) and tuple(values.shape) == (2, 1),
              f"actions={tuple(actions.shape)}, values={tuple(values.shape)}")

        # 4) повторная миграция идемпотентна (второй заход уже 802 -> 802)
        ppo2 = _migrate_checkpoint_dim(old_zip, target_dim=N_FEATURES)
        w2 = ppo2.policy.features_extractor.net[0].weight
        check("повторная миграция даёт тот же результат", bool(torch.equal(w, w2)))

        # 4b) тот же сценарий, что у пользователя: путь без .zip на resume
        from agents.policy_player import resolve_checkpoint_path
        noext = os.path.join(td, "self_play_qualified_19")
        zipped = noext + ".zip"
        os.rename(old_zip, zipped)
        check("resume без расширения .zip находит файл", resolve_checkpoint_path(noext) == zipped)
        ppo3 = _migrate_checkpoint_dim(resolve_checkpoint_path(noext), target_dim=N_FEATURES)
        w3 = ppo3.policy.features_extractor.net[0].weight
        check("миграция по пути без .zip даёт 715 -> N_FEATURES",
              tuple(w3.shape) == (512, N_FEATURES), str(tuple(w3.shape)))

    # 5) VecNormalize со старой статистикой должен паддиться, а не падать
    with tempfile.TemporaryDirectory() as td:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from agents.vecnorm_utils import load_vecnormalize_for_dim, load_vecnorm_stats

        old_dim = 418
        env_old = DummyVecEnv([lambda: FakeEnv(old_dim)])
        vn = VecNormalize(env_old, norm_obs=True, norm_reward=False, gamma=0.99,
                          norm_obs_keys=["observation"])
        rng = np.random.RandomState(0)
        for _ in range(20):
            vn.obs_rms["observation"].update(rng.randn(1, old_dim).astype(np.float32))
        vn_path = os.path.join(td, "vecnormalize.pkl")
        vn.save(vn_path)
        env_new = DummyVecEnv([lambda: FakeEnv(N_FEATURES)])
        try:
            vn2 = load_vecnormalize_for_dim(vn_path, env_new, N_FEATURES)
            rms2 = vn2.obs_rms["observation"]
            check("VecNormalize: статистика добита до N_FEATURES без падения",
                  tuple(rms2.mean.shape) == (N_FEATURES,), str(tuple(rms2.mean.shape)))
            check("VecNormalize: старые stats сохранены, новые mean=0/var=1",
                  bool(np.allclose(rms2.mean[:old_dim], vn.obs_rms["observation"].mean))
                  and bool(np.all(rms2.mean[old_dim:] == 0)) and bool(np.all(rms2.var[old_dim:] == 1)))
            obs_vec = vn2.reset()
            if isinstance(obs_vec, tuple):  # старый API gym
                obs_vec = obs_vec[0]
            check("VecNormalize: normalize_obs работает на новых obs",
                  obs_vec["observation"].shape == (1, N_FEATURES),
                  str(obs_vec["observation"].shape))
        except Exception as e:
            check(f"VecNormalize: миграция статистики ({type(e).__name__}: {e})", False)
        st = load_vecnorm_stats(vn_path, N_FEATURES)
        check("load_vecnorm_stats: статистика на N_FEATURES",
              st is not None and st.ready, st.describe() if st else "None")
        env_old.close()
        env_new.close()

    # 6) старый датасет для BC добивается нулями (без пересборки 3.6GB)
    from agents.training import _pad_obs_to_features

    old_dim = 715
    small = np.random.randn(50, old_dim).astype(np.float32)
    padded = _pad_obs_to_features(small, N_FEATURES, label="тест", memmap_threshold=1000)
    check("BC-датасет: форма добита до N_FEATURES",
          padded.shape == (50, N_FEATURES), str(padded.shape))
    check("BC-датасет: старые признаки сохранены побитово",
          bool(np.array_equal(padded[:, :old_dim], small)))
    check("BC-датасет: новые признаки нулевые",
          bool(np.all(padded[:, old_dim:] == 0)))
    same = _pad_obs_to_features(np.zeros((4, N_FEATURES), dtype=np.float32), N_FEATURES)
    check("BC-датасет: совпадающая размерность не копируется",
          same.shape == (4, N_FEATURES))
    cut = _pad_obs_to_features(np.ones((3, N_FEATURES + 5), dtype=np.float32), N_FEATURES)
    check("BC-датасет: лишние признаки обрезаются", cut.shape == (3, N_FEATURES), str(cut.shape))
    big = _pad_obs_to_features(np.ones((40, old_dim), dtype=np.float32), N_FEATURES,
                               label="тест-mmap", memmap_threshold=10)
    check("BC-датасет: большой путь через mmap работает",
          big.shape == (40, N_FEATURES) and float(big[0, -1]) == 0.0 and float(big[0, 0]) == 1.0,
          f"shape={big.shape}, edge={float(big[0, 0])}/{float(big[0, -1])}")

    # 7) fallback-путь миграции (если основная упала) тоже должен работать и не требовать сервера
    with tempfile.TemporaryDirectory() as td:
        full2 = make_checkpoint(N_FEATURES, os.path.join(td, "full2.zip"))
        old2 = shrink_checkpoint(full2, os.path.join(td, "old2.zip"), 715)
        try:
            ppo_fb = _migrate_checkpoint_dim(old2, target_dim=N_FEATURES, force_fallback=True)
            w_fb = ppo_fb.policy.features_extractor.net[0].weight
            check("fallback-миграция: первый слой расширен до N_FEATURES",
                  tuple(w_fb.shape) == (512, N_FEATURES), str(tuple(w_fb.shape)))
            check("fallback-миграция: новые колонки нулевые",
                  int(torch.count_nonzero(w_fb[:, 715:])) == 0)
            check("fallback-миграция: observation_space обновлён",
                  tuple(ppo_fb.observation_space["observation"].shape) == (N_FEATURES,),
                  str(tuple(ppo_fb.observation_space["observation"].shape)))
            try:
                ppo_fb.policy.env = ppo_fb.get_env()
            except Exception:
                pass
            check("fallback-миграция: политика делает forward",
                  tuple(ppo_fb.policy({
                      "observation": torch.zeros((1, N_FEATURES), dtype=torch.float32),
                      "action_mask": torch.ones((1, 9), dtype=torch.bool),
                  })[0].shape) == (1,))
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"fallback-миграция (упала: {type(e).__name__}: {e})", False)

    # 8) детектор несовпадения размерностей по тексту ошибки
    from agents.policy_player import _looks_like_dim_mismatch
    check("ошибка SB3 'size mismatch' распознаётся",
          _looks_like_dim_mismatch(
              "size mismatch for features_extractor.net.0.weight: copying a param with shape "
              "torch.Size([512, 715]) from checkpoint, the shape in current model is torch.Size([512, 802])."))
    check("ошибка 'Error(s) in loading state_dict' распознаётся",
          _looks_like_dim_mismatch("Error(s) in loading state_dict for MaskedActorCriticPolicy: size mismatch"))
    check("ошибка формы в matmul распознаётся",
          _looks_like_dim_mismatch("mat1 and mat2 shapes cannot be multiplied (4x715 and 802x512)"))
    check("посторонние ошибки не считаются несовпадением размеров",
          not _looks_like_dim_mismatch("FileNotFoundError: nope")
          and not _looks_like_dim_mismatch("RuntimeError: CUDA out of memory"))

    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
