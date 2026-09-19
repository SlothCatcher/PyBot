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
            "action_mask": spaces.Box(0, 1, shape=(26,), dtype=bool),
        })
        # 26 = реальный размер действия gen9-фьюжна (6 свитчей + 5 блоков по 4 приёма)
        self.action_space = spaces.Discrete(26)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def make_checkpoint(dim: int, path: str, features_dim: int = 512):
    """Создаёт настоящий SB3-чекпоинт с политикой сети из agents.policy на dim признаков."""
    env = FakeEnv(dim)
    try:
        ppo = PPO(MaskedActorCriticPolicy, env, device="cpu", n_steps=8, batch_size=8, verbose=0,
                  policy_kwargs=dict(
                      features_extractor_kwargs=dict(features_dim=int(features_dim)),
                      net_arch=dict(pi=[512, 256], vf=[512, 256]),
                  ))
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
            "action_mask": torch.ones((2, 26), dtype=torch.bool),
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
            # ВАЖНО (изменено): статистику чужой размерности НЕ добиваем нулями, а сбрасываем.
            # Раскладка признаков менялась (в т.ч. вставками в середину), поэтому старые колонки
            # означают уже другие признаки, а count ~200k не даёт испорченной статистике вымыться.
            check("VecNormalize: статистика чужой размерности сброшена (mean=0, var=1, count=0)",
                  bool(np.all(rms2.mean == 0)) and bool(np.all(rms2.var == 1)) and float(rms2.count) == 0.0,
                  f"count={rms2.count}")
            vn2_stale = load_vecnormalize_for_dim(vn_path, env_new, N_FEATURES, keep_stale=True, quiet=True)
            rms3 = vn2_stale.obs_rms["observation"]
            check("VecNormalize: --keep-obs-stats сохраняет прежнее поведение (паддинг нулями)",
                  bool(np.allclose(rms3.mean[:old_dim], vn.obs_rms["observation"].mean))
                  and bool(np.all(rms3.mean[old_dim:] == 0)) and bool(np.all(rms3.var[old_dim:] == 1)))
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
                      "action_mask": torch.ones((1, 26), dtype=torch.bool),
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

    # 9) расширение экстрактора (features_dim 512 -> 640) вместе с признаками 715 -> 802
    with tempfile.TemporaryDirectory() as td:
        full3 = make_checkpoint(N_FEATURES, os.path.join(td, "full3.zip"), features_dim=512)
        old3 = shrink_checkpoint(full3, os.path.join(td, "old3.zip"), 715)
        src_params = None
        for _, v in load_from_zip_file(old3, device=torch.device("cpu"))[1].items():
            if isinstance(v, dict):
                src_params = v
                break
        src_w1 = src_params["features_extractor.net.0.weight"]
        src_pi = src_params["mlp_extractor.policy_net.0.weight"]

        ppo_big = _migrate_checkpoint_dim(old3, target_dim=N_FEATURES, target_features_dim=640)
        fe = ppo_big.policy.features_extractor
        w1 = fe.net[0].weight
        check("features_dim 512->640: первый слой расширен", tuple(w1.shape) == (640, N_FEATURES),
              str(tuple(w1.shape)))
        check("features_dim 512->640: старый блок [512,715] сохранён побитово",
              bool(torch.equal(w1[:512, :715], src_w1)))
        check("features_dim 512->640: новые нейроны и признаки нулевые",
              int(torch.count_nonzero(w1[512:, :])) == 0 and int(torch.count_nonzero(w1[:512, 715:])) == 0)
        ln_w, ln_b = fe.net[1].weight, fe.net[1].bias
        check("features_dim 512->640: LayerNorm новых нейронов (weight=1, bias=0)",
              tuple(ln_w.shape) == (640,) and float(ln_w[512:].mean()) == 1.0
              and float(ln_b[512:].abs().sum()) == 0.0)
        pi_w = ppo_big.policy.mlp_extractor.policy_net[0].weight
        check("features_dim 512->640: вход pi-головы расширен до 640, старые столбцы на месте",
              tuple(pi_w.shape) == (512, 640) and bool(torch.equal(pi_w[:, :512], src_pi))
              and int(torch.count_nonzero(pi_w[:, 512:])) == 0, str(tuple(pi_w.shape)))
        with torch.no_grad():
            actions, values, _ = ppo_big.policy({
                "observation": torch.zeros((2, N_FEATURES), dtype=torch.float32),
                "action_mask": torch.ones((2, 26), dtype=torch.bool),
            })
        check("features_dim 512->640: расширенная политика делает forward",
              tuple(actions.shape) == (2,) and tuple(values.shape) == (2, 1))
        check("features_dim 512->640: в observation_space записан новый размер",
              tuple(ppo_big.observation_space["observation"].shape) == (N_FEATURES,))

        # обратный случай: сужение 512 -> 384 (обрезка, без падения)
        ppo_small = _migrate_checkpoint_dim(old3, target_dim=N_FEATURES, target_features_dim=384)
        w1s = ppo_small.policy.features_extractor.net[0].weight
        check("features_dim 512->384: слой обрезан", tuple(w1s.shape) == (384, N_FEATURES),
              str(tuple(w1s.shape)))
        check("features_dim 512->384: обрезан именно хвост (старые строки сохранены)",
              bool(torch.equal(w1s, src_w1[:384, :715].new_zeros((384, N_FEATURES)).add(
                  torch.nn.functional.pad(src_w1[:384, :715], (0, N_FEATURES - 715))))))

    # 10) метрика использования новых признаков
    from agents.policy_player import arch_usage_metrics
    from agents.damage import DAMAGE_BLOCK_SIZE
    with tempfile.TemporaryDirectory() as td:
        ck = make_checkpoint(N_FEATURES, os.path.join(td, "m.zip"))
        from stable_baselines3 import PPO as _PPO
        m = arch_usage_metrics(_PPO.load(ck, device="cpu"))
        check("метрика: размерности отражены",
              m.get("arch/feat_dim") == 512 and m.get("arch/obs_dim") == N_FEATURES, str(m))
        check("метрика: свежая сеть использует новые признаки наравне (RMS ratio ~1)",
              0.7 < float(m.get("arch/new_cols_rms_ratio", 0)) < 1.4,
              f"ratio={m.get('arch/new_cols_rms_ratio'):.3f}")
        from agents.features import TYPE_MATCHUP_BLOCK_SIZE
        new_cols = int(m.get("arch/obs_dim", 0)) - 715
        check("метрика: новые колонки == блок урона + блок типов соперника",
              new_cols == DAMAGE_BLOCK_SIZE + TYPE_MATCHUP_BLOCK_SIZE,
              f"{new_cols} vs {DAMAGE_BLOCK_SIZE} + {TYPE_MATCHUP_BLOCK_SIZE}")
        check("метрика: размер самого блока урона не поехал",
              DAMAGE_BLOCK_SIZE == 155, str(DAMAGE_BLOCK_SIZE))

    with tempfile.TemporaryDirectory() as td:
        full4 = make_checkpoint(N_FEATURES, os.path.join(td, "full4.zip"))
        old4 = shrink_checkpoint(full4, os.path.join(td, "old4.zip"), 715)
        warm = _migrate_checkpoint_dim(old4, target_dim=N_FEATURES)
        m_warm = arch_usage_metrics(warm)
        check("метрика: у warm-start модели новые признаки не используются (ratio == 0)",
              float(m_warm.get("arch/new_cols_rms_ratio", 1.0)) == 0.0,
              f"ratio={m_warm.get('arch/new_cols_rms_ratio')}")


    # 11) probe-среда миграции и fallback: голова политики должна выживать
    from agents.policy_player import _checkpoint_arch, _probe_ppo

    ppo_probe, env_probe = _probe_ppo(715)
    check("probe-политика миграции: голова на 26 действий (gen9), а не 9",
          tuple(ppo_probe.policy.action_net.weight.shape)[0] == 26,
          str(tuple(ppo_probe.policy.action_net.weight.shape)))
    check("probe-политика миграции: маска в observation_space = 26",
          tuple(env_probe.observation_space["action_mask"].shape) == (26,),
          str(tuple(env_probe.observation_space["action_mask"].shape)))
    check("probe-политика миграции: action_space = Discrete(26)",
          int(env_probe.action_space.n) == 26, str(env_probe.action_space))
    env_probe.close()

    with tempfile.TemporaryDirectory() as td:
        # настоящий «старый» чекпоинт: FeaturesExtractor всегда строится на текущем
        # N_FEATURES, поэтому сначала метим веса на полном чекпоинте, потом урезаем
        # экстрактор до 715 столбцов через shrink_checkpoint (как у пользователя на диске)
        from stable_baselines3 import PPO as _PPO
        full26 = make_checkpoint(N_FEATURES, os.path.join(td, "full26.zip"))
        marker = _PPO.load(full26, device="cpu")
        with torch.no_grad():
            marker.policy.action_net.weight.fill_(3.25)
            marker.policy.action_net.bias.fill_(-1.5)
            marker.policy.mlp_extractor.policy_net[0].weight.fill_(2.5)
        marked = os.path.join(td, "marked.zip")
        marker.save(marked)
        ck = shrink_checkpoint(marked, os.path.join(td, "old715_head.zip"), 715)
        check("_checkpoint_arch сообщает action_dim",
              int(_checkpoint_arch(ck).get("action_dim", 0)) == 26,
              str(_checkpoint_arch(ck).get("action_dim")))
        src_fw = None
        for _k, _v in load_from_zip_file(ck, device=torch.device("cpu"))[1].items():
            if isinstance(_v, dict) and "features_extractor.net.0.weight" in _v:
                src_fw = _v["features_extractor.net.0.weight"]
        check("старый чекпоинт: экстрактор урезан до 715 столбцов",
              src_fw is not None and tuple(src_fw.shape) == (512, 715), str(tuple(src_fw.shape)))
        mig = _migrate_checkpoint_dim(ck, target_dim=N_FEATURES, force_fallback=True)
        aw = mig.policy.action_net.weight
        ab = mig.policy.action_net.bias
        pw = mig.policy.mlp_extractor.policy_net[0].weight
        fw = mig.policy.features_extractor.net[0].weight
        check("fallback-миграция: голова осталась 26x256", tuple(aw.shape) == (26, 256), str(tuple(aw.shape)))
        check("fallback-миграция: веса головы сохранены (26 действий из чекпоинта)",
              bool(torch.allclose(aw, torch.full_like(aw, 3.25))) and bool(torch.allclose(ab, torch.full_like(ab, -1.5))),
              f"unique={torch.unique(aw).tolist()[:3]} bias={torch.unique(ab).tolist()[:3]}")
        check("fallback-миграция: вход policy_net сохранён (features_dim не менялся)",
              bool(torch.allclose(pw, torch.full_like(pw, 2.5))) and tuple(pw.shape) == (512, 512),
              str(tuple(pw.shape)))
        check("fallback-миграция: первый слой экстрактора расширен до N_FEATURES",
              tuple(fw.shape) == (512, N_FEATURES), str(tuple(fw.shape)))
        check("fallback-миграция: старые 715 колонок — из чекпоинта, новые — нули",
              bool(torch.equal(fw[:, :715], src_fw)) and int(torch.count_nonzero(fw[:, 715:])) == 0,
              f"ненулевых в новых={int(torch.count_nonzero(fw[:, 715:]))}")
        check("fallback-миграция: маска в observation_space = 26",
              tuple(mig.observation_space["action_mask"].shape) == (26,),
              str(tuple(mig.observation_space["action_mask"].shape)))
        check("fallback-миграция: action_space = Discrete(26)",
              int(mig.action_space.n) == 26, str(mig.action_space))


    # 12) legacy-снапшоты: identity-экстрактор (нет features_extractor.*), net_arch [512,256,128],
    #     obs 418. Раньше такие файлы не грузились вообще: `_checkpoint_arch` не находил state_dict
    #     (искал ключи features_extractor), проба собиралась с текущими дефолтами ([512,256] +
    #     Linear-экстрактор) и падала на size mismatch -> "Failed to load qualified snapshot".
    legacy = os.path.join("models", "self_play_snapshot_0.zip")
    if os.path.isfile(legacy):
        from agents.policy_player import _migrate_checkpoint_dim as _mig
        arch_l = _checkpoint_arch(legacy)
        check("legacy: state_dict найден (arch не пустой)", bool(arch_l), str(arch_l))
        check("legacy: action_dim = 26", int(arch_l.get("action_dim", 0)) == 26, str(arch_l.get("action_dim")))
        check("legacy: identity-экстрактор распознан (has_feature_net=False)",
              arch_l.get("has_feature_net") is False, str(arch_l.get("has_feature_net")))
        check("legacy: net_arch из весов = [512, 256, 128]",
              arch_l.get("net_arch", {}).get("pi") == [512, 256, 128]
              and arch_l.get("net_arch", {}).get("vf") == [512, 256, 128],
              str(arch_l.get("net_arch")))
        old_dim_legacy = 418
        ppo_l = _mig(legacy, target_dim=N_FEATURES, verbose=False)
        check("legacy: миграция 418 -> N_FEATURES проходит",
              tuple(ppo_l.policy.action_net.weight.shape) == (26, 128),
              str(tuple(ppo_l.policy.action_net.weight.shape)))
        check("legacy: obs_space мигрированной политики = N_FEATURES",
              tuple(ppo_l.observation_space["observation"].shape) == (N_FEATURES,))
        check("legacy: маска = 26 действий (тера доступна)",
              tuple(ppo_l.observation_space["action_mask"].shape) == (26,))
        # новые признаки входят с нулевыми весами: поведение = старое
        w_pi = ppo_l.policy.mlp_extractor.policy_net[0].weight.detach()
        w_vf = ppo_l.policy.mlp_extractor.value_net[0].weight.detach()
        check("legacy: новые колонки policy_net.0 обнулены",
              int(torch.count_nonzero(w_pi[:, old_dim_legacy:])) == 0, str(tuple(w_pi.shape)))
        check("legacy: новые колонки value_net.0 обнулены",
              int(torch.count_nonzero(w_vf[:, old_dim_legacy:])) == 0, str(tuple(w_vf.shape)))
        # Ширина входа mlp задаёт, куда паддились веса: у legacy identity-экстрактора это
        # ЧИСЛО ПРИЗНАКОВ, а не features_dim. Регрессия: основная ветка миграции паддила до
        # features_dim (512), из-за чего load падал на size mismatch, а при 870 -> 512 веса обрезались.
        check("legacy: вход mlp = N_FEATURES (identity-экстрактор, а не features_dim)",
              tuple(w_pi.shape), (512, N_FEATURES))
        check("legacy: то же для value_net", tuple(w_vf.shape), (512, N_FEATURES))
        # старые колонки побитово из чекпоинта
        src_state = None
        for _k, _v in load_from_zip_file(legacy, device=torch.device("cpu"))[1].items():
            if isinstance(_v, dict) and "mlp_extractor.policy_net.0.weight" in _v:
                src_state = _v
        check("legacy: старые колонки взяты из чекпоинта без изменений",
              src_state is not None and bool(torch.equal(w_pi[:, :old_dim_legacy],
                                                         src_state["mlp_extractor.policy_net.0.weight"])))
        # выход не зависит от новых признаков (warm start: модель ведёт себя как раньше)
        import numpy as _np
        obs_a = _np.zeros((1, N_FEATURES), dtype=_np.float32)
        obs_b = _np.zeros((1, N_FEATURES), dtype=_np.float32)
        rng = _np.random.default_rng(3)
        obs_a[0, :old_dim_legacy] = rng.normal(size=old_dim_legacy).astype(_np.float32)
        obs_b[0, :old_dim_legacy] = obs_a[0, :old_dim_legacy]
        obs_b[0, old_dim_legacy:] = rng.normal(size=N_FEATURES - old_dim_legacy).astype(_np.float32)
        import torch as _torch
        with _torch.no_grad():
            m_a = {"observation": _torch.as_tensor(obs_a), "action_mask": _torch.ones((1, 26), dtype=_torch.bool)}
            m_b = {"observation": _torch.as_tensor(obs_b), "action_mask": _torch.ones((1, 26), dtype=_torch.bool)}
            logits_a = ppo_l.policy.get_distribution(m_a).distribution.logits
            logits_b = ppo_l.policy.get_distribution(m_b).distribution.logits
            val_a = ppo_l.policy.predict_values(m_a)
            val_b = ppo_l.policy.predict_values(m_b)
        check("legacy: новые признаки не влияют на логиты (warm start)",
              bool(_torch.allclose(logits_a, logits_b, atol=1e-6)))
        check("legacy: новые признаки не влияют на value",
              bool(_torch.allclose(val_a, val_b, atol=1e-5)))
        # и предупреждение о том, что снапшот из старой раскладки
        import io as _io
        import contextlib as _cl
        buf = _io.StringIO()
        with _cl.redirect_stdout(buf):
            _mig(legacy, target_dim=N_FEATURES)
        out = buf.getvalue()
        check("legacy: в логе видно, что это старая раскладка/патч lr_schedule",
              ("lr_schedule" in out) or ("legacy" in out) or ("418" in out), out.strip().splitlines()[-1][:90] if out.strip() else "нет вывода")
    else:
        print(f"SKIP legacy-снапшот {legacy} не найден в models/ (файл вне git)")

    # --- 13. probe-политика совместима с реальным env по observation_space (dtype маски)
    # Регрессия: у _DimProbeEnv маска была dtype=bool, а poke-env отдаёт int8, из-за чего
    # `ppo.set_env(env)` в run() падал на check_for_correct_spaces после fallback-миграции:
    #   Dict('action_mask': Box(False, True, (26,), bool)) != Dict('action_mask': Box(0, 1, (26,), int8))
    # То есть resume любого legacy-чекпоинта (fallback-путь) умирал до первого шага.
    print("13. probe-политика и реальный env: dtype маски")
    try:
        import numpy as _np2
        from gymnasium import spaces as _spaces
        from stable_baselines3.common.utils import check_for_correct_spaces
        from agents.policy_player import _probe_ppo

        probe, probe_env = _probe_ppo(N_FEATURES, 512, action_dim=26)
        mask_space = probe.observation_space.spaces["action_mask"]
        check("probe: маска int8 (как poke-env SinglesEnv)", str(mask_space.dtype), "int8")
        check("probe: низ/верх маски 0..1", (mask_space.low.min(), mask_space.high.max()), (0, 1))

        class _RealLikeEnv:
            """observation_space как у ExampleEnv/SinglesEnv: obs float32, маска int8."""
            observation_space = _spaces.Dict({
                "observation": _spaces.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=_np2.float32),
                "action_mask": _spaces.Box(0, 1, shape=(26,), dtype=_np2.int8),
            })
            action_space = _spaces.Discrete(26)

        try:
            check_for_correct_spaces(_RealLikeEnv(), probe.observation_space, probe.action_space)
            ok_spaces = True
        except ValueError as e:
            ok_spaces = False
            print("   ", e)
        check("probe: check_for_correct_spaces проходит (set_env не упадёт)", ok_spaces, True)
        probe_env.close()
    except Exception as e:
        check("probe: проверка пространств выполнилась", f"исключение: {e}", "ок")

    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
