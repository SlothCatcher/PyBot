"""Нормализация obs на инференсе и защита от чужой статистики VecNormalize.

Контекст (найденные ошибки):
  1. Обучение идёт с `VecNormalize(norm_obs=True)`, то есть политика видит НОРМАЛИЗОВАННЫЕ
     признаки. Живые боты (`index.py`) и self-play оппоненты в обучении (`agents/env.py`)
     получали СЫРЫЕ признаки — obs сдвинут по масштабу, решения хуже, чем в обучении.
  2. `models/vecnormalize.pkl` в репо хранит статистику на 418 признаков, а текущая
     раскладка — 870. Прежняя «миграция» добивала статистику нулями (mean=0, var=1), то есть
     первые 418 колонок нормализовались статистиками ДРУГОЙ раскладки (раскладка менялась,
     в т.ч. вставками в середину), а `count` ~200k не даёт этому вымыться.

Здесь проверяем: вердикт по статистике, сброс вместо добивания, сайдкар с отпечатком
признаков и применение нормализации в PolicyPlayer.

Запуск: PYTHONPATH=/home/user/PyBot /tmp/venv_pe/bin/python test_obs_norm_inference.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from agents import config as _cfg  # noqa: E402,F401  (монки-патчи poke-env)
from agents.config import N_FEATURES, VECNORM_PATH  # noqa: E402
from agents.vecnorm_utils import (  # noqa: E402
    LiveVecNormalizeAdapter,
    VecNormStats,
    features_fingerprint,
    load_vecnorm_state,
    load_vecnorm_stats,
    load_vecnormalize_for_dim,
    padded_column_mask,
    read_stats_meta,
    reset_obs_rms,
    save_vecnormalize_with_meta,
    stats_meta_path,
    stats_verdict,
    vecnormalize_of,
)

OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


def check_eq(name, got, want):
    check(name, got == want, f"got={got!r} want={want!r}")


class _DictObsEnv(gym.Env):
    """Мини-env с тем же Dict-obs, что у боевого: [observation(N_FEATURES), action_mask(26)]."""

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


def make_vecnormalize(dim: int = N_FEATURES, *, seed_stats: bool = True) -> VecNormalize:
    """Настоящий SB3 VecNormalize на Dict-obs (как в обучении)."""
    venv = DummyVecEnv([lambda: _DictObsEnv(dim)])
    vec = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=0.99, norm_obs_keys=["observation"])
    if seed_stats:
        rng = np.random.default_rng(0)
        for _ in range(3):
            vec.obs_rms["observation"].update(rng.normal(0.5, 1.5, size=(64, dim)).astype(np.float32))
    return vec


def write_old_stats(path: str, dim: int = 418) -> str:
    """Пишет pkl со статистикой «старой» размерности (как models/vecnormalize.pkl на 418)."""
    vec = make_vecnormalize(dim)
    vec.save(path)
    vec.close()
    return path


def main() -> int:
    # models/ в git не хранится (untrack), поэтому в свежем клоне репо-артефакта может не быть.
    # Для проверки вердикта подставляем такой же legacy-файл на 418, НЕ трогая VECNORM_PATH:
    # отсутствие файла — отдельный сценарий, который проверяется ниже (eval -> None + предупреждение).
    repo_path = VECNORM_PATH
    if load_vecnorm_state(repo_path) is None:
        tmp_ref = tempfile.mkdtemp(prefix="vecnorm_ref_")
        repo_path = write_old_stats(os.path.join(tmp_ref, "vecnormalize.pkl"), 418)
        print(f"models/vecnormalize.pkl недоступен — для вердикта беру синтетическую статистику 418: "
              f"{repo_path}")
    print(f"N_FEATURES={N_FEATURES}, VECNORM_PATH={VECNORM_PATH}")
    print(f"repo-статистика читается: {load_vecnorm_state(VECNORM_PATH) is not None}")
    print("-" * 74)

    # ------------------------------------------------------------------ вердикт ---
    src_dim = int(np.asarray(load_vecnorm_state(repo_path)["mean"]).size)
    st = load_vecnorm_state(repo_path)
    keep, reason = stats_verdict(repo_path, N_FEATURES, st["mean"], st["var"])
    check(f"repo-статистика ({src_dim}) не считается статистикой текущей раскладки ({N_FEATURES})",
          (src_dim == N_FEATURES) or (keep is False), f"keep={keep}, reason={reason}")

    # ----------------------------- нормализация в боях за винрейт (eval) --------------
    # Регрессия: в evaluate_win_rates стоял `target_dim=N_FEATURES` без импорта -> NameError
    # глотался except'ом, и модель в боях оценки играла на СЫРЫХ признаках, хотя обучение
    # шло с нормализацией (винрейт и свитчи/тера в боях не сопоставимы с [mix]).
    from agents import training as _training
    check_eq("training.N_FEATURES импортирован на уровне модуля (иначе нормализация eval падала молча)",
             getattr(_training, "N_FEATURES", None), N_FEATURES)

    class _FakePPO:
        def __init__(self, vn):
            self._vn = vn
        def get_vec_normalize_env(self):
            return self._vn

    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        vn_live = make_vecnormalize(N_FEATURES)
        norm = _training.eval_normalizer_for(_FakePPO(vn_live))
        check("eval: нормализация берётся из живого VecNormalize (как в обучении)",
              norm is not None, type(norm).__name__ if norm is not None else "None")
        if norm is not None:
            raw = np.zeros(N_FEATURES, dtype=np.float32)
            raw[0] = float(np.asarray(vn_live.obs_rms["observation"].mean)[0]) + 3.0
            got = np.asarray(norm.normalize(raw), dtype=np.float32)
            check("eval: нормализатор реально меняет сырые признаки",
                  bool(abs(float(got[0]) - float(raw[0])) > 1e-3), f"{raw[0]:.3f} -> {got[0]:.3f}")

        # PYBOT_SELF_PLAY_NORM относится к self-play ОППОНЕНТАМ, а не к оценке: если живой
        # VecNormalize есть, eval обязан нормализовать так же, как обучение.
        os.environ["PYBOT_SELF_PLAY_NORM"] = "0"
        try:
            check("eval: живой VecNormalize применяется и при PYBOT_SELF_PLAY_NORM=0",
                  _training.eval_normalizer_for(_FakePPO(vn_live)) is not None, True)
            check_eq("eval: без живого VecNormalize и с PYBOT_SELF_PLAY_NORM=0 -> без нормализации",
                     _training.eval_normalizer_for(_FakePPO(None)), None)
        finally:
            os.environ.pop("PYBOT_SELF_PLAY_NORM", None)

        # нет ни живого VecNormalize, ни файла статистики -> None, но с громким предупреждением
        import io as _io
        import contextlib as _cl
        buf = _io.StringIO()
        with _cl.redirect_stdout(buf):
            none_norm = _training.eval_normalizer_for(_FakePPO(None))
        out = buf.getvalue()
        check_eq("eval: без статистики нормализатора нет", none_norm, None)
        check("eval: про отключённую нормализацию сказано явно (не молча)",
              "БЕЗ нормализации" in out, out.strip().splitlines()[-1][:90] if out.strip() else "нет вывода")

    with tempfile.TemporaryDirectory() as td:
        dim = 100
        p = os.path.join(td, "vn.pkl")
        write_old_stats(p, dim)

        # 1) размерность совпадает, сайдкара нет -> доверяем (провенанс неизвестен)
        st = load_vecnorm_state(p)
        keep, reason = stats_verdict(p, dim, st["mean"], st["var"])
        check("совпадение размерности без сайдкара: статистика принимается", keep, reason)

        # 2) сайдкар с текущим отпечатком -> точно доверяем
        from agents.vecnorm_utils import write_stats_meta
        write_stats_meta(p, dim, features_fingerprint())
        keep, reason = stats_verdict(p, dim, st["mean"], st["var"])
        check("сайдкар совпал: статистика принимается", keep, reason)

        # 3) сайдкар от другого кода признаков -> не доверяем (dim тот же, раскладка могла поменяться)
        write_stats_meta(p, dim, "deadbeef" * 4)
        keep, reason = stats_verdict(p, dim, st["mean"], st["var"])
        check("сайдкар от другого кода признаков: статистика отклоняется", not keep, reason)

        # 4) подпись паддинга (>=5% колонок mean=0/var=1) -> отклоняем
        m = np.zeros(dim); v = np.ones(dim)
        m[:60], v[:60] = 0.4, 2.0
        keep, reason = stats_verdict(os.path.join(td, "нет-такого.pkl"), dim, m, v)
        check("паддинг (40 из 100 колонок mean=0/var=1): статистика отклоняется", not keep, reason)
        check("паддинг: маска ловит ровно добитые колонки",
              int(np.count_nonzero(padded_column_mask(m, v))) == 40)

        m2 = np.full(dim, 0.3); v2 = np.full(dim, 1.0)   # реальные признаки: var=1, но mean!=0
        keep2, _ = stats_verdict(os.path.join(td, "нет-такого.pkl"), dim, m2, v2)
        check("настоящие признаки (var=1, mean=0.3) не считаются паддингом", keep2)

    print("-" * 74)

    # ------------------------------------------ загрузка для обучения: сброс ---
    with tempfile.TemporaryDirectory() as td:
        old_dim = 418
        p = os.path.join(td, "old418.pkl")
        write_old_stats(p, old_dim)
        env = DummyVecEnv([lambda: _DictObsEnv(N_FEATURES)])

        vec = load_vecnormalize_for_dim(p, env, N_FEATURES)
        rms = vec.obs_rms["observation"]
        check("418 -> 870: размерность статистики стала 870",
              int(np.asarray(rms.mean).size) == N_FEATURES, str(np.asarray(rms.mean).size))
        check("418 -> 870: статистика сброшена (mean=0, var=1, count=0), а не добита нулями",
              bool(np.all(np.asarray(rms.mean) == 0.0)) and bool(np.all(np.asarray(rms.var) == 1.0))
              and float(rms.count) == 0.0,
              f"count={rms.count}, ненулевых mean={int(np.count_nonzero(rms.mean))}")
        # нормализация при сброшенной статистике не портит obs (mean 0, var 1, clip 10)
        probe = np.full((1, N_FEATURES), 3.0, dtype=np.float32)
        normed = vec.normalize_obs({"observation": probe})["observation"]
        check("после сброса нормализация не искажает obs (3.0 -> 3.0)",
              bool(np.allclose(normed, 3.0)), f"{normed[0, :3]}")

        # прежнее поведение доступно флагом keep_stale: добиваем нулями
        vec2 = load_vecnormalize_for_dim(p, env, N_FEATURES, keep_stale=True, quiet=True)
        rms2 = vec2.obs_rms["observation"]
        check("keep_stale=True: старая статистика сохранена в первых 418 колонках (прежнее поведение)",
              float(np.asarray(rms2.mean)[0]) != 0.0 and int(np.asarray(rms2.mean).size) == N_FEATURES)
        check("keep_stale=True: хвост добит нейтрально (mean=0, var=1)",
              bool(np.all(np.asarray(rms2.mean)[old_dim:] == 0.0))
              and bool(np.all(np.asarray(rms2.var[old_dim:]) == 1.0)))

        # force_reset сбрасывает даже совпадающую статистику
        p2 = os.path.join(td, "cur.pkl")
        vec_cur = make_vecnormalize(N_FEATURES)
        vec_cur.save(p2)
        vec_cur.close()
        before = np.asarray(load_vecnorm_state(p2)["mean"])
        check("подготовка: у свежей статистики mean не нулевой", float(np.max(np.abs(before))) > 0)
        vec3 = load_vecnormalize_for_dim(p2, env, N_FEATURES, force_reset=True)
        rms3 = vec3.obs_rms["observation"]
        check("--reset-obs-stats: сбрасывает даже совпадающую статистику",
              float(np.max(np.abs(np.asarray(rms3.mean)))) == 0.0 and float(rms3.count) == 0.0)
        vec.close(); env.close()
        for v in (vec2,):
            try:
                v.venv.close()
            except Exception:
                pass

    print("-" * 74)

    # -------------------------------------------------- сайдкар и сохранение ---
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "saved.pkl")
        vec = make_vecnormalize(N_FEATURES)
        check("save_vecnormalize_with_meta: сохранение успешно", save_vecnormalize_with_meta(vec, p))
        meta = read_stats_meta(p)
        check("сайдкар создан", isinstance(meta, dict) and os.path.isfile(stats_meta_path(p)))
        check_eq("сайдкар: obs_dim = N_FEATURES", int(meta.get("obs_dim", 0)), N_FEATURES)
        check_eq("сайдкар: features_hash = текущий отпечаток",
                 meta.get("features_hash"), features_fingerprint())
        check("meta — обычный json", json.load(open(stats_meta_path(p), encoding="utf-8"))["obs_dim"] == N_FEATURES)

        # сайдкар позволяет принять статистику при resume (без него — «провенанс неизвестен»)
        keep, reason = stats_verdict(p, N_FEATURES, vec.obs_rms["observation"].mean,
                                     vec.obs_rms["observation"].var)
        check("после сохранения с сайдкаром статистика принимается", keep, reason)

        # vecnormalize_of видит вложенный VecNormalize (как в CuriosityVecWrapper)
        class _Wrapper:
            def __init__(self, venv):
                self.venv = venv
            def save(self, path):
                self.venv.save(path)

        check("vecnormalize_of находит VecNormalize внутри обёртки",
              vecnormalize_of(_Wrapper(vec)) is vec)

        # Обёртка БЕЗ своего save (как CuriosityVecWrapper при --icm): раньше сохранение
        # статистики молча падало, а следующая фаза перезагружала устаревший файл
        class _WrapperNoSave:
            def __init__(self, venv):
                self.venv = venv

        p_icm = os.path.join(td, "saved_icm.pkl")
        saved_ok = save_vecnormalize_with_meta(_WrapperNoSave(vec), p_icm)
        check("обёртка без save (--icm): статистика всё равно сохранена", saved_ok and os.path.isfile(p_icm))
        check("обёртка без save: сайдкар записан", read_stats_meta(p_icm) is not None)
        q = vecnormalize_of(_WrapperNoSave(vec))
        check("обёртка без save: сохранён именно VecNormalize",
              q is vec and int(np.asarray(load_vecnorm_state(p_icm)["mean"]).size) == N_FEATURES)
        check("vecnormalize_of(None) -> None", vecnormalize_of(None) is None)
        vec.venv.close()

    print("-" * 74)

    # -------------------------------------------- применение на инференсе ---
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "vn_inf.pkl")
        vec = make_vecnormalize(N_FEATURES)
        vec.save(p)
        rng = np.random.default_rng(1)
        x = rng.normal(1.0, 3.0, size=N_FEATURES).astype(np.float32)

        stats = load_vecnorm_stats(p, N_FEATURES)
        check("load_vecnorm_stats: статистика прочитана", stats is not None)
        check_eq("статистика помечена как актуальная (stale=False)", stats.stale, False)
        check("VecNormStats.normalize совпадает с SB3 normalize_obs бит-в-бит",
              bool(np.allclose(stats.normalize(x),
                               vec.normalize_obs({"observation": x[None, :]})["observation"][0])),
              f"max|Δ|={float(np.max(np.abs(stats.normalize(x) - vec.normalize_obs({'observation': x[None, :]})['observation'][0]))):.2e}")

        adapter = LiveVecNormalizeAdapter(vec, target_dim=N_FEATURES)
        check("LiveVecNormalizeAdapter.normalize совпадает с normalize_obs",
              bool(np.allclose(adapter.normalize(x),
                               vec.normalize_obs({"observation": x[None, :]})["observation"][0])))
        try:
            adapter.normalize(np.zeros(7, dtype=np.float32))
            check("адаптер падает на чужой размерности", False, "исключения не было")
        except ValueError:
            check("адаптер падает на чужой размерности", True)
        vec.venv.close()

        # чужая статистика: применяется, но помечена stale + есть предупреждение
        p_old = os.path.join(td, "old.pkl")
        v_old = make_vecnormalize(418)
        v_old.save(p_old)
        v_old.venv.close()
        stats_old = load_vecnorm_stats(p_old, N_FEATURES)
        check("старая статистика: добита до N_FEATURES", stats_old.mean.size == N_FEATURES)
        check("старая статистика помечена stale", stats_old.stale is True, stats_old.provenance)
        check("для старой статистики есть предупреждение про --reset-obs-stats",
              "--reset-obs-stats" in stats_old.stale_warning())

    print("-" * 74)

    # ------------------------------------------------ PolicyPlayer нормализует ---
    from agents.players import PolicyPlayer

    class _StubNorm:
        def __init__(self):
            self.seen = None
        def normalize(self, obs):
            self.seen = np.asarray(obs, dtype=np.float32)
            return self.seen + 1.0

    player = PolicyPlayer(policy=None, battle_format="gen9fusionmonsrandombattle", start_listening=False)
    check("без нормализатора obs не меняется",
          bool(np.allclose(player._apply_obs_norm(np.zeros(4, dtype=np.float32)), 0.0)))
    norm = _StubNorm()
    player.obs_normalizer = norm
    out = player._apply_obs_norm(np.zeros(4, dtype=np.float32))
    check("с нормализатором obs проходит через normalize()", bool(np.allclose(out, 1.0)))

    class _BadNorm:
        def normalize(self, obs):
            raise ValueError("чужая размерность")

    player.obs_normalizer = _BadNorm()
    check("падение нормализатора не ломает ход (возвращаются сырые признаки)",
          bool(np.allclose(player._apply_obs_norm(np.zeros(4, dtype=np.float32)), 0.0)))

    # нормализация реально применяется внутри embed_battle
    import logging

    from agents.env import _attach_fusion_parser
    from poke_env.battle import Battle

    def make_battle():
        b = Battle(battle_tag="battle-gen9fusionmonsrandombattle-1", username="Me",
                   logger=logging.getLogger("quiet"), gen=9)
        for msg in (["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""], ["", "start"],
                    ["", "switch", "p1a: Aqua", "swampert, L50, M", "362/362"],
                    ["", "switch", "p2a: Skarm", "skarmory, L50, F", "300/300"]):
            b.parse_message(msg)
        return b

    battle = make_battle()
    player2 = PolicyPlayer(policy=None, battle_format="gen9fusionmonsrandombattle", start_listening=False)
    _attach_fusion_parser(player2)
    raw = np.asarray(player2.embed_battle(battle), dtype=np.float32)
    check_eq("embed_battle: форма obs", raw.shape, (N_FEATURES,))
    norm2 = _StubNorm()
    player2.obs_normalizer = norm2
    buffed = np.asarray(player2.embed_battle(battle), dtype=np.float32)
    check("embed_battle применяет нормализатор", norm2.seen is not None and bool(np.allclose(buffed, raw + 1.0)))

    print("-" * 74)

    # ------------------------------------------- self-play оппоненты и index.py ---
    import agents.config as cfg
    import agents.env as env_mod

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "vecnormalize.pkl")
        v = make_vecnormalize(N_FEATURES)
        save_vecnormalize_with_meta(v, p)
        v.venv.close()

        old_path = cfg.VECNORM_PATH
        cfg.VECNORM_PATH = p
        env_mod.VECNORM_PATH = p
        env_mod._SELF_PLAY_NORM_CACHE.clear()
        stats = env_mod._self_play_obs_normalizer()
        check("self-play: нормализатор подхватывается из VECNORM_PATH", stats is not None)
        check("self-play: нормализатор на N_FEATURES", getattr(stats, "mean", None) is not None
              and stats.mean.size == N_FEATURES)
        check("self-play: результат кэшируется",
              env_mod._self_play_obs_normalizer() is stats)
        os.environ["PYBOT_SELF_PLAY_NORM"] = "0"
        check("self-play: PYBOT_SELF_PLAY_NORM=0 выключает нормализацию",
              env_mod._self_play_obs_normalizer() is None)
        os.environ.pop("PYBOT_SELF_PLAY_NORM", None)

        env_mod._SELF_PLAY_NORM_CACHE.clear()
        cfg.VECNORM_PATH = os.path.join(td, "нет-файла.pkl")
        env_mod.VECNORM_PATH = cfg.VECNORM_PATH
        check("self-play: нет файла -> None (играем на сырых признаках)",
              env_mod._self_play_obs_normalizer() is None)
        cfg.VECNORM_PATH = old_path
        env_mod.VECNORM_PATH = old_path
        env_mod._SELF_PLAY_NORM_CACHE.clear()

        # index.py: тот же нормализатор для живых ботов
        import index as index_mod
        old_path2 = cfg.VECNORM_PATH
        cfg.VECNORM_PATH = p
        logs = []
        stats_live = index_mod.load_obs_normalizer(log=logs.append)
        check("index.py: нормализатор для живых ботов получен", stats_live is not None)
        check("index.py: в лог ушла строка про нормализацию",
              any("нормализация obs включена" in m for m in logs), str(logs[:1]))
        cfg.VECNORM_PATH = os.path.join(td, "нет-файла.pkl")
        logs2 = []
        check("index.py: нет файла -> None и понятный лог",
              index_mod.load_obs_normalizer(log=logs2.append) is None
              and any("нет — играю на сырых" in m for m in logs2), str(logs2))
        cfg.VECNORM_PATH = old_path2
        # настоящая проверка проброса: фабрика -> make_player -> PolicyPlayer.obs_normalizer
        from agents.players import PolicyPlayer as _PP
        import agents.config as _c
        cfg.VECNORM_PATH = p
        spec = index_mod.BotSpec(name="t", account="Tester", avatar="1", battle_format="gen9fusionmonsrandombattle")
        player = index_mod.make_factory(None, spec, ("l1", "l2", "pw", "srv"),
                                        battles=2, obs_normalizer=stats_live,
                                        start_listening=False)()
        check("index.py: фабрика передаёт obs_normalizer в PolicyPlayer",
              isinstance(player, _PP) and player.obs_normalizer is stats_live)
        cfg.VECNORM_PATH = old_path2

    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
