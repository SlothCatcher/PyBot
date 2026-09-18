"""Регресс: self-play оппоненты со старой размерностью obs.

Из лога пользователя: после N_FEATURES=715 -> 802 все qualified-снапшоты падали
    Failed to load qualified snapshot self_play_qualified_12.zip:
    size mismatch ... [512, 715] vs [512, 802]
на каждой фазе, и обучение молча шло без self-play.

Запуск:
    PYTHONPATH=/home/user/PyBot /tmp/venv_pe/bin/python test_self_play_migration.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.config import N_FEATURES

OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


def main():
    from test_dim_migration import make_checkpoint, shrink_checkpoint
    from agents.checkpoint_utils import checkpoint_obs_dim, load_policy_compat, cached_migration_path
    from agents.env import _make_self_play_opponents

    with tempfile.TemporaryDirectory() as td:
        model_dir = os.path.join(td, "models")
        cache_dir = os.path.join(model_dir, "_migrated")
        os.makedirs(model_dir, exist_ok=True)

        # «старые» qualified-снапшоты 715, как у пользователя
        for n in (12, 17, 19):
            full = make_checkpoint(N_FEATURES, os.path.join(td, f"full{n}.zip"))
            shrink_checkpoint(full, os.path.join(model_dir, f"self_play_qualified_{n}.zip"), 715)

        check("checkpoint_obs_dim видит 715",
              checkpoint_obs_dim(os.path.join(model_dir, "self_play_qualified_12.zip")) == 715)

        # 1) загрузка совместимым загрузчиком
        ppo, info = load_policy_compat(os.path.join(model_dir, "self_play_qualified_12.zip"),
                                       N_FEATURES, cache_dir=cache_dir)
        check("load_policy_compat: снапшот 715 загружен (а не упал)", ppo is not None,
              str(info.get("error")))
        check("load_policy_compat: отмечена миграция 715 -> 802", info.get("migrated_from") == 715,
              str(info))
        if ppo is not None:
            w = ppo.policy.features_extractor.net[0].weight
            check("load_policy_compat: веса расширены до 802", tuple(w.shape) == (512, N_FEATURES),
                  str(tuple(w.shape)))
            check("load_policy_compat: старые признаки сохранены, новые нулевые",
                  int(torch.count_nonzero(w[:, 715:])) == 0)

        # 2) кэш появляется и переиспользуется
        cached = cached_migration_path(os.path.join(model_dir, "self_play_qualified_12.zip"),
                                       N_FEATURES, cache_dir)
        check("кэш мигрированного снапшота создан", os.path.isfile(cached), cached)
        ppo2, info2 = load_policy_compat(os.path.join(model_dir, "self_play_qualified_12.zip"),
                                        N_FEATURES, cache_dir=cache_dir)
        check("повторная загрузка идёт из кэша", bool(info2.get("cached")) and ppo2 is not None,
              str(info2))

        # 3) главное: _make_self_play_opponents поднимает оппонентов, а не печатает Failed
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            players = _make_self_play_opponents(model_dir=model_dir + os.sep, cache_dir=cache_dir)
        out = buf.getvalue()
        check("self-play: оппоненты загружены из старых снапшотов", len(players) == 3,
              f"загружено {len(players)}")
        check("self-play: в логе нет 'Failed to load qualified snapshot'",
              "Failed to load qualified snapshot" not in out, out.strip()[:120])
        if players:
            dim = int(players[0].policy.observation_space["observation"].shape[0])
            check("self-play: политика оппонента ждёт 802 признака", dim == N_FEATURES, str(dim))

        # 4) битый файл -> одна строка ошибки, без исключения и без повторов
        bad = os.path.join(model_dir, "self_play_qualified_99.zip")
        with open(bad, "wb") as fh:
            fh.write(b"not a zip")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            players_bad = _make_self_play_opponents(model_dir=model_dir + os.sep, cache_dir=cache_dir)
            players_bad2 = _make_self_play_opponents(model_dir=model_dir + os.sep, cache_dir=cache_dir)
        out_bad = buf.getvalue()
        err_lines = [l for l in out_bad.splitlines()
                     if "Failed to load qualified snapshot self_play_qualified_99" in l]
        bad_reported_for_good = [l for l in out_bad.splitlines() if "Failed" in l
                                 and any(f"_{n}.zip" in l for n in (12, 17, 19))]
        # функция берёт до 3 лучших снапшотов, поэтому битый файл может вытеснить один хороший
        check("битый снапшот: загрузка не падает и остальные оппоненты живы",
              len(players_bad) >= 2, f"загружено {len(players_bad)}")
        check("битый снапшот: ошибка печатается ровно один раз (два вызова подряд)",
              len(err_lines) == 1, f"строк ошибки: {len(err_lines)}")
        check("битый снапшот: рабочие снапшоты 12/17/19 не помечены как сбойные",
              not bad_reported_for_good, str(bad_reported_for_good)[:120])
        check("битый снапшот: повторный вызов даёт то же число оппонентов",
              len(players_bad2) == len(players_bad), f"{len(players_bad2)} vs {len(players_bad)}")

        # 5) снапшот нужной размерности грузится без миграции
        fresh = make_checkpoint(N_FEATURES, os.path.join(model_dir, "self_play_qualified_42.zip"))
        ppo3, info3 = load_policy_compat(fresh, N_FEATURES, cache_dir=cache_dir)
        check("совпадающая размерность: миграции нет",
              ppo3 is not None and info3.get("migrated_from") is None and not info3.get("cached"),
              str(info3))

    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
