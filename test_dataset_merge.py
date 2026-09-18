"""Тесты сборки/мержа датасета: воспроизводим падение

    ValueError: could not broadcast input array from shape (26918,715) into shape (26918,713)

и проверяем, что датасет собирается из СВЕЖЕГО пересчёта, а не из чужих чанков.
Сервер не нужен: сырые чанки делаются из живых `poke_env.battle.Battle` (как их пишет
HeuristicRecorder), дальше всё локально.

Запуск: python test_dataset_merge.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import training as T  # noqa: E402
from agents.config import N_FEATURES  # noqa: E402

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def check_true(label, cond, extra=""):
    check(label + (f": {extra}" if extra else ""), bool(cond), True)


# ------------------------------------------------------------------ утилиты ---
def make_chunk(path, n, obs_dim, mask_w=26, has_ret=True, fill=None):
    obs = np.full((n, obs_dim), fill if fill is not None else float(obs_dim % 7) / 7.0, dtype=np.float32)
    mask = np.ones((n, mask_w), dtype=np.int8)
    action = np.arange(n, dtype=np.int64) % 26
    kwargs = dict(obs=obs, mask=mask, action=action)
    if has_ret:
        kwargs["ret"] = np.linspace(1.0, 2.0, n).astype(np.float32)
    np.savez_compressed(path, **kwargs)
    return path


def live_battle(tag="battle-gen9fusionmonsrandombattle-1", our="swampert", opp="steelix",
                move="surf", hp="362/362", reveal_move=None):
    """Настоящий Battle с одним ходом — ровно такой объект пишется в сырой кэш."""
    import logging

    from poke_env.battle import Battle

    b = Battle(battle_tag=tag, username="Me", logger=logging.getLogger("quiet"), gen=9)
    msgs = [["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""],
            ["", "start"],
            ["", "switch", "p1a: Aqua", f"{our}, L50, M", hp],
            ["", "switch", "p2a: Rock", f"{opp}, L50, F", "300/300"]]
    if reveal_move:
        # без раскрытого приёма противника блок урона нулевой и фьюжн-карты в нём не видны
        msgs += [["", "move", "p2a: Rock", reveal_move, "p1a: Aqua"],
                 ["", "-damage", "p1a: Aqua", "300/362"]]
    for msg in msgs:
        b.parse_message(msg)
    return b


def raw_entry(battle, action=0, tag=None):
    """Запись сырого кэша в формате HeuristicRecorder (8 или 9 элементов)."""
    from agents.features import embed_battle_with_fusion  # noqa: F401  (проверка импорта)
    from poke_env.environment.singles_env import SinglesEnv

    mask = np.array(SinglesEnv.get_action_mask(battle))
    tag = tag or battle.battle_tag
    return (battle, mask, action, tag, None, None, 0, 0)


def write_raw_chunk(path, entries, battle_tags, battles_won=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "raw_dataset": entries,
        "battles": {t: SimpleNamespace(won=battles_won) for t in battle_tags},
        "chunk_idx": int(os.path.basename(path).split("_")[-1].split(".")[0]),
    }
    with open(path, "wb") as f:
        import pickle
        pickle.dump(payload, f)
    return path


# -------------------------------------------------------------------- тесты ---
def test_merge_same_dims(tmpdir):
    """Обычный случай: одинаковые размерности — порядок и данные сохраняются."""
    files = [make_chunk(os.path.join(tmpdir, f"c{i}.npz"), 5, N_FEATURES) for i in range(3)]
    out = os.path.join(tmpdir, "merged.npz")
    T._merge_dataset_chunks(files, out)
    d = np.load(out)
    check("всего примеров", int(d["obs"].shape[0]), 15)
    check("obs_dim", int(d["obs"].shape[1]), N_FEATURES)
    check("mask_dim", int(d["mask"].shape[1]), 26)
    check_true("ret на месте", "ret" in d)
    check("action первого", int(d["action"][0]), 0)
    check("action шестого (начало 2-го чанка)", int(d["action"][5]), 0)


def test_merge_mixed_obs_dim_gives_clear_error(tmpdir):
    """Чанки 713 и 715 (как в логе) — понятная ошибка вместо broadcast-падения середины записи."""
    files = [make_chunk(os.path.join(tmpdir, "old713.npz"), 4, 713),
             make_chunk(os.path.join(tmpdir, "new715.npz"), 4, 715)]
    out = os.path.join(tmpdir, "merged.npz")
    try:
        T._merge_dataset_chunks(files, out)
        err = None
    except ValueError as exc:
        err = str(exc)
    check_true("ValueError вместо broadcast", err is not None, str(err)[:120])
    check_true("в сообщении видны обе размерности", err is not None and "713" in err and "715" in err)
    check_true("в сообщении есть что делать (и не советует стереть сырой кэш)",
               err is not None and "dataset_chunk_" in err and "ТОЛЬКО чанки датасета" in err
               and "сырой кэш и --force-recollect" in err.lower())
    check_true("недописанный файл не остался", not os.path.exists(out))


def test_merge_mixed_mask_dim_gives_clear_error(tmpdir):
    """Разная ширина маски = разный action space (9 vs 26) — тоже понятная ошибка."""
    files = [make_chunk(os.path.join(tmpdir, "m9.npz"), 3, N_FEATURES, mask_w=9),
             make_chunk(os.path.join(tmpdir, "m26.npz"), 3, N_FEATURES, mask_w=26)]
    try:
        T._merge_dataset_chunks(files, os.path.join(tmpdir, "out.npz"))
        err = None
    except ValueError as exc:
        err = str(exc)
    check_true("ValueError про mask_dim", err is not None and "mask_dim" in err, str(err)[:120])


def test_wrong_dim_chunks_detection(tmpdir):
    """Утилита, по которой ветка «кэш чанков» решает пересчитать obs."""
    files = [make_chunk(os.path.join(tmpdir, "a.npz"), 2, N_FEATURES),
             make_chunk(os.path.join(tmpdir, "b.npz"), 2, 715),
             make_chunk(os.path.join(tmpdir, "c.npz"), 2, 715)]
    check("найдены чанки старой размерности", T._wrong_dim_dataset_chunks(files, N_FEATURES), {715: 2})
    check("совпадающие чанки не считаются плохими",
          T._wrong_dim_dataset_chunks([files[0]], N_FEATURES), {})
    check("сводка размерностей", T._dataset_chunk_dims(files), {(N_FEATURES, 26): 1, (715, 26): 2})


def test_store_dataset_prefers_recompute_merge(tmpdir):
    """_store_dataset копирует свежий recompute-мерж, а не пересобирает из чужих чанков."""
    old_tmp = T.HEURISTIC_DATASET_TMP_DIR
    try:
        T.HEURISTIC_DATASET_TMP_DIR = os.path.join(tmpdir, "heuristic_dataset_tmp")
        recompute_dir = T.HEURISTIC_DATASET_TMP_DIR + "_recompute"
        os.makedirs(recompute_dir, exist_ok=True)
        merged = os.path.join(recompute_dir, "_merged.npz")
        make_chunk(merged, 6, N_FEATURES)
        dataset = [(np.zeros(N_FEATURES, dtype=np.float32), np.ones(26, dtype=np.int8), 0, 1.0)] * 6
        path = os.path.join(tmpdir, "heuristic_dataset.npz")
        T._store_dataset(dataset, path)
        check("obs_dim скопированного файла", T._get_npz_obs_dim(path), N_FEATURES)
        check("число примеров", T._get_npz_n_transitions(path), 6)

        # если мерж не совпадает по количеству — пишем из памяти
        make_chunk(merged, 3, N_FEATURES)
        path2 = os.path.join(tmpdir, "from_memory.npz")
        T._store_dataset(dataset, path2)
        check("несовпадающий мерж не используется", T._get_npz_n_transitions(path2), 6)
    finally:
        T.HEURISTIC_DATASET_TMP_DIR = old_tmp


def test_stale_cache_recomputes_instead_of_merging(tmpdir):
    """Полный сценарий: сырой кэш 2 боёв + устаревшие чанки 715 + нет финального датасета.

    Должно: (1) пересчитать obs текущим кодом, (2) собрать датасет N_FEATURES,
    (3) не трогать/не мешать устаревшие чанки, (4) при повторном вызове — взять готовый
    мерж без повторного пересчёта.
    """
    old_raw, old_tmp = T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR
    try:
        T.HEURISTIC_RAW_CACHE_DIR = os.path.join(tmpdir, "heuristic_raw_chunks")
        T.HEURISTIC_DATASET_TMP_DIR = os.path.join(tmpdir, "heuristic_dataset_tmp")
        os.makedirs(T.HEURISTIC_RAW_CACHE_DIR, exist_ok=True)
        os.makedirs(T.HEURISTIC_DATASET_TMP_DIR, exist_ok=True)

        # сырой кэш: 2 боя, по 2 перехода каждый
        b1 = live_battle(tag="battle-x1"); e1 = [raw_entry(b1, 0), raw_entry(b1, 6)]
        b2 = live_battle(tag="battle-x2", our="garchomp", opp="skarmory", move="dragonclaw")
        e2 = [raw_entry(b2, 6), raw_entry(b2, 1)]
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0000.pkl"), e1, ["battle-x1"])
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0001.pkl"), e2, ["battle-x2"])

        # устаревшие чанки датасета прошлой версии признаков (в логе были 713/715)
        stale = make_chunk(os.path.join(T.HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_0000.npz"), 4, 715)
        stale2 = make_chunk(os.path.join(T.HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_0001.npz"), 4, 713)
        check("устаревшие чанки найдены", T._wrong_dim_dataset_chunks(
            T._list_dataset_chunk_files(), N_FEATURES), {713: 1, 715: 1})

        dataset = T._recompute_from_chunked_cache(2)
        check("пересчёт вернул примеры", len(dataset), 4)
        check("размерность obs — текущая", int(np.asarray(dataset[0][0]).shape[0]), N_FEATURES)
        merged = os.path.join(T.HEURISTIC_DATASET_TMP_DIR + "_recompute", "_merged.npz")
        check_true("merged-файл создан", os.path.exists(merged))
        check("размерность merged", T._get_npz_obs_dim(merged), N_FEATURES)
        check_true("сайдкар с отпечатком записан", os.path.exists(merged + ".meta.json"))
        meta = json.load(open(merged + ".meta.json", encoding="utf-8"))
        check("в сайдкаре число боёв", meta.get("battles_requested"), 2)
        check("в сайдкаре размерность", meta.get("obs_dim"), N_FEATURES)

        # устаревшие чанки не тронуты (их перезапишет только полный сбор)
        check("устаревший чанк 715 не изменён", T._get_npz_obs_dim(stale), 715)
        check("устаревший чанк 713 не изменён", T._get_npz_obs_dim(stale2), 713)

        # повторный вызов: тот же сырой кэш и код -> без пересчёта
        os.remove(merged + ".tmp.npz") if os.path.exists(merged + ".tmp.npz") else None
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dataset2 = T._recompute_from_chunked_cache(2)
        out = buf.getvalue()
        check_true("повтор берёт готовый мерж", "готовый мерж" in out and "пересчёт не нужен" in out, out[:160])
        check("примеров столько же", len(dataset2), 4)

        # меняем сырой кэш (добор боёв) -> кэш пересчёта инвалидируется
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0002.pkl"), e1, ["battle-x3"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T._recompute_from_chunked_cache(3)
        check_true("после добора боёв пересчёт заново", "пересобираю" in buf.getvalue(), buf.getvalue()[:160])
    finally:
        T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR = old_raw, old_tmp


def _recompute_in_isolated_dirs(tmpdir, entries, tags, n_battles=1):
    """Гоняет реальный пересчёт из сырого кэша в изолированных каталогах."""
    old_raw, old_tmp = T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR
    try:
        T.HEURISTIC_RAW_CACHE_DIR = os.path.join(tmpdir, "raw")
        T.HEURISTIC_DATASET_TMP_DIR = os.path.join(tmpdir, "tmp")
        os.makedirs(T.HEURISTIC_RAW_CACHE_DIR, exist_ok=True)
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0000.pkl"), entries, tags)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dataset = T._recompute_from_chunked_cache(n_battles)
        return dataset, buf.getvalue()
    finally:
        T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR = old_raw, old_tmp


def test_user_scenario_dim_mismatch_plus_stale_chunks(tmpdir):
    """Полный сценарий из лога пользователя, но после фикса.

    Есть: старый `models/heuristic_dataset.npz` (713), сырой кэш, устаревшие чанки датасета
    в tmp (713/715) и готовый recompute-мерж (870) без сайдкара.
    Ожидаем: датасет пересобран текущим кодом, записан с размерностью N_FEATURES, устаревшие
    чанки не смотрены, повторного пересчёта нет.
    """
    old_raw, old_tmp = T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR
    try:
        T.HEURISTIC_RAW_CACHE_DIR = os.path.join(tmpdir, "heuristic_raw_chunks")
        T.HEURISTIC_DATASET_TMP_DIR = os.path.join(tmpdir, "heuristic_dataset_tmp")
        os.makedirs(T.HEURISTIC_RAW_CACHE_DIR, exist_ok=True)
        os.makedirs(T.HEURISTIC_DATASET_TMP_DIR, exist_ok=True)

        b1 = live_battle(tag="battle-u1"); e1 = [raw_entry(b1, 0), raw_entry(b1, 6)]
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0000.pkl"), e1, ["battle-u1"])

        # устаревшие чанки датасета (как в логе: 26918x715 и 26918x713)
        make_chunk(os.path.join(T.HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_0000.npz"), 3, 715)
        make_chunk(os.path.join(T.HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_0001.npz"), 3, 713)

        # старый финальный датасет
        path = os.path.join(tmpdir, "heuristic_dataset.npz")
        make_chunk(path, 4, 713)
        # готовый пересчёт текущей размерности (сайдкара ещё нет, файл новее кода)
        merged_dir = T.HEURISTIC_DATASET_TMP_DIR + "_recompute"
        os.makedirs(merged_dir, exist_ok=True)
        merged = os.path.join(merged_dir, "_merged.npz")
        make_chunk(merged, 2, N_FEATURES)
        future = time.time() + 3600
        os.utime(merged, (future, future))

        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dataset = T.collect_or_load_dataset(1, path, force_recollect=False)
        out = buf.getvalue()
        check("датасет вернулся", len(dataset), 2)
        check("файл перезаписан текущей размерностью", T._get_npz_obs_dim(path), N_FEATURES)
        check_true("падения broadcast нет", "could not broadcast" not in out, out[-200:])
        check_true("пересчёт не запускался заново", "adopted" not in out and "использую мерж" in out, out[:300])
        check_true("устаревшие чанки не смотрели", "Мержу" not in out or "_recompute" in out, out[:300])
    finally:
        T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR = old_raw, old_tmp


def test_recorder_stores_team_fusions(tmpdir):
    """HeuristicRecorder пишет командные фьюжн-карты в сырой кэш (9-й элемент записи)."""
    import logging

    from poke_env.battle import Battle

    from agents.players import HeuristicRecorder

    b = Battle(battle_tag="battle-gen9fusionmonsrandombattle-7", username="Me",
               logger=logging.getLogger("quiet"), gen=9)
    for msg in (["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""],
                ["", "start"],
                ["", "switch", "p1a: Aqua", "swampert, L50, M", "362/362"],
                ["", "switch", "p2a: Rock", "steelix, L50, F", "300/300"]):
        b.parse_message(msg)
    b.parse_request({
        "active": [{"moves": [{"move": "Surf", "id": "surf", "pp": 24, "maxpp": 24,
                               "target": "normal", "disabled": False}]}],
        "side": {"id": "p1", "name": "Me", "pokemon": [
            {"ident": "p1: Aqua", "details": "swampert, L50, M", "condition": "362/362",
             "active": True, "stats": {"atk": 257, "def": 257, "spa": 257, "spd": 257, "spe": 257},
             "moves": ["surf"], "baseAbility": "torrent", "item": "leftovers",
             "pokeball": "pokeball", "ability": "torrent"}]}, "rqid": 1})

    rec = HeuristicRecorder(dataset=[], raw_dataset=[],
                            battle_format="gen9fusionmonsrandombattle", start_listening=False)
    rec.choose_move(b)
    check("записано одно решение", len(rec.raw_dataset), 1)
    entry = rec.raw_dataset[0]
    check("сырая запись 9-элементная", len(entry), 9)
    check("командные карты — пара словарей", isinstance(entry[8], tuple) and len(entry[8]), 2)
    check_true("obs записан текущей размерности",
               int(np.asarray(rec.dataset[0][0]).shape[0]) == N_FEATURES)
    fields = T._raw_entry_fields(entry)
    check("хелпер разбирает 9-элементную запись", fields[8], entry[8][0])
    check("хелпер разбирает старую 8-элементную запись", T._raw_entry_fields(entry[:8])[8], None)
    check("хелпер отбрасывает короткие записи", T._raw_entry_fields((1, 2, 3)), None)


def test_recompute_uses_team_fusions(tmpdir):
    """Пересчёт из кэша даёт тот же obs, что живой путь с теми же командными картами.

    Без этого пересобранный датасет расходился с тем, что модель видит в бою (блок урона).
    """
    from agents.features import embed_battle_with_fusion

    huge = {"base_stats": {"hp": 200, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}}
    our_map = {"swampert": huge}
    opp_map = {"steelix": huge}

    battle_a = live_battle(tag="battle-t1", reveal_move="earthquake")
    entry8 = raw_entry(battle_a, 0)
    battle_b = live_battle(tag="battle-t2", reveal_move="earthquake")
    entry9 = raw_entry(battle_b, 0) + ((our_map, opp_map),)

    ds8, _ = _recompute_in_isolated_dirs(os.path.join(tmpdir, "a"), [entry8], ["battle-t1"])
    ds9, out9 = _recompute_in_isolated_dirs(os.path.join(tmpdir, "b"), [entry9], ["battle-t2"])
    obs8, obs9 = np.asarray(ds8[0][0]), np.asarray(ds9[0][0])
    check("обе obs текущей размерности", (int(obs8.shape[0]), int(obs9.shape[0])), (N_FEATURES, N_FEATURES))
    check_true("карты доходят до пересчёта (obs отличаются)",
               not np.array_equal(obs8, obs9),
               f"различий {int((obs8 != obs9).sum())}")
    live = embed_battle_with_fusion(battle_b, None, None, our_team_fusions=our_map,
                                    opp_team_fusions=opp_map)
    check_true("пересчёт совпадает с живым obs при тех же картах",
               bool(np.array_equal(np.asarray(live), obs9)),
               f"максимальное расхождение {float(np.abs(np.asarray(live) - obs9).max()):.6f}")


def test_adopts_merge_without_sidecar_when_newer_than_code(tmpdir):
    """Мерж без сайдкара (посчитан прежней версией кода) принимается, только если он новее кода.

    Это ровно случай пользователя: пересчёт уже был сделан, сайдкара нет (появился только
    сейчас), и без этого правила пришлось бы пересчитывать 800k+ переходов ещё раз.
    """
    old_raw, old_tmp = T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR
    try:
        T.HEURISTIC_RAW_CACHE_DIR = os.path.join(tmpdir, "raw")
        T.HEURISTIC_DATASET_TMP_DIR = os.path.join(tmpdir, "tmp")
        os.makedirs(T.HEURISTIC_RAW_CACHE_DIR, exist_ok=True)
        merged_dir = T.HEURISTIC_DATASET_TMP_DIR + "_recompute"
        os.makedirs(merged_dir, exist_ok=True)
        b1 = live_battle(tag="battle-z1")
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0000.pkl"),
                        [raw_entry(b1, 0), raw_entry(b1, 6)], ["battle-z1"])
        merged = os.path.join(merged_dir, "_merged.npz")
        make_chunk(merged, 2, N_FEATURES)          # без сайдкара
        future = time.time() + 3600
        os.utime(merged, (future, future))          # файл заведомо новее кода признаков

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dataset = T._recompute_from_chunked_cache(1)
        out = buf.getvalue()
        check("примеров из готового мержа", len(dataset), 2)
        check_true("мерж принят без сайдкара", "без сайдкара" in out, out[:200])
        check_true("сайдкар дописан после принятия", os.path.exists(merged + ".meta.json"))

        # а если файл старше кода признаков — пересчитываем (кэш инвалидируется)
        code_mtime = max(os.path.getmtime(os.path.join(os.path.dirname(T.__file__), n))
                         for n in ("features.py", "damage.py", "config.py"))
        os.utime(merged, (code_mtime - 100, code_mtime - 100))
        os.remove(merged + ".meta.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dataset = T._recompute_from_chunked_cache(1)
        check_true("старый мерж без сайдкара не принимается",
                   "старше кода признаков" in buf.getvalue(), buf.getvalue()[:200])
        check("после пересчёта примеры на месте", len(dataset), 2)
    finally:
        T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR = old_raw, old_tmp


def test_collect_or_load_does_not_merge_foreign_chunks(tmpdir):
    """collect_or_load_dataset пишет свежий датасет, а не мержит чужие чанки с диска."""
    old_raw, old_tmp = T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR
    try:
        T.HEURISTIC_RAW_CACHE_DIR = os.path.join(tmpdir, "raw")
        T.HEURISTIC_DATASET_TMP_DIR = os.path.join(tmpdir, "tmp")
        os.makedirs(T.HEURISTIC_RAW_CACHE_DIR, exist_ok=True)
        os.makedirs(T.HEURISTIC_DATASET_TMP_DIR, exist_ok=True)
        b1 = live_battle(tag="battle-y1"); e1 = [raw_entry(b1, 0), raw_entry(b1, 6)]
        write_raw_chunk(os.path.join(T.HEURISTIC_RAW_CACHE_DIR, "raw_chunk_0000.pkl"), e1, ["battle-y1"])
        # чужие чанки в tmp: если бы они попали в мерж, получился бы датасет размерности 715
        make_chunk(os.path.join(T.HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_0000.npz"), 2, 715)
        make_chunk(os.path.join(T.HEURISTIC_DATASET_TMP_DIR, "dataset_chunk_0001.npz"), 2, 713)

        path = os.path.join(tmpdir, "heuristic_dataset.npz")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dataset = T.collect_or_load_dataset(1, path, force_recollect=False)
        out = buf.getvalue()
        check_true("датасет собран", isinstance(dataset, list) and len(dataset) > 0, out[-200:])
        check_true("файл записан", os.path.exists(path))
        check("размерность записанного датасета — текущая", T._get_npz_obs_dim(path), N_FEATURES)
        # мерж допустим только для собственных свежих чанков пересчёта (каталог *_recompute),
        # чужие чанки из models/heuristic_dataset_tmp в него попадать не должны
        merged_lines = [ln for ln in out.splitlines() if ln.startswith("Мержу")]
        check_true("мержатся только свежие чанки пересчёта",
                   all("_recompute" in ln for ln in merged_lines), str(merged_lines))
        check_true("устаревшие чанки датасета остались на месте",
                   T._get_npz_obs_dim(os.path.join(T.HEURISTIC_DATASET_TMP_DIR,
                                                   "dataset_chunk_0000.npz")) == 715)
    finally:
        T.HEURISTIC_RAW_CACHE_DIR, T.HEURISTIC_DATASET_TMP_DIR = old_raw, old_tmp


def main() -> int:
    with tempfile.TemporaryDirectory() as root:
        def sub(name):
            d = os.path.join(root, name)
            os.makedirs(d, exist_ok=True)
            return d
        test_merge_same_dims(sub("merge_same"))
        print("-" * 74)
        test_merge_mixed_obs_dim_gives_clear_error(sub("merge_mixed_obs"))
        print("-" * 74)
        test_merge_mixed_mask_dim_gives_clear_error(sub("merge_mixed_mask"))
        print("-" * 74)
        test_wrong_dim_chunks_detection(sub("wrong_dim"))
        print("-" * 74)
        test_store_dataset_prefers_recompute_merge(sub("store"))
        print("-" * 74)
        test_stale_cache_recomputes_instead_of_merging(sub("stale"))
        print("-" * 74)
        test_adopts_merge_without_sidecar_when_newer_than_code(sub("adopt"))
        print("-" * 74)
        test_user_scenario_dim_mismatch_plus_stale_chunks(sub("user_scenario"))
        print("-" * 74)
        test_recorder_stores_team_fusions(sub("recorder"))
        print("-" * 74)
        test_recompute_uses_team_fusions(sub("fusions"))
        print("-" * 74)
        test_collect_or_load_does_not_merge_foreign_chunks(sub("collect"))
    print("-" * 74)
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)} -> {FAILED}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
