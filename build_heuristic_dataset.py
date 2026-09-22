#!/usr/bin/env python
"""
Только сбор heuristic_dataset — без BC/PPO/RL.

Что делает:
  SimpleHeuristics vs SimpleHeuristics (poke-env) -> embed_battle_with_fusion (N_FEATURES=715)
  -> (obs [715], mask [9], action [0..26], ret ±30*0.99**remaining)
  -> models/heuristic_dataset.npz (+ кэш models/heuristic_raw_chunks/ и models/heuristic_dataset_tmp/)

В отличие от `agents.policy_player --pretrain-battles`:
  - не создаёт PPO, не грузит policy, не делает pretrain_policy_bc, не запускает RL
  - можно собрать 1k/10k/50k и потом использовать в любой тренировке через --dataset-path

Запуск:
  python build_heuristic_dataset.py --n-battles 1000
  python build_heuristic_dataset.py --n-battles 5000 --output models/heuristic_dataset_5k.npz --chunk-size 1000
  python -m agents.collect_heuristic_simple --n-battles 1000  # алиас (тот же код)

Требует: poke_env, numpy, уже пропатченный agents/config.py (BATTLE_FORMAT, N_FEATURES)
"""
import argparse
import os
import sys

# позволяет запускать и как `python build_heuristic_dataset.py` и как `python -m build_heuristic_dataset`
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.training import collect_heuristic_dataset, collect_or_load_dataset
from agents.config import N_FEATURES


def main():
    ap = argparse.ArgumentParser(description="ТОЛЬКО сбор heuristic_dataset (без обучения)")
    ap.add_argument("--n-battles", type=int, default=1000, help="Сколько боев SimpleHeuristics vs SimpleHeuristics (обычно 8-25 ходов/бой)")
    ap.add_argument("--output", "-o", type=str, default="models/heuristic_dataset.npz", help="Куда писать npz")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Чанк для больших n (>1000) — пишет на диск пачками, resume-safe")
    ap.add_argument("--force", action="store_true", help="Игнорировать кэш, пересобрать с нуля (удаляет raw_chunks/tmp)")
    ap.add_argument("--no-cache", action="store_true", help="Не использовать кэш вообще")
    ap.add_argument("--no-resume", action="store_true", help="Не возобновлять прерванные чанки")
    args = ap.parse_args()

    print(f"[heuristic-only] N_FEATURES={N_FEATURES} n_battles={args.n_battles} -> {args.output} chunk={args.chunk_size}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # быстрый путь для маленьких сборок без чанков — но надёжнее всегда идти через chunked collector
    # используем прямой collect_heuristic_dataset чтобы прокинуть chunk-size/cache флаги,
    # затем мержим чанки в output (экономит RAM vs save_dataset)

    # если пользователь хочет дефолтный chunk=1000 и кэш — можно делегировать в collect_or_load_dataset (там dim-check + merge)
    use_direct = (args.chunk_size != 1000 or args.no_cache or args.no_resume or args.force and args.n_battles > 1000)

    if not use_direct:
        # самый надёжный путь: dim mismatch (713->715) пересоберёт из raw_cache без новых боев
        print(f"[heuristic-only] via collect_or_load_dataset(n_battles={args.n_battles}, path={args.output}, force={args.force})")
        dataset = collect_or_load_dataset(n_battles=args.n_battles, path=args.output, force_recollect=args.force)
        n = len(dataset) if hasattr(dataset, "__len__") else -1
        print(f"[heuristic-only] done: {n} примеров -> {args.output}")
    else:
        print(f"[heuristic-only] via collect_heuristic_dataset(n_battles={args.n_battles}, chunk_size={args.chunk_size}, use_cache={not args.no_cache}, resume={not args.no_resume}, force={args.force})")
        dataset = collect_heuristic_dataset(
            n_battles=int(args.n_battles),
            force_recollect=bool(args.force),
            use_cache=not bool(args.no_cache),
            chunk_size=int(args.chunk_size),
            resume=not bool(args.no_resume),
        )
        # dataset уже в памяти (list) + чанки на диске в models/heuristic_dataset_tmp/
        # мержим чанки в финальный npz (память 3.6GB для 50k вместо 61GB)
        try:
            from agents.training import _list_dataset_chunk_files, _merge_dataset_chunks, save_dataset
            chunk_files = _list_dataset_chunk_files()
            if len(chunk_files) > 0:
                print(f"[heuristic-only] merging {len(chunk_files)} chunks -> {args.output}")
                _merge_dataset_chunks(chunk_files, args.output)
                print(f"[heuristic-only] merged: {args.output} ({len(dataset) if dataset else 0} in-mem, see npz)")
            elif dataset and len(dataset) > 0:
                print(f"[heuristic-only] no chunks, saving list directly -> {args.output}")
                save_dataset(dataset, args.output)
            else:
                print("[heuristic-only] WARNING: dataset empty, nothing saved")
        except Exception as e:
            print(f"[heuristic-only] merge failed: {e}, fallback to save_dataset")
            import traceback; traceback.print_exc()
            if dataset and len(dataset) > 0:
                from agents.training import save_dataset
                save_dataset(dataset, args.output)

    # финальная проверка
    try:
        import numpy as np
        sz = os.path.getsize(args.output)
        d = np.load(args.output, mmap_mode="r")
        has_ret = "ret" in d
        print(f"[heuristic-only] file: {args.output} {sz/1024/1024:.1f} MB  obs{d['obs'].shape} mask{d['mask'].shape} ret={has_ret} N_FEATURES check {d['obs'].shape[1]}=={N_FEATURES} -> {'OK' if d['obs'].shape[1]==N_FEATURES else 'MISMATCH'}")
        if has_ret:
            print(f"  ret mean {float(d['ret'].mean()):.2f} min {float(d['ret'].min()):.1f} max {float(d['ret'].max()):.1f}")
    except Exception as e:
        print(f"[heuristic-only] verify failed: {e}")

    print("[heuristic-only] done — никакого обучения не было. Используй дальше: python -m agents.policy_player --dataset-path", args.output, "--epochs 5 ...")


if __name__ == "__main__":
    main()
