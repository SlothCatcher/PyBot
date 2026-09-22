#!/usr/bin/env python
"""
heuristic_dataset_only.py — ТОЛЬКО сбор датасета, без BC и без RL.

- бьёт SimpleHeuristics vs SimpleHeuristics
- эмбедит obs 715 (is_tera) + mask 9 + action + ret
- сохраняет models/heuristic_dataset.npz
- НИЧЕГО не тренирует (нет PPO, нет pretrain_policy_bc, нет ppo.learn)

Запуск:
  python heuristic_dataset_only.py --n-battles 1000
  python heuristic_dataset_only.py --n-battles 5000 --output models/heuristic_dataset_5k.npz
  python heuristic_dataset_only.py --n-battles 1000 --force
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.training import collect_heuristic_dataset
from agents.training import save_dataset, _list_dataset_chunk_files, _merge_dataset_chunks
from agents.config import N_FEATURES

def main():
    p = argparse.ArgumentParser(description="Сбор heuristic_dataset БЕЗ обучения")
    p.add_argument("--n-battles", type=int, default=1000, help="Кол-во боев")
    p.add_argument("--output", type=str, default="models/heuristic_dataset.npz", help="Куда писать npz")
    p.add_argument("--chunk-size", type=int, default=1000, help="Чанк для больших сборок")
    p.add_argument("--force", action="store_true", help="Пересобрать с нуля")
    a = p.parse_args()

    print(f"[only-collect] N_FEATURES={N_FEATURES} n_battles={a.n_battles} -> {a.output}")
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)

    # только сбор — никакого PPO/BC
    dataset = collect_heuristic_dataset(
        n_battles=int(a.n_battles),
        force_recollect=bool(a.force),
        use_cache=True,
        chunk_size=int(a.chunk_size),
        resume=True,
    )

    # датасет уже в чанках на диске — мержим в финальный файл (экономит RAM)
    chunks = _list_dataset_chunk_files()
    if chunks:
        print(f"[only-collect] merging {len(chunks)} chunks -> {a.output}")
        _merge_dataset_chunks(chunks, a.output)
    elif dataset:
        print(f"[only-collect] saving {len(dataset)} examples -> {a.output}")
        save_dataset(dataset, a.output)
    else:
        print("[only-collect] WARNING: пустой датасет")

    # проверка
    try:
        import numpy as np
        d = np.load(a.output, mmap_mode="r")
        print(f"[only-collect] done: {a.output} obs{d['obs'].shape} mask{d['mask'].shape} ret={'ret' in d} {os.path.getsize(a.output)/1024/1024:.1f} MB")
    except Exception as e:
        print(f"[only-collect] verify: {e}")
    print("[only-collect] ГОТОВО — обучения не было")

if __name__ == "__main__":
    main()
