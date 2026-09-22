"""
Сбор heuristic_dataset без тренировки модели.

Использование:
  python -m agents.collect_heuristic_simple --n-battles 1000 --output models/heuristic_dataset.npz
  python -m agents.collect_heuristic_simple --n-battles 5000 --chunk-size 1000 --force
  python collect_heuristic.py --n-battles 200 --output models/heuristic_dataset_200.npz

Отличие от `policy_player --pretrain-battles`:
  - не создаёт PPO, не грузит policy, не делает BC
  - только бьёт SimpleHeuristics vs SimpleHeuristics, эмбедит obs (N_FEATURES) и сохраняет npz
  - поддерживает chunked кэш (models/heuristic_raw_chunks/, models/heuristic_dataset_tmp/) и resume
  - пересобирает obs из сырого кэша если N_FEATURES изменился

Датасет формат: npz с keys obs [N,N_FEATURES] float32, mask [N,9] int8, action [N] int64, ret [N] float32
  ret = ±30 * 0.99**(steps_remaining)  (победа/поражение)
"""

import argparse
import os
import sys

# позволяет запускать и как `python agents/collect_heuristic_simple.py` и как `python -m agents.collect_heuristic_simple`
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.training import collect_or_load_dataset, collect_heuristic_dataset
from agents.config import N_FEATURES


def main():
    ap = argparse.ArgumentParser(description="Сбор heuristic_dataset (SimpleHeuristics vs SimpleHeuristics) без тренировки")
    ap.add_argument("--n-battles", type=int, default=1000, help="Сколько боёв наиграть (по 8-25 ходов каждый, ~8 переходов/бой)")
    ap.add_argument("--output", "-o", type=str, default="models/heuristic_dataset.npz", help="Куда сохранить npz")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Размер чанка для больших сборок (>1000) — пишет на диск пачками")
    ap.add_argument("--force", action="store_true", help="Пересобрать заново, игнорируя кэш (удаляет raw_chunks/tmp)")
    ap.add_argument("--no-cache", action="store_true", help="Не использовать кэш, собрать с нуля (без chunked reuse)")
    ap.add_argument("--no-resume", action="store_true", help="Не возобновлять прерванные чанки, начать с нуля")
    ap.add_argument("--quick", action="store_true", help="Быстрый путь без чанков (для n_battles <= 1000, быстрее, но без resume)")
    args = ap.parse_args()

    print(f"N_FEATURES={N_FEATURES} output={args.output} n_battles={args.n_battles} chunk_size={args.chunk_size}")

    # quick path для маленьких сборок — напрямую без chunked merge (1 батч)
    if args.quick and args.n_battles <= 1000 and not args.force and not args.no_cache:
        print("quick: собираю одним батчем без чанков...")
        from agents.training import collect_heuristic_dataset as chd
        dataset = chd(n_battles=args.n_battles, force_recollect=args.force, use_cache=not args.no_cache, chunk_size=args.chunk_size, resume=not args.no_resume)
        # collect_heuristic_dataset in quick mode не сохраняет merged финальный файл сам — сохраняем через collect_or_load логику
        # поэтому просто делегируем в collect_or_load для мержа/сохранения
        # если датасет уже >0, сохраним
        if dataset and len(dataset) > 0:
            from agents.training import save_dataset
            # если чанки уже есть, лучше мержить их, а не сохранять список (экономит RAM)
            from agents.training import _list_dataset_chunk_files, _merge_dataset_chunks
            import glob
            chunk_files = _list_dataset_chunk_files()
            if len(chunk_files) > 1:
                print(f"Найдены чанки {len(chunk_files)}, мержу в {args.output} вместо save_dataset")
                _merge_dataset_chunks(chunk_files, args.output)
            else:
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                save_dataset(dataset, args.output)
        print(f"Готово: {args.output}")
        return

    # основной путь — через collect_or_load_dataset (поддерживает chunked, resume, dim-check, merge)
    # collect_or_load_dataset сам вызовет collect_heuristic_dataset с chunk_size из DEFAULT_CHUNK_SIZE,
    # но мы хотим прокинуть chunk_size — поэтому вызываем collect_heuristic напрямую если нужен кастомный chunk
    if args.chunk_size != 1000 or args.no_cache or args.no_resume:
        # кастомный chunk_size / cache flags — идём напрямую
        print(f"Сбор напрямую collect_heuristic_dataset(chunk_size={args.chunk_size}, use_cache={not args.no_cache}, resume={not args.no_resume})")
        dataset = collect_heuristic_dataset(
            n_battles=args.n_battles,
            force_recollect=args.force,
            use_cache=not args.no_cache,
            chunk_size=int(args.chunk_size),
            resume=not args.no_resume,
        )
        # после chunked сбора датасет уже в tmp чанках, нужно смержить в output
        if args.output:
            from agents.training import _list_dataset_chunk_files, _merge_dataset_chunks, save_dataset
            chunk_files = _list_dataset_chunk_files()
            if len(chunk_files) > 0:
                # если датасет большой — мержим чанки (экономит 3.6GB вместо 61GB)
                # если запрошено ровно n_battles, а в чанках больше — collect_heuristic уже обрезал через recompute
                print(f"Мержу {len(chunk_files)} чанков в {args.output} ...")
                _merge_dataset_chunks(chunk_files, args.output)
                # также можно оставить dataset в памяти для лога
                print(f"Готово: {args.output} ({len(dataset) if dataset else 0} примеров в памяти, на диске см. npz)")
            else:
                # маленький датасет без чанков — сохраняем список
                if dataset and len(dataset) > 0:
                    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                    save_dataset(dataset, args.output)
                    print(f"Готово: {args.output} ({len(dataset)} примеров)")
                else:
                    print(f"WARNING: датасет пустой, файл не создан")
        else:
            print(f"Собрано {len(dataset) if dataset else 0} примеров (output не задан, на диск не писал)")
        return

    # дефолт — самый надёжный путь с dim-check и merge
    print(f"Сбор через collect_or_load_dataset({args.n_battles}, {args.output}, force={args.force}) ...")
    dataset = collect_or_load_dataset(n_battles=args.n_battles, path=args.output, force_recollect=args.force)
    print(f"Готово: {args.output} ({len(dataset) if hasattr(dataset, '__len__') else 'unknown'} примеров)")
    # доп. инфа о размере файла
    try:
        import numpy as np
        sz = os.path.getsize(args.output)
        print(f"  файл: {sz/1024/1024:.1f} MB")
        d = np.load(args.output, mmap_mode='r')
        print(f"  keys: {list(d.keys())} obs {d['obs'].shape} mask {d['mask'].shape} has_ret {'ret' in d}")
    except Exception as e:
        print(f"  (не удалось прочитать npz: {e})")


if __name__ == "__main__":
    main()
