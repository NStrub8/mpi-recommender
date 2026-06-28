import _threadcap  # noqa: F401

import argparse

import numpy as np
from mpi4py import MPI

from cf_core import build_model, recommend_batch

TOP_K = 40
TOP_N = 10


def split_indices(n, size):
    base, extra = divmod(n, size)
    bounds = []
    start = 0
    for r in range(size):
        length = base + (1 if r < extra else 0)
        bounds.append((start, start + length))
        start += length
    return bounds


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = None
    if rank == 0:
        from data_loader import load_ratings
        print(f"[mpi] Процессов: {size}")
        print("[mpi] rank 0: загрузка данных ...")
        data = load_ratings()
        Rc, mask_f, user_means, norms = build_model(data["R"], data["mask"])
        shape = np.array(Rc.shape, dtype=np.int64)
        print(f"[mpi] rank 0: матрица {shape[0]} x {shape[1]}, рассылаю модель ...")
    else:
        Rc = mask_f = user_means = norms = None
        shape = np.empty(2, dtype=np.int64)

    comm.Barrier()
    t_start = MPI.Wtime()

    comm.Bcast(shape, root=0)
    n_users, n_items = int(shape[0]), int(shape[1])

    if rank != 0:
        Rc = np.empty((n_users, n_items), dtype=np.float32)
        mask_f = np.empty((n_users, n_items), dtype=np.float32)
        user_means = np.empty(n_users, dtype=np.float32)
        norms = np.empty(n_users, dtype=np.float32)

    comm.Bcast(Rc, root=0)
    comm.Bcast(mask_f, root=0)
    comm.Bcast(user_means, root=0)
    comm.Bcast(norms, root=0)

    total = n_users if args.limit <= 0 else min(args.limit, n_users)
    lo, hi = split_indices(total, size)[rank]
    my_targets = np.arange(lo, hi)

    t_local0 = MPI.Wtime()
    local_results = recommend_batch(my_targets, Rc, mask_f, user_means, norms, top_k=TOP_K, top_n=TOP_N)
    t_local = MPI.Wtime() - t_local0

    all_results = comm.gather(local_results, root=0)
    all_local_times = comm.gather(t_local, root=0)

    comm.Barrier()
    t_total = MPI.Wtime() - t_start

    if rank == 0:
        flat = [item for chunk in all_results for item in chunk]
        print(f"[mpi] Обработано пользователей: {len(flat)}")
        print(f"[mpi] Макс. локальное время расчёта: {max(all_local_times):.3f} c")
        print(f"[mpi] Общее время (bcast+расчёт+gather): {t_total:.3f} c")
        print(f"MPI_TOTAL {size} {max(all_local_times):.6f}")

        if args.users > 0:
            _print_recommendations(flat[:args.users], data)


def _print_recommendations(results, data):
    movie_ids = data["movie_ids"]
    user_ids = data["user_ids"]
    titles = data["movie_title"]
    print("\n=== Примеры рекомендаций ===")
    for u, items, scores in results:
        print(f"\nПользователь {user_ids[u]}:")
        for it, sc in zip(items, scores):
            mid = int(movie_ids[it])
            print(f"   {sc:5.2f}  {titles.get(mid, f'movie {mid}')}")


if __name__ == "__main__":
    main()
