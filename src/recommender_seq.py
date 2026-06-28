import _threadcap  # noqa: F401

import time
import argparse

import numpy as np

from data_loader import load_ratings
from cf_core import build_model, recommend_batch

TOP_K = 40
TOP_N = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("[seq] Загрузка данных ...")
    data = load_ratings()
    R, mask = data["R"], data["mask"]
    print(f"[seq] Матрица: {R.shape[0]} пользователей x {R.shape[1]} фильмов")

    t0 = time.perf_counter()
    Rc, mask_f, user_means, norms = build_model(R, mask)
    t_model = time.perf_counter() - t0

    targets = np.arange(R.shape[0])
    if args.limit > 0:
        targets = targets[:args.limit]

    t1 = time.perf_counter()
    results = recommend_batch(targets, Rc, mask_f, user_means, norms, top_k=TOP_K, top_n=TOP_N)
    t_rec = time.perf_counter() - t1

    print(f"[seq] Построение модели:   {t_model:.3f} c")
    print(f"[seq] Расчёт рекомендаций: {t_rec:.3f} c  ({len(targets)} пользователей)")
    print(f"SEQ_TOTAL {t_rec:.6f}")

    if args.users > 0:
        _print_recommendations(results[:args.users], data)


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
