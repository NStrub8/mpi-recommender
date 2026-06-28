import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from cf_core import build_model, recommend_batch
from recommender_mpi import split_indices


def make_synthetic(n_users=120, n_items=200, density=0.1, seed=0):
    rng = np.random.default_rng(seed)
    R = np.zeros((n_users, n_items), dtype=np.float32)
    mask = rng.random((n_users, n_items)) < density
    R[mask] = rng.integers(1, 6, size=int(mask.sum())).astype(np.float32)
    for u in range(n_users):
        if not mask[u].any():
            j = rng.integers(0, n_items)
            mask[u, j] = True
            R[u, j] = 3.0
    return R, mask


def test_split_indices_partitions_all():
    for n in (1, 7, 100, 610):
        for size in (1, 2, 3, 4, 8):
            bounds = split_indices(n, size)
            covered = []
            for lo, hi in bounds:
                covered.extend(range(lo, hi))
            assert covered == list(range(n))
            sizes = [hi - lo for lo, hi in bounds]
            assert max(sizes) - min(sizes) <= 1
    print("[ok] split_indices: разбиение полное и сбалансированное")


def test_distributed_equals_sequential():
    R, mask = make_synthetic()
    Rc, mask_f, user_means, norms = build_model(R, mask)

    n_users = R.shape[0]
    ref = recommend_batch(np.arange(n_users), Rc, mask_f, user_means, norms, top_k=20, top_n=10)
    ref_map = {u: (items, scores) for u, items, scores in ref}

    for size in (2, 3, 4, 8):
        merged = {}
        for lo, hi in split_indices(n_users, size):
            part = recommend_batch(np.arange(lo, hi), Rc, mask_f, user_means, norms, top_k=20, top_n=10)
            for u, items, scores in part:
                merged[u] = (items, scores)

        assert set(merged) == set(ref_map)
        for u in ref_map:
            ri, rs = ref_map[u]
            mi, ms = merged[u]
            assert np.array_equal(ri, mi)
            assert np.allclose(rs, ms)
    print("[ok] распределённый результат идентичен последовательному (size=2,3,4,8)")


if __name__ == "__main__":
    test_split_indices_partitions_all()
    test_distributed_equals_sequential()
    print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ")
