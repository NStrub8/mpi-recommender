import os
import io
import zipfile
import urllib.request

import numpy as np

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATASET_DIR = os.path.join(DATA_DIR, "ml-latest-small")
RATINGS_CSV = os.path.join(DATASET_DIR, "ratings.csv")
MOVIES_CSV = os.path.join(DATASET_DIR, "movies.csv")


def ensure_dataset():
    if os.path.exists(RATINGS_CSV):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[data] Скачиваю датасет MovieLens из {MOVIELENS_URL} ...")
    with urllib.request.urlopen(MOVIELENS_URL, timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(DATA_DIR)
    print(f"[data] Готово: {DATASET_DIR}")


def _read_csv_skip_header(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines[1:]


def load_ratings():
    ensure_dataset()

    rows = _read_csv_skip_header(RATINGS_CSV)
    user_raw = np.empty(len(rows), dtype=np.int64)
    movie_raw = np.empty(len(rows), dtype=np.int64)
    rating_raw = np.empty(len(rows), dtype=np.float32)

    for k, line in enumerate(rows):
        u, m, r, _ = line.split(",", 3)
        user_raw[k] = int(u)
        movie_raw[k] = int(m)
        rating_raw[k] = float(r)

    user_ids, user_idx = np.unique(user_raw, return_inverse=True)
    movie_ids, movie_idx = np.unique(movie_raw, return_inverse=True)

    R = np.zeros((user_ids.shape[0], movie_ids.shape[0]), dtype=np.float32)
    mask = np.zeros_like(R, dtype=np.bool_)
    R[user_idx, movie_idx] = rating_raw
    mask[user_idx, movie_idx] = True

    movie_title = {}
    if os.path.exists(MOVIES_CSV):
        for line in _read_csv_skip_header(MOVIES_CSV):
            mid, rest = line.split(",", 1)
            if rest.startswith('"'):
                title = rest[1:].rsplit('"', 1)[0]
            else:
                title = rest.rsplit(",", 1)[0]
            movie_title[int(mid)] = title

    return {
        "R": R,
        "mask": mask,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "movie_title": movie_title,
    }


if __name__ == "__main__":
    data = load_ratings()
    R = data["R"]
    print(f"Пользователей: {R.shape[0]}")
    print(f"Фильмов:       {R.shape[1]}")
    print(f"Оценок:        {int(data['mask'].sum())}")
