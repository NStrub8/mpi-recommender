import numpy as np


def build_model(R, mask):
    R = np.ascontiguousarray(R, dtype=np.float32)
    mask_f = mask.astype(np.float32)

    counts = mask_f.sum(axis=1)
    counts[counts == 0] = 1.0
    user_means = (R.sum(axis=1) / counts).astype(np.float32)

    Rc = (R - user_means[:, None]) * mask_f
    norms = np.sqrt((Rc * Rc).sum(axis=1)).astype(np.float32)
    norms[norms == 0] = 1e-8

    return Rc, mask_f, user_means, norms


def recommend_user(u, Rc, mask_f, user_means, norms, top_k=40, top_n=10):
    sims = (Rc @ Rc[u]) / (norms * norms[u])
    sims[u] = 0.0

    if top_k < sims.shape[0]:
        idx = np.argpartition(-np.abs(sims), top_k)[:top_k]
        neigh = np.zeros_like(sims)
        neigh[idx] = sims[idx]
        sims = neigh

    numerator = sims @ Rc
    denominator = np.abs(sims) @ mask_f
    denominator[denominator == 0] = 1e-8

    preds = user_means[u] + numerator / denominator
    preds[mask_f[u] > 0] = -np.inf

    n = min(top_n, preds.shape[0])
    top_idx = np.argpartition(-preds, n - 1)[:n]
    top_idx = top_idx[np.argsort(-preds[top_idx])]
    scores = np.clip(preds[top_idx], 0.5, 5.0)
    return top_idx, scores


def recommend_batch(user_indices, Rc, mask_f, user_means, norms, top_k=40, top_n=10):
    out = []
    for u in user_indices:
        items, scores = recommend_user(u, Rc, mask_f, user_means, norms, top_k=top_k, top_n=top_n)
        out.append((int(u), items, scores))
    return out
