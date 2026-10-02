"""Supervised flat model and neighborhood retrieval.

``flat`` fits a linear model on every training label. ``rfm`` does not fit
weights: it standardizes the neighborhood features and predicts from the
nearest labeled rows (in-context retrieval).
"""

from __future__ import annotations

import numpy as np

from .metrics import auroc, hit_rate, mae


def _standardize(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (query - mu) / sd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, n_iter: int = 30) -> np.ndarray:
    """L2-regularized logistic regression, class-balanced Newton steps.

    Returns weights including a bias term in column 0. The bias is not
    regularized.
    """
    y = np.asarray(y, dtype=np.float64)
    Xb = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    n, d = Xb.shape
    w = np.zeros(d)
    # Each class contributes the same total weight.
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / neg) * n
    for _ in range(n_iter):
        p = _sigmoid(Xb @ w)
        hess_w = sw * p * (1.0 - p)
        H = Xb.T @ (Xb * hess_w[:, None])
        g = Xb.T @ (sw * (p - y))
        reg = np.eye(d) * l2
        reg[0, 0] = 0.0
        H = H + reg
        g = g + reg @ w
        try:
            step = np.linalg.solve(H + 1e-6 * np.eye(d), g)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    return _sigmoid(Xb @ w)


def fit_ridge(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> np.ndarray:
    """Ridge regression. Bias in column 0 is not regularized."""
    y = np.asarray(y, dtype=np.float64)
    Xb = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    d = Xb.shape[1]
    reg = np.eye(d) * l2
    reg[0, 0] = 0.0
    A = Xb.T @ Xb + reg
    b = Xb.T @ y
    return np.linalg.solve(A, b)


def predict_ridge(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    return Xb @ w


def subsample_context(
    X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample for classification; plain subsample for regression."""
    y = np.asarray(y)
    if len(y) <= n:
        return X, y
    classes = np.unique(y)
    # Regression labels are continuous: more than a handful of distinct values.
    if y.dtype.kind == "f" and len(classes) > 12:
        idx = rng.choice(len(y), size=n, replace=False)
        return X[idx], y[idx]
    per = max(1, n // len(classes))
    picks: list[np.ndarray] = []
    for c in classes:
        pool = np.flatnonzero(y == c)
        take = min(len(pool), per)
        picks.append(rng.choice(pool, size=take, replace=False))
    idx = np.concatenate(picks)
    if len(idx) < n:
        rest = np.setdiff1d(np.arange(len(y)), idx, assume_unique=False)
        extra_n = min(len(rest), n - len(idx))
        if extra_n:
            idx = np.concatenate([idx, rng.choice(rest, size=extra_n, replace=False)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def context_scores(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    k: int = 31,
) -> np.ndarray:
    """Distance-weighted label average of the ``k`` nearest context rows."""
    if len(X_context) == 0:
        raise ValueError("context set is empty")
    Xc, Xq = _standardize(X_context, X_query)
    y = np.asarray(y_context, dtype=np.float64)
    # (n_query, n_context) squared distances. Context is capped by the caller.
    d2 = ((Xq[:, None, :] - Xc[None, :, :]) ** 2).sum(axis=2)
    kk = min(k, len(y))
    if kk == len(y):
        nn = np.broadcast_to(np.arange(kk), (len(Xq), kk)).copy()
        nd = d2
    else:
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        nd = np.take_along_axis(d2, nn, axis=1)
    bw = float(np.median(nd))
    if bw < 1e-8:
        bw = 1e-8
    weights = np.exp(-nd / bw)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum = np.where(weights_sum < 1e-12, 1.0, weights_sum)
    picked = y[nn]
    return (weights * picked).sum(axis=1) / weights_sum.ravel()


def supervised_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    Xtr, Xte = _standardize(X_train, X_test)
    if kind == "binary":
        w = fit_logistic(Xtr, y_train)
        return predict_logistic(w, Xte)
    if kind == "regression":
        w = fit_ridge(Xtr, y_train)
        return np.clip(predict_ridge(w, Xte), 0.0, None)
    raise ValueError(f"unknown kind {kind}")


def retrieval_class(
    keys_context: np.ndarray,
    y_context: np.ndarray,
    keys_query: np.ndarray,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    """Majority label among context rows with the same discrete key.

    ``keys_*`` are integer vectors (packed retrieval keys). ``fallback`` is
    used when a query key was never seen.
    """
    y_context = np.asarray(y_context, dtype=np.int64)
    buckets: dict[int, list[int]] = {}
    for key, label in zip(keys_context.tolist(), y_context.tolist(), strict=True):
        buckets.setdefault(int(key), []).append(int(label))
    majority: dict[int, int] = {}
    for key, labels in buckets.items():
        vals, counts = np.unique(labels, return_counts=True)
        majority[key] = int(vals[np.argmax(counts)])
    out = np.array(fallback, dtype=np.int64, copy=True)
    for i, key in enumerate(keys_query.tolist()):
        if int(key) in majority:
            out[i] = majority[int(key)]
    return out


def score_task(kind: str, metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "auroc":
        return auroc(y_true, y_pred)
    if metric == "mae":
        return mae(y_true, y_pred)
    if metric == "hit@1":
        return hit_rate(y_true, y_pred)
    raise ValueError(metric)
