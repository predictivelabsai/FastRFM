"""Run flat, rfm, and (optionally) kumo on the same labeled rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import kumo
from .features import (
    FLAT_CUSTOMER,
    Indexed,
    customer_matrix,
    customer_snapshot,
    notify_keys,
    payment_matrix,
)
from .models import context_scores, retrieval_class, score_task, subsample_context, supervised_predict
from .tasks import TASKS, Examples, build_examples
from .warehouse import Warehouse

MODEL_NOTES = {
    "flat": "logistic or ridge on recency, frequency, and monetary value. Every training label is used. No foreign-key joins.",
    "rfm": "in-context retrieval over neighborhood features (tickets, refunds, devices, constructors). A capped set of labeled rows, no fitted weights. Not Kumo's network.",
    "kumo": "hosted NVIDIA Kumo Relational (ex-KumoRFM) on the NVIDIA API Catalog; legacy kumoai SDK for non-nvapi keys. Needs KUMO_API_KEY. Weights stay on NVIDIA's side.",
}


def evaluate_warehouse(
    warehouse: Warehouse,
    tasks: list[str],
    models: list[str],
    *,
    context_size: int = 64,
    k: int = 31,
    seed: int = 0,
    kumo_max_calls: int = 8,
) -> dict[str, Any]:
    index = Indexed.build(warehouse.tables)
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "source": "synthetic",
        "seed": warehouse.seed,
        "context_size": context_size,
        "anchors": {name: ts.isoformat() for name, ts in warehouse.anchors.items()},
        "models": {name: MODEL_NOTES[name] for name in models},
        "tasks": {},
    }
    for task in tasks:
        report["tasks"][task] = _one_synthetic(
            warehouse, index, task, models, context_size=context_size, k=k, rng=rng, kumo_max_calls=kumo_max_calls
        )
    return report


def _one_synthetic(
    warehouse: Warehouse,
    index: Indexed,
    task: str,
    models: list[str],
    *,
    context_size: int,
    k: int,
    rng: np.random.Generator,
    kumo_max_calls: int,
) -> dict[str, Any]:
    spec = TASKS[task]
    examples = build_examples(warehouse, task)
    train = examples.split("train")
    test = examples.split("test")
    result: dict[str, Any] = {
        "question": spec["question"],
        "metric": spec["metric"],
        "kind": spec["kind"],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate": _positive_rate(test["label"].to_numpy(), spec["kind"]),
        "models": {},
    }
    if len(train) == 0 or len(test) == 0:
        result["error"] = "empty train or test split"
        return result

    predictions: dict[str, np.ndarray] = {}
    if task == "notify":
        predictions.update(_notify(index, train, test, models, rng))
    else:
        predictions.update(
            _tabular(index, train, test, task, spec["entity"], spec["kind"], models, context_size, k, rng)
        )

    y = test["label"].to_numpy(dtype=np.float64)
    for name, pred in predictions.items():
        result["models"][name] = _model_block(spec["kind"], spec["metric"], y, pred, context_size if name == "rfm" else len(train))

    if "kumo" in models:
        result["models"]["kumo"] = _run_kumo(warehouse.tables, task, test, spec, kumo_max_calls)

    if task in ("churn", "fraud") and "flat" in predictions and "rfm" in predictions:
        result["story"] = _story(index, task, test, predictions["flat"], predictions["rfm"])
    return result


def _tabular(index, train, test, task, entity, kind, models, context_size, k, rng) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    y_train = train["label"].to_numpy(dtype=np.float64)
    if entity == "customer":
        def matrix(frame, mode):
            return customer_matrix(index, frame["entity_id"].to_numpy(), frame["anchor"].to_numpy(), mode=mode)
    else:
        def matrix(frame, mode):
            return payment_matrix(index, frame["entity_id"].to_numpy(), mode=mode)

    if "flat" in models:
        out["flat"] = supervised_predict(matrix(train, "flat"), y_train, matrix(test, "flat"), kind=kind)
    if "rfm" in models:
        Xtr = matrix(train, "rfm")
        Xte = matrix(test, "rfm")
        Xc, yc = subsample_context(Xtr, y_train, context_size, rng)
        out["rfm"] = context_scores(Xc, yc, Xte, k=k)
        if kind == "regression":
            out["rfm"] = np.clip(out["rfm"], 0.0, None)
    return out


def _notify(index: Indexed, train: pd.DataFrame, test: pd.DataFrame, models: list[str], rng) -> dict[str, np.ndarray]:
    del rng  # retrieval uses every training label; the cap is the population
    keys_tr, mode_tr = notify_keys(index, train["entity_id"].to_numpy(), train["anchor"].to_numpy())
    keys_te, mode_te = notify_keys(index, test["entity_id"].to_numpy(), test["anchor"].to_numpy())
    y_train = train["label"].to_numpy(dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    if "flat" in models:
        out["flat"] = mode_te.astype(np.int64)
    if "rfm" in models:
        # Same region and the same "had a recent meal ticket" bit. Neighbors
        # voted with the restaurant they actually ordered next.
        out["rfm"] = retrieval_class(keys_tr, y_train, keys_te, fallback=mode_te)
    return out


def _run_kumo(tables, task, test, spec, kumo_max_calls: int) -> dict[str, Any]:
    try:
        raw = kumo.predict_task(
            tables,
            task,
            test["entity_id"].to_numpy(),
            test["anchor"],
            max_calls=kumo_max_calls,
        )
    except kumo.KumoSkip as exc:
        return {"status": "skipped", "reason": exc.reason}
    except Exception as exc:  # noqa: BLE001 — the hosted call must not sink the local scores
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    scores = raw["scores"]
    y = test["label"].to_numpy(dtype=np.float64)
    mask = np.isfinite(scores)
    block = _model_block(spec["kind"], spec["metric"], y[mask], scores[mask], n_labels=None)
    block["status"] = "ok"
    block["calls"] = raw["calls"]
    block["query"] = raw["query"]
    block["n_scored"] = raw["n_scored"]
    block["n_requested"] = raw["n_requested"]
    block["columns"] = raw["columns"]
    return block


def _model_block(kind: str, metric: str, y: np.ndarray, pred: np.ndarray, n_labels: int | None) -> dict[str, Any]:
    value = score_task(kind, metric, y, pred)
    block: dict[str, Any] = {"status": "ok", metric: None if np.isnan(value) else round(float(value), 4)}
    if n_labels is not None:
        block["n_labels"] = int(n_labels)
    return block


def _positive_rate(y: np.ndarray, kind: str) -> float | None:
    if kind != "binary" or len(y) == 0:
        return None
    return round(float(np.mean(y == 1)), 4)


def _story(index: Indexed, task: str, test: pd.DataFrame, flat: np.ndarray, rfm: np.ndarray) -> dict[str, Any]:
    """The row where the two models disagree most, among true positives."""
    y = test["label"].to_numpy(dtype=np.float64)
    gap = rfm - flat
    candidates = np.flatnonzero(y == 1)
    if len(candidates) == 0:
        return {"note": "no positive labels in the test split"}
    pick = candidates[int(np.argmax(gap[candidates]))]
    entity = int(test["entity_id"].iloc[pick])
    anchor = pd.Timestamp(test["anchor"].iloc[pick])
    body: dict[str, Any] = {
        "entity_id": entity,
        "anchor": anchor.isoformat(),
        "label": 1,
        "flat_score": round(float(flat[pick]), 4),
        "rfm_score": round(float(rfm[pick]), 4),
    }
    if task == "churn":
        snap = customer_snapshot(index, entity, anchor)
        body["snapshot"] = {k: round(v, 3) for k, v in snap.items()}
        body["reading"] = _churn_reading(snap, flat[pick], rfm[pick])
    else:
        body["reading"] = (
            f"payment {entity}: flat P(fraud)={flat[pick]:.2f} from the customer's own history, "
            f"rfm P(fraud)={rfm[pick]:.2f} after looking at other payments on the same device. "
            f"The payment was fraud."
        )
    return body


def _churn_reading(snap: dict[str, float], flat_p: float, rfm_p: float) -> str:
    return (
        f"Last order {snap['recency_days']:.0f} days ago, "
        f"{snap['frequency_365']:.0f} orders and ${snap['monetary_365']:.0f} in the past year. "
        f"Flat, which only sees those three numbers, says P(churn)={flat_p:.2f}. "
        f"There are {snap['tickets_21']:.0f} severe account tickets in the last 21 days "
        f"(severity sum {snap['severity_21']:.0f}) and a {snap['refund_rate_60']:.0%} refund rate "
        f"over 60 days. The relational model says P(churn)={rfm_p:.2f}. "
        f"They placed no order in the next 90 days."
    )


def evaluate_relbench(
    root,
    tasks: list[str],
    models: list[str],
    *,
    split: str = "test",
    context_size: int = 64,
    k: int = 31,
    seed: int = 0,
    kumo_max_calls: int = 8,
) -> dict[str, Any]:
    from .relbench import driver_matrix, examples_from_task, load_tables, load_task, task_kind

    tables = load_tables(root)
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "source": "rel-f1",
        "path": str(root),
        "license": "CC BY-SA 4.0",
        "context_size": context_size,
        "models": {name: MODEL_NOTES[name] for name in models},
        "tasks": {},
    }
    for task in tasks:
        manifest, frame = load_task(root, task, split=split)
        kind, metric = task_kind(manifest)
        labeled = examples_from_task(manifest, frame)
        # Fit and retrieve on the train split. `labeled` is the split being scored.
        train_manifest, train_frame = load_task(root, task, split="train")
        train = examples_from_task(train_manifest, train_frame)
        y_train = train["label"].to_numpy(dtype=np.float64)
        y_test = labeled["label"].to_numpy(dtype=np.float64)
        block: dict[str, Any] = {
            "question": str(manifest.get("description", "")).strip(),
            "metric": metric,
            "kind": kind,
            "scored_split": manifest.get("scored_split"),
            "n_train": int(len(train)),
            "n_test": int(len(labeled)),
            "positive_rate": _positive_rate(y_test, kind),
            "models": {},
        }
        if "flat" in models:
            pred = supervised_predict(
                driver_matrix(tables, train["entity_id"].to_numpy(), train["anchor"].to_numpy(), mode="flat"),
                y_train,
                driver_matrix(tables, labeled["entity_id"].to_numpy(), labeled["anchor"].to_numpy(), mode="flat"),
                kind=kind,
            )
            block["models"]["flat"] = _model_block(kind, metric, y_test, pred, len(train))
        if "rfm" in models:
            Xtr = driver_matrix(tables, train["entity_id"].to_numpy(), train["anchor"].to_numpy(), mode="rfm")
            Xte = driver_matrix(tables, labeled["entity_id"].to_numpy(), labeled["anchor"].to_numpy(), mode="rfm")
            Xc, yc = subsample_context(Xtr, y_train, context_size, rng)
            pred = context_scores(Xc, yc, Xte, k=k)
            if kind == "regression":
                pred = np.clip(pred, 0.0, None)
            block["models"]["rfm"] = _model_block(kind, metric, y_test, pred, len(yc))
        if "kumo" in models:
            block["models"]["kumo"] = _run_kumo(tables, task, labeled, {"kind": kind, "metric": metric}, kumo_max_calls)
        report["tasks"][task] = block
    return report


# Re-export so callers can build examples without importing tasks.
__all__ = ["evaluate_relbench", "evaluate_warehouse", "Examples", "build_examples", "FLAT_CUSTOMER"]
