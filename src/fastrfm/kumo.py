"""Hosted KumoRFM / NVIDIA Kumo Relational. Nothing here runs without a key.

KumoRFM weights are not on Hugging Face. Since 2026 the hosted model is NVIDIA
Kumo Relational on the NVIDIA API Catalog. With an ``nvapi-`` key in
``KUMO_API_KEY`` the call goes through :mod:`fastrfm.nvidia` (NVIDIA's
``kumo-relational-engine`` builds the request, Bearer auth to the catalog).
Any other key falls back to the legacy ``kumoai`` SDK (``rfm.init()``).

The graph passed to Kumo is truncated to rows before the anchor, so the
service's default anchor (the latest timestamp it was given) is our cutoff.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd

from .warehouse import as_day

SIGNUP_URL = "https://build.nvidia.com/nvidia/kumo-relational"
KEY_ENV = "KUMO_API_KEY"

# Time columns used to enforce the cutoff. Tables absent from this map are
# static and are sent whole.
TIME_COLUMNS = {
    "customers": "signup_date",
    "orders": "ts",
    "payments": "ts",
    "tickets": "ts",
    "results": "date",
    "qualifying": "date",
    "standings": "date",
    "constructor_standings": "date",
    "constructor_results": "date",
    "races": "date",
}


class KumoSkip(Exception):
    """The hosted model was not run. ``reason`` is safe to print."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _backend() -> str:
    from . import nvidia

    key = nvidia.api_key()
    if not key:
        return "none"
    return "nvidia" if nvidia.is_nvidia_key(key) else "kumoai"


def key_status() -> dict[str, str]:
    backend = _backend()
    return {
        "env": KEY_ENV,
        "state": "missing" if backend == "none" else "set",
        "backend": {"nvidia": "NVIDIA API Catalog (kumo-relational, Bearer)",
                    "kumoai": "legacy kumoai SDK (kumorfm.ai)", "none": "-"}[backend],
        "required_for": "the kumo model only",
        "signup": SIGNUP_URL,
        "weights": "hosted by NVIDIA, not downloaded",
    }


def _load_sdk():
    from . import nvidia

    nvidia.load_env()
    if not os.environ.get(KEY_ENV):
        raise KumoSkip(
            f"{KEY_ENV} is not set. A free key is issued at {SIGNUP_URL} "
            "(1000 queries/day). The flat and rfm models do not need it."
        )
    try:
        import kumoai.experimental.rfm as rfm
        return rfm
    except ImportError:
        pass
    try:
        from kumoai import rfm
        return rfm
    except ImportError as exc:
        raise KumoSkip(
            "kumoai is not installed. Run `pip install kumoai` "
            "(or `pip install 'fastrfm[kumo]'`) and export KUMO_API_KEY."
        ) from exc


def truncate(tables: dict[str, pd.DataFrame], anchor: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Drop rows at or after ``anchor``. Static tables are copied through."""
    out: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        column = TIME_COLUMNS.get(name)
        if column is None or column not in frame.columns:
            out[name] = frame.copy()
            continue
        kept = frame.loc[pd.to_datetime(frame[column]) < anchor].copy()
        out[name] = kept
    return out


def pql(task: str, entity_ids: list[int] | None = None) -> str:
    """Predictive query for one task. ``entity_ids`` narrows a ``FOR ... IN`` list."""
    ids = ""
    if entity_ids is not None:
        ids = ", ".join(str(int(i)) for i in entity_ids)
    queries = {
        "churn": "PREDICT COUNT(orders.*, 0, 90, days)=0 FOR EACH customers.customer_id",
        "ltv": "PREDICT SUM(orders.net_amount, 0, 180, days) FOR EACH customers.customer_id",
        "notify": (
            "PREDICT LIST_DISTINCT(orders.restaurant_id, 0, 30, days) "
            "RANK TOP 1 FOR EACH customers.customer_id"
        ),
        "fraud": f"PREDICT payments.is_fraud FOR payments.payment_id IN ({ids})",
        "driver-dnf": (
            f"PREDICT MAX(results.statusId, 0, 30, days)>1 FOR drivers.driverId IN ({ids})"
        ),
        "driver-top3": (
            f"PREDICT MIN(qualifying.position, 0, 30, days)<=3 "
            f"FOR drivers.driverId IN ({ids})"
        ),
        "driver-position": (
            f"PREDICT AVG(results.positionOrder, 0, 60, days) FOR drivers.driverId IN ({ids})"
        ),
    }
    if task not in queries:
        raise KumoSkip(f"no Predictive Query is defined for task {task!r}")
    if entity_ids is not None and not entity_ids:
        raise KumoSkip("refusing to send an empty entity list")
    return queries[task]


def run_query(
    tables: dict[str, pd.DataFrame],
    query: str,
    indices: list[int] | None = None,
    anchor_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Execute one PQL query. Tests monkeypatch this.

    ``indices`` and ``anchor_time`` are used by the NVIDIA backend, which needs
    explicit entity ids for ``FOR EACH`` queries.
    """
    if _backend() == "nvidia":
        from . import nvidia

        kwargs: dict[str, Any] = {}
        if indices is not None:
            kwargs["indices"] = [int(i) for i in indices]
        if anchor_time is not None:
            kwargs["anchor_time"] = pd.Timestamp(anchor_time)
        try:
            frame = nvidia.KumoRelational(tables).predict(query, **kwargs).frame
        except ImportError as exc:
            raise KumoSkip("kumo-relational-client[relational] is not installed") from exc
        # Normalize CLASS/SCORE outputs to one number per entity.
        collapsed = nvidia.positive_scores(frame)
        if collapsed is None:
            collapsed = nvidia.top_class(frame)
            if collapsed is not None:
                collapsed = collapsed[["ENTITY", "CLASS"]]
        return frame if collapsed is None else collapsed
    rfm = _load_sdk()
    rfm.init()
    if hasattr(rfm, "LocalGraph"):
        graph = rfm.LocalGraph.from_data(tables)
    else:
        graph = rfm.Graph.from_data(tables)
    model = rfm.KumoRFM(graph)
    result = model.predict(query)
    if isinstance(result, pd.DataFrame):
        return result
    for attr in ("df", "to_pandas", "frame"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            frame = value() if callable(value) else value
            if isinstance(frame, pd.DataFrame):
                return frame
    raise KumoSkip(f"Kumo returned {type(result).__name__}, not a table of predictions")


def _call(graph, query, ids, anchor):
    """``run_query`` with ids/anchor when it accepts them (tests patch a 2-arg version)."""
    import inspect

    try:
        params = inspect.signature(run_query).parameters
    except (TypeError, ValueError):
        params = {}
    if "indices" in params:
        return run_query(graph, query, ids, anchor)
    return run_query(graph, query)


_SCORE_HINTS = (
    "class",  # top-1 class from a ranking, after nvidia.top_class
    "true_prob",
    "prob_true",
    "probability",
    "target_pred",
    "prediction",
    "score",
    "prob",
)


def scores_from_frame(
    frame: pd.DataFrame, entity_ids: np.ndarray
) -> tuple[np.ndarray, str, str]:
    """Align a Kumo result frame to ``entity_ids``.

    Returns ``(scores, entity_column, score_column)``. Scores are NaN where
    the entity is missing from the frame.
    """
    if frame.empty:
        raise KumoSkip("Kumo returned no rows")
    idset = {int(i) for i in entity_ids.tolist()}
    entity_col = _entity_column(frame, idset)
    score_col = _score_column(frame, entity_col)
    entities = pd.to_numeric(frame[entity_col], errors="coerce")
    scores = pd.to_numeric(frame[score_col], errors="coerce")
    # Ranking cells are often strings ("[12]") rather than floats.
    if scores.notna().mean() < 0.5:
        parsed = frame[score_col].map(_first_number)
        scores = pd.to_numeric(parsed, errors="coerce")
    bucket: dict[int, list[float]] = {}
    for eid, score in zip(entities.tolist(), scores.tolist(), strict=True):
        if eid is None or (isinstance(eid, float) and np.isnan(eid)):
            continue
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        bucket.setdefault(int(eid), []).append(float(score))
    out = np.full(len(entity_ids), np.nan)
    for i, eid in enumerate(entity_ids.tolist()):
        values = bucket.get(int(eid))
        if values:
            out[i] = float(np.mean(values))
    if np.isfinite(out).sum() == 0:
        raise KumoSkip(
            "could not match Kumo rows to the requested entities; "
            f"columns were {list(frame.columns)}"
        )
    return out, entity_col, score_col


def _entity_column(frame: pd.DataFrame, idset: set[int]) -> str:
    best: str | None = None
    best_n = -1
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().mean() < 0.5:
            continue
        n = int(values.dropna().astype(np.int64).isin(idset).sum())
        if n > best_n:
            best, best_n = col, n
    if best is None or best_n <= 0:
        raise KumoSkip(f"no entity column in Kumo result; columns were {list(frame.columns)}")
    return best


def _score_column(frame: pd.DataFrame, entity_col: str) -> str:
    lowered = {col: str(col).lower().replace(" ", "_") for col in frame.columns}
    for hint in _SCORE_HINTS:
        for col, name in lowered.items():
            if col == entity_col:
                continue
            if hint in name and "false" not in name:
                return col
    # Binary outputs sometimes use a column ending in _PROB for the positive class.
    for col, name in lowered.items():
        if col != entity_col and name.endswith("_prob") and "false" not in name and not name.startswith("0"):
            return col
    numeric = []
    for col in frame.columns:
        if col == entity_col:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            numeric.append(col)
    if len(numeric) == 1:
        return numeric[0]
    if numeric:
        return numeric[-1]
    # Last resort: a non-entity column that contains a number somewhere.
    for col in frame.columns:
        if col != entity_col:
            return col
    raise KumoSkip(f"no score column in Kumo result; columns were {list(frame.columns)}")


def _first_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return _first_number(value[0])
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def mask_fraud_labels(payments: pd.DataFrame, target_ids: list[int]) -> pd.DataFrame:
    """Hide the labels being scored. Earlier fraud labels stay, so the device history is usable."""
    frame = payments.copy()
    frame["is_fraud"] = frame["is_fraud"].astype(np.float64)
    hide = frame["payment_id"].isin(target_ids)
    frame.loc[hide, "is_fraud"] = np.nan
    return frame


def predict_task(
    tables: dict[str, pd.DataFrame],
    task: str,
    entity_ids: np.ndarray,
    anchors: pd.Series,
    *,
    max_calls: int = 8,
) -> dict[str, Any]:
    """Score ``entity_ids``. Customer tasks share one anchor; fraud is batched by day.

    Returns a dict with ``scores`` aligned to ``entity_ids`` and a ``calls`` count.
    Raises ``KumoSkip`` when the model cannot be called.
    """
    if task == "fraud":
        return _predict_fraud(tables, entity_ids, anchors, max_calls=max_calls)
    if len(entity_ids) == 0:
        raise KumoSkip("no entities to score")
    # One cutoff: the anchor these rows share. Mixed anchors are batched.
    days = np.array([as_day(t) for t in anchors.tolist()], dtype=np.int32)
    unique_days = list(dict.fromkeys(days.tolist()))
    if len(unique_days) > max_calls:
        # Keep the latest cutoffs. The caller reports how many rows that covers.
        unique_days = unique_days[-max_calls:]
    scores = np.full(len(entity_ids), np.nan)
    calls = 0
    columns: list[str] = []
    query_text = ""
    for day in unique_days:
        member = np.flatnonzero(days == day)
        anchor = anchors.iloc[int(member[0])]
        graph = truncate(tables, pd.Timestamp(anchor))
        ids = [int(i) for i in entity_ids[member].tolist()]
        # FOR EACH queries ignore the id list; IN queries need it.
        query = pql(task, ids if task not in ("churn", "ltv", "notify") else None)
        query_text = query
        frame = _call(graph, query, ids, pd.Timestamp(anchor))
        calls += 1
        part, entity_col, score_col = scores_from_frame(frame, entity_ids[member])
        scores[member] = part
        columns = [entity_col, score_col]
    covered = int(np.isfinite(scores).sum())
    if covered == 0:
        raise KumoSkip("Kumo returned no scores on the selected anchors")
    return {
        "scores": scores,
        "calls": calls,
        "query": query_text,
        "columns": columns,
        "n_scored": covered,
        "n_requested": int(len(entity_ids)),
    }


def _predict_fraud(
    tables: dict[str, pd.DataFrame],
    entity_ids: np.ndarray,
    anchors: pd.Series,
    *,
    max_calls: int,
) -> dict[str, Any]:
    days = np.array([as_day(t) for t in anchors.tolist()], dtype=np.int32)
    unique_days = sorted(set(days.tolist()))
    if len(unique_days) > max_calls:
        # Evenly spaced days so the sample is not just the first week.
        pick = np.linspace(0, len(unique_days) - 1, max_calls).round().astype(int)
        chosen = {unique_days[i] for i in pick.tolist()}
        unique_days = [d for d in unique_days if d in chosen]
    scores = np.full(len(entity_ids), np.nan)
    calls = 0
    columns: list[str] = []
    query_text = ""
    for day in unique_days:
        member = np.flatnonzero(days == day)
        ids = [int(i) for i in entity_ids[member].tolist()]
        # Include payments up through this day, and blank the labels of the
        # payments being scored. Later days are not in the graph.
        anchor = pd.Timestamp(anchors.iloc[int(member[0])]) + pd.Timedelta(days=1)
        graph = truncate(tables, anchor)
        graph["payments"] = mask_fraud_labels(graph["payments"], ids)
        if _backend() == "nvidia":
            # Nullable bool, so the hosted model sees a binary target, not 0.0/1.0 classes.
            graph["payments"]["is_fraud"] = graph["payments"]["is_fraud"].astype("boolean")
        query = pql("fraud", ids)
        query_text = query
        frame = _call(graph, query, ids, None)
        calls += 1
        part, entity_col, score_col = scores_from_frame(frame, entity_ids[member])
        scores[member] = part
        columns = [entity_col, score_col]
    return {
        "scores": scores,
        "calls": calls,
        "query": query_text,
        "columns": columns,
        "n_scored": int(np.isfinite(scores).sum()),
        "n_requested": int(len(entity_ids)),
    }
