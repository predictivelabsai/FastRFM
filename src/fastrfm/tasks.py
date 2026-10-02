"""Label tables. Labels look forward from the anchor; features must not."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .warehouse import Warehouse, as_day

TASKS = {
    "churn": {
        "kind": "binary",
        "metric": "auroc",
        "entity": "customer",
        "horizon_days": 90,
        "question": "Will this customer place no order in the next 90 days?",
    },
    "fraud": {
        "kind": "binary",
        "metric": "auroc",
        "entity": "payment",
        "horizon_days": 0,
        "question": "Is this payment fraud, given the account, the device, and earlier payments?",
    },
    "ltv": {
        "kind": "regression",
        "metric": "mae",
        "entity": "customer",
        "horizon_days": 180,
        "question": "What net spend will this customer generate over the next 180 days?",
    },
    "notify": {
        "kind": "ranking",
        "metric": "hit@1",
        "entity": "customer",
        "horizon_days": 30,
        "question": "Which restaurant does this customer order from next, within 30 days?",
    },
}


@dataclass
class Examples:
    task: str
    frame: pd.DataFrame  # split, entity_id, anchor, label

    def split(self, name: str) -> pd.DataFrame:
        return self.frame.loc[self.frame["split"] == name].reset_index(drop=True)


def _orders_by_customer(orders: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    days = np.array([as_day(t) for t in orders["ts"].tolist()], dtype=np.int32)
    cid = orders["customer_id"].to_numpy(np.int64)
    net = orders["net_amount"].to_numpy(np.float64)
    rid = orders["restaurant_id"].to_numpy(np.int64)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for customer in np.unique(cid):
        mask = cid == customer
        order = np.argsort(days[mask], kind="mergesort")
        out[int(customer)] = (days[mask][order], net[mask][order], rid[mask][order])
    return out


def build_examples(warehouse: Warehouse, task: str) -> Examples:
    if task not in TASKS:
        known = ", ".join(TASKS)
        raise ValueError(f"unknown task {task!r}; synthetic tasks are {known}")
    tables = warehouse.tables
    if task == "fraud":
        return Examples(task, _fraud_examples(tables["payments"]))
    grouped = _orders_by_customer(tables["orders"])
    rows: list[tuple[str, int, pd.Timestamp, float]] = []
    spec = TASKS[task]
    horizon = spec["horizon_days"]
    for split, anchor in warehouse.anchors.items():
        anchor_day = as_day(anchor)
        for cid, (days, net, restaurants) in grouped.items():
            hist = days < anchor_day
            if not hist.any():
                continue
            # Noon on day D is inside (anchor midnight, anchor+horizon midnight]
            # exactly when anchor_day <= D < anchor_day + horizon.
            future = (days >= anchor_day) & (days < anchor_day + horizon)
            if task == "churn":
                label = 0.0 if future.any() else 1.0
            elif task == "ltv":
                label = float(net[future].sum()) if future.any() else 0.0
            elif task == "notify":
                if not future.any():
                    continue
                label = float(restaurants[future][0])
            else:
                raise AssertionError(task)
            rows.append((split, cid, anchor, label))
    frame = pd.DataFrame(rows, columns=["split", "entity_id", "anchor", "label"])
    return Examples(task, frame)


def _fraud_examples(payments: pd.DataFrame) -> pd.DataFrame:
    """Train / val / test by the payment's own day, aligned to the anchors."""
    days = np.array([as_day(t) for t in payments["ts"].tolist()], dtype=np.int32)
    # Days [90, 360) train, [360, 450) val, [540, 600) test.
    # The gap [450, 540) is unused so the test window sits on the same
    # anchor the customer tasks use.
    splits = np.full(len(days), "", dtype=object)
    splits[(days >= 90) & (days < 360)] = "train"
    splits[(days >= 360) & (days < 450)] = "val"
    splits[(days >= 540) & (days < 600)] = "test"
    keep = splits != ""
    # Anchor is the payment time. Features use events on earlier days.
    anchors = payments["ts"].to_numpy()
    frame = pd.DataFrame(
        {
            "split": splits[keep],
            "entity_id": payments["payment_id"].to_numpy(np.int64)[keep],
            "anchor": anchors[keep],
            "label": payments["is_fraud"].to_numpy(np.float64)[keep],
        }
    )
    return frame
