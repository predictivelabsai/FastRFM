"""Synthetic warehouse: the joins hold the signal, and the future stays out of the features."""

import numpy as np
import pandas as pd

from fastrfm.features import Indexed, customer_matrix, payment_matrix
from fastrfm.tasks import build_examples
from fastrfm.warehouse import generate


def test_generate_is_deterministic_and_keyed():
    a = generate(n_customers=40, seed=3)
    b = generate(n_customers=40, seed=3)
    for name in a.tables:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name])
    orders = a.tables["orders"]
    payments = a.tables["payments"]
    assert orders["order_id"].is_unique
    assert payments["payment_id"].is_unique
    assert set(payments["order_id"]) <= set(orders["order_id"])
    assert set(orders["customer_id"]) <= set(a.tables["customers"]["customer_id"])


def test_churn_signal_is_in_the_tickets_not_in_recency():
    warehouse = generate(n_customers=400, seed=0)
    examples = build_examples(warehouse, "churn").split("test")
    index = Indexed.build(warehouse.tables)
    flat = customer_matrix(index, examples["entity_id"].to_numpy(), examples["anchor"].to_numpy(), mode="flat")
    rfm = customer_matrix(index, examples["entity_id"].to_numpy(), examples["anchor"].to_numpy(), mode="rfm")
    y = examples["label"].to_numpy()
    # Column 0 is recency. Relational column 4 is severe tickets in 21 days
    # (flat has 4 columns; tickets_21 is the first extra).
    recency_gap = abs(flat[y == 1, 0].mean() - flat[y == 0, 0].mean())
    ticket_gap = rfm[y == 1, 4].mean() - rfm[y == 0, 4].mean()
    assert recency_gap < 10
    assert ticket_gap > 1.5


def test_features_ignore_events_on_or_after_the_anchor():
    warehouse = generate(n_customers=80, seed=1)
    examples = build_examples(warehouse, "churn").split("test")
    ids = examples["entity_id"].to_numpy()
    anchors = examples["anchor"].to_numpy()
    before = customer_matrix(Indexed.build(warehouse.tables), ids, anchors, mode="rfm")

    mutated = {name: frame.copy() for name, frame in warehouse.tables.items()}
    anchor = pd.Timestamp(examples["anchor"].iloc[0])
    orders = mutated["orders"]
    future = orders["ts"] >= anchor
    orders.loc[future, "amount"] = 1e9
    orders.loc[future, "status"] = "returned"
    tickets = mutated["tickets"]
    extra = tickets.iloc[:3].copy()
    extra["ts"] = anchor + pd.Timedelta(days=2)
    extra["severity"] = 5
    extra["topic"] = "account"
    mutated["tickets"] = pd.concat([tickets, extra], ignore_index=True)
    after = customer_matrix(Indexed.build(mutated), ids, anchors, mode="rfm")
    assert np.allclose(before, after)

    # A ticket before the anchor is visible. The same edit in the past moves features.
    past = tickets.copy()
    past_rows = past.loc[past["ts"] < anchor]
    assert len(past_rows) > 0
    edited = mutated["tickets"].copy()
    # Use the original tickets, not the future ones we appended.
    original = tickets.copy()
    flip = original["ts"] < anchor
    original.loc[flip, "severity"] = 0
    original.loc[flip, "topic"] = "meal"
    mutated["tickets"] = original
    changed = customer_matrix(Indexed.build(mutated), ids, anchors, mode="rfm")
    assert not np.allclose(before, changed)


def test_fraud_label_of_the_payment_itself_is_not_a_feature():
    warehouse = generate(n_customers=100, seed=2)
    examples = build_examples(warehouse, "fraud").split("test")
    # One timestamp, so none of these payments is history for another:
    # features only read earlier days.
    last = examples["anchor"].max()
    ids = examples.loc[examples["anchor"] == last, "entity_id"].to_numpy()
    assert len(ids) > 0
    before = payment_matrix(Indexed.build(warehouse.tables), ids, mode="rfm")

    payments = warehouse.tables["payments"].copy()
    payments["is_fraud"] = 1 - payments["is_fraud"]
    changed = payment_matrix(Indexed.build({**warehouse.tables, "payments": payments}), ids, mode="rfm")
    assert not np.allclose(before, changed)

    only_targets = warehouse.tables["payments"].copy()
    only_targets.loc[only_targets["payment_id"].isin(ids), "is_fraud"] = 1
    held = payment_matrix(Indexed.build({**warehouse.tables, "payments": only_targets}), ids, mode="rfm")
    assert np.allclose(before, held)


def test_labels_read_the_future():
    warehouse = generate(n_customers=60, seed=4)
    examples = build_examples(warehouse, "churn").split("test")
    assert examples["label"].nunique() == 2
    orders = warehouse.tables["orders"].copy()
    anchor = pd.Timestamp(examples["anchor"].iloc[0])
    # Delete every future order: everyone churns.
    orders = orders.loc[orders["ts"] < anchor].copy()
    warehouse.tables["orders"] = orders
    relabeled = build_examples(warehouse, "churn").split("test")
    assert relabeled["label"].eq(1).all()
