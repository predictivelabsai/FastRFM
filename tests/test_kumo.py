"""Hosted KumoRFM is optional. Parsing its tables does not require a key."""

import numpy as np
import pandas as pd
import pytest

from fastrfm import kumo
from fastrfm.evaluate import evaluate_warehouse
from fastrfm.warehouse import generate


def test_missing_key_skips(monkeypatch):
    monkeypatch.delenv("KUMO_API_KEY", raising=False)
    warehouse = generate(n_customers=30, seed=0)
    report = evaluate_warehouse(warehouse, ["churn"], ["flat", "kumo"], context_size=16, seed=0)
    skipped = report["tasks"]["churn"]["models"]["kumo"]
    assert skipped["status"] == "skipped"
    assert "KUMO_API_KEY" in skipped["reason"]
    assert report["tasks"]["churn"]["models"]["flat"]["status"] == "ok"


def test_scores_align_to_entity_ids():
    frame = pd.DataFrame(
        {
            "ENTITY": [3, 1, 3],
            "TARGET_PRED": [0.2, 0.9, 0.4],
            "note": ["a", "b", "c"],
        }
    )
    scores, entity_col, score_col = kumo.scores_from_frame(frame, np.array([1, 2, 3]))
    assert entity_col == "ENTITY"
    assert score_col == "TARGET_PRED"
    assert scores[0] == pytest.approx(0.9)
    assert np.isnan(scores[1])
    assert scores[2] == pytest.approx(0.3)  # mean of 0.2 and 0.4


def test_ranking_cell_parses_a_list():
    frame = pd.DataFrame({"customer_id": [4], "PREDICTION": ["[12, 15]"]})
    scores, _, _ = kumo.scores_from_frame(frame, np.array([4]))
    assert scores[0] == pytest.approx(12)


def test_truncate_drops_the_future():
    warehouse = generate(n_customers=20, seed=0)
    anchor = warehouse.anchors["test"]
    trimmed = kumo.truncate(warehouse.tables, anchor)
    assert (trimmed["orders"]["ts"] < anchor).all()
    assert len(trimmed["orders"]) < len(warehouse.tables["orders"])
    assert len(trimmed["restaurants"]) == len(warehouse.tables["restaurants"])


def test_predict_task_uses_the_injected_client(monkeypatch):
    monkeypatch.delenv("KUMO_API_KEY", raising=False)

    def fake_run(tables, query):
        assert "COUNT(orders.*" in query
        assert (tables["orders"]["ts"] < tables["orders"]["ts"].max() + pd.Timedelta(days=1)).all()
        ids = tables["customers"]["customer_id"].tolist()
        return pd.DataFrame({"customer_id": ids, "True_PROB": [0.25] * len(ids)})

    monkeypatch.setattr(kumo, "run_query", fake_run)
    warehouse = generate(n_customers=25, seed=0)
    examples = __import__("fastrfm.tasks", fromlist=["build_examples"]).build_examples(warehouse, "churn").split("test")
    raw = kumo.predict_task(
        warehouse.tables,
        "churn",
        examples["entity_id"].to_numpy(),
        examples["anchor"],
        max_calls=4,
    )
    assert raw["calls"] == 1
    assert raw["n_scored"] == len(examples)
    assert np.allclose(raw["scores"], 0.25)
