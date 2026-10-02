"""The local relational model beats a flattened recency/frequency/monetary fit
on the synthetic warehouse, where that is the point of the data."""

from fastrfm.evaluate import evaluate_warehouse
from fastrfm.warehouse import generate


def _scores(seed: int) -> dict:
    warehouse = generate(n_customers=400, seed=seed)
    report = evaluate_warehouse(
        warehouse,
        ["churn", "fraud", "ltv", "notify"],
        ["flat", "rfm"],
        context_size=64,
        seed=seed,
    )
    return report


def test_rfm_beats_flat_when_the_signal_is_a_join():
    report = _scores(0)
    churn = report["tasks"]["churn"]["models"]
    fraud = report["tasks"]["fraud"]["models"]
    ltv = report["tasks"]["ltv"]["models"]
    notify = report["tasks"]["notify"]["models"]

    assert churn["rfm"]["auroc"] > churn["flat"]["auroc"] + 0.15
    assert churn["rfm"]["auroc"] > 0.80
    assert fraud["rfm"]["auroc"] > fraud["flat"]["auroc"] + 0.15
    assert fraud["rfm"]["auroc"] > 0.75
    # Lower MAE is better.
    assert ltv["rfm"]["mae"] < ltv["flat"]["mae"]
    assert notify["rfm"]["hit@1"] > notify["flat"]["hit@1"]


def test_gap_holds_for_a_second_seed():
    report = _scores(1)
    churn = report["tasks"]["churn"]["models"]
    fraud = report["tasks"]["fraud"]["models"]
    assert churn["rfm"]["auroc"] > churn["flat"]["auroc"] + 0.15
    assert fraud["rfm"]["auroc"] > fraud["flat"]["auroc"] + 0.15


def test_rfm_uses_the_capped_context():
    warehouse = generate(n_customers=200, seed=0)
    report = evaluate_warehouse(warehouse, ["churn"], ["rfm"], context_size=64, seed=0)
    assert report["tasks"]["churn"]["models"]["rfm"]["n_labels"] == 64
    assert report["tasks"]["churn"]["n_train"] > 64
