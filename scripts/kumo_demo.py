"""Real predictions from NVIDIA Kumo Relational (ex-KumoRFM) on the FastRFM warehouse.

    python scripts/kumo_demo.py            # writes results/<UTC timestamp>/

Needs KUMO_API_KEY=nvapi-... in .env. Runs, against the hosted NVIDIA API Catalog:

  1. churn     binary classification  PREDICT COUNT(orders.*, 0, 90, days)=0
  2. ltv       regression             PREDICT SUM(orders.net_amount, 0, 180, days)
  3. notify    ranking                PREDICT LIST_DISTINCT(orders.restaurant_id, 0, 30, days) RANK TOP 1
  4. forecast  forecasting            PREDICT SUM(orders.net_amount, 0, 30, days) FORECAST 3 TIMEFRAMES

The graph is cut at the test anchor (2024-06-24), so the model sees no future
rows. Labels come from the rows after the anchor and are only used for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastrfm import nvidia  # noqa: E402
from fastrfm.kumo import truncate  # noqa: E402
from fastrfm.metrics import auroc  # noqa: E402
from fastrfm.tasks import build_examples  # noqa: E402
from fastrfm.warehouse import generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customers", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forecast-customers", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    key = nvidia.api_key()
    if not nvidia.is_nvidia_key(key):
        print(f"KUMO_API_KEY (nvapi-...) not found in env or .env. Get one at {nvidia.SIGNUP_URL}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "results" / stamp
    out.mkdir(parents=True, exist_ok=True)

    wh = generate(n_customers=args.customers, seed=args.seed)
    anchor = wh.anchors["test"]
    graph_tables = truncate(wh.tables, anchor)
    model = nvidia.KumoRelational(graph_tables)
    summary: dict = {
        "endpoint": nvidia.CATALOG_URL + "/predictions",
        "auth": "Authorization: Bearer <NVIDIA API key>",
        "model": "nvidia/kumo-relational",
        "run_utc": stamp,
        "dataset": {
            "source": "fastrfm synthetic warehouse",
            "seed": args.seed,
            "rows": {k: int(len(v)) for k, v in wh.tables.items()},
            "anchor": anchor.isoformat(),
        },
        "tasks": {},
    }

    def labels(task: str) -> pd.DataFrame:
        test = build_examples(wh, task).split("test")
        return test[["entity_id", "label"]]

    # 1. churn
    lab = labels("churn")
    q = "PREDICT COUNT(orders.*, 0, 90, days)=0 FOR EACH customers.customer_id"
    pred = model.predict(q, indices=lab["entity_id"].tolist(), anchor_time=anchor)
    df = lab.merge(pred.frame.rename(columns={"ENTITY": "entity_id"}), on="entity_id")
    df.to_csv(out / "churn_predictions.csv", index=False)
    summary["tasks"]["churn"] = {
        "kind": "binary_classification",
        "query": q,
        "n": int(len(df)),
        "positive_rate": round(float(df["label"].mean()), 4),
        "auroc": round(float(auroc(df["label"].to_numpy(), df["TRUE_PROB"].to_numpy())), 4),
        "mean_p_churn": round(float(df["TRUE_PROB"].mean()), 4),
        "latency_s": round(pred.seconds, 2),
    }
    print("churn", summary["tasks"]["churn"])

    # 2. ltv
    lab = labels("ltv")
    q = "PREDICT SUM(orders.net_amount, 0, 180, days) FOR EACH customers.customer_id"
    pred = model.predict(q, indices=lab["entity_id"].tolist(), anchor_time=anchor)
    df = lab.merge(pred.frame.rename(columns={"ENTITY": "entity_id"}), on="entity_id")
    df.to_csv(out / "ltv_predictions.csv", index=False)
    err = (df["PREDICTION"] - df["label"]).abs()
    summary["tasks"]["ltv"] = {
        "kind": "regression",
        "query": q,
        "n": int(len(df)),
        "mae": round(float(err.mean()), 2),
        "mae_constant_mean_baseline": round(float((df["label"] - df["label"].mean()).abs().mean()), 2),
        "mean_actual": round(float(df["label"].mean()), 2),
        "mean_predicted": round(float(df["PREDICTION"].mean()), 2),
        "latency_s": round(pred.seconds, 2),
    }
    print("ltv", summary["tasks"]["ltv"])

    # 3. notify (ranking)
    lab = labels("notify")
    q = "PREDICT LIST_DISTINCT(orders.restaurant_id, 0, 30, days) RANK TOP 1 FOR EACH customers.customer_id"
    pred = model.predict(q, indices=lab["entity_id"].tolist(), anchor_time=anchor)
    top = nvidia.top_class(pred.frame).rename(columns={"ENTITY": "entity_id"})
    df = lab.merge(top, on="entity_id")
    df.to_csv(out / "notify_predictions.csv", index=False)
    hit = (pd.to_numeric(df["CLASS"]) == df["label"]).mean()
    summary["tasks"]["notify"] = {
        "kind": "ranking (temporal link prediction)",
        "query": q,
        "n": int(len(df)),
        "hit_at_1": round(float(hit), 4),
        "latency_s": round(pred.seconds, 2),
    }
    print("notify", summary["tasks"]["notify"])

    # 4. forecast: three 30-day windows of net spend for the busiest customers.
    orders = wh.tables["orders"]
    hist = orders[orders["ts"] < anchor]
    busiest = hist.groupby("customer_id")["net_amount"].sum().sort_values(ascending=False)
    rows, latencies = [], []
    for cid in busiest.index[: args.forecast_customers].tolist():
        q = f"PREDICT SUM(orders.net_amount, 0, 30, days) FORECAST 3 TIMEFRAMES FOR customers.customer_id={int(cid)}"
        pred = model.predict(q, anchor_time=anchor)
        latencies.append(pred.seconds)
        mine = orders[orders["customer_id"] == cid]
        for _, r in pred.frame.iterrows():
            step = int(r["FORECAST_STEP"])
            lo = anchor + pd.Timedelta(days=30 * (step - 1))
            hi = anchor + pd.Timedelta(days=30 * step)
            actual = float(mine.loc[(mine["ts"] >= lo) & (mine["ts"] < hi), "net_amount"].sum())
            rows.append({"customer_id": int(cid), "step": step, "window_start": lo.date().isoformat(),
                         "predicted": round(float(r["PREDICTION"]), 2), "actual": round(actual, 2)})
    fc = pd.DataFrame(rows)
    fc.to_csv(out / "forecast_predictions.csv", index=False)
    summary["tasks"]["forecast"] = {
        "kind": "forecasting",
        "query": "PREDICT SUM(orders.net_amount, 0, 30, days) FORECAST 3 TIMEFRAMES FOR customers.customer_id=<id>",
        "entities": int(fc["customer_id"].nunique()),
        "mae": round(float((fc["predicted"] - fc["actual"]).abs().mean()), 2),
        "mean_actual": round(float(fc["actual"].mean()), 2),
        "mean_predicted": round(float(fc["predicted"].mean()), 2),
        "latency_s_per_entity": round(float(np.mean(latencies)), 2),
    }
    print("forecast", summary["tasks"]["forecast"])

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(_markdown(summary, fc))
    print(f"wrote {out}")
    return 0


def _markdown(s: dict, fc: pd.DataFrame) -> str:
    t = s["tasks"]
    lines = [
        f"# Kumo Relational run {s['run_utc']}",
        "",
        f"Endpoint `POST {s['endpoint']}`, {s['auth']}. Synthetic FastRFM warehouse (seed {s['dataset']['seed']}), "
        f"graph cut at {s['dataset']['anchor']}.",
        "",
        "| task | kind | n | result | latency |",
        "| --- | --- | ---: | --- | ---: |",
        f"| churn | binary | {t['churn']['n']} | AUROC {t['churn']['auroc']} (positive rate {t['churn']['positive_rate']}) | {t['churn']['latency_s']} s |",
        f"| ltv | regression | {t['ltv']['n']} | MAE {t['ltv']['mae']} vs {t['ltv']['mae_constant_mean_baseline']} for a constant (mean) guess | {t['ltv']['latency_s']} s |",
        f"| notify | ranking | {t['notify']['n']} | hit@1 {t['notify']['hit_at_1']} | {t['notify']['latency_s']} s |",
        f"| forecast | forecasting | {t['forecast']['entities']} customers x 3 | MAE {t['forecast']['mae']} (mean actual {t['forecast']['mean_actual']}) | {t['forecast']['latency_s_per_entity']} s / customer |",
        "",
        "## Forecast detail",
        "",
        fc.to_markdown(index=False) if _has_tabulate() else fc.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
