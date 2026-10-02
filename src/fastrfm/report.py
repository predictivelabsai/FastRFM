"""Text and JSON renderings of an evaluation report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.floating):
        return to_jsonable(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(report), indent=2))


def format_report(report: dict) -> str:
    lines: list[str] = []
    source = report.get("source", "")
    lines.append(f"FastRFM  source={source}  context_size={report.get('context_size')}")
    if report.get("anchors"):
        lines.append("anchors: " + ", ".join(f"{k}={v}" for k, v in report["anchors"].items()))
    if report.get("license"):
        lines.append(f"license: {report['license']}  path={report.get('path')}")
    lines.append("")
    lines.append(f"{'task':<18} {'metric':<8} {'n':>6} {'flat':>10} {'rfm':>10} {'kumo':>10}")
    lines.append("-" * 66)
    for name, task in report.get("tasks", {}).items():
        metric = task.get("metric", "")
        n = task.get("n_test", 0)
        cells = []
        for model in ("flat", "rfm", "kumo"):
            cells.append(_cell(task.get("models", {}).get(model), metric))
        lines.append(f"{name:<18} {metric:<8} {n:>6} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")
        note = _task_note(task)
        if note:
            lines.append(f"  {note}")
    lines.append("")
    for name, task in report.get("tasks", {}).items():
        story = task.get("story")
        if not story or "reading" not in story:
            continue
        lines.append(f"{name}  entity {story['entity_id']}  anchor {story['anchor']}")
        lines.append(f"  {story['reading']}")
        lines.append("")
    skipped = _skipped_reasons(report)
    if skipped:
        lines.append(skipped)
    if report.get("source") == "rel-f1":
        flat_line = "flat uses only that driver's own results."
    else:
        flat_line = "flat fits a linear model on recency, frequency, and monetary value."
    lines.append(
        f"{flat_line} "
        "rfm retrieves labeled neighborhoods and does not fit task weights. "
        "kumo is hosted NVIDIA Kumo Relational (ex-KumoRFM) and runs only when KUMO_API_KEY is set."
    )
    return "\n".join(lines).rstrip() + "\n"


def _cell(block: dict | None, metric: str) -> str:
    if block is None:
        return "—"
    status = block.get("status")
    if status == "skipped":
        return "needs key"
    if status == "error":
        return "error"
    value = block.get(metric)
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _task_note(task: dict) -> str:
    bits = []
    if task.get("scored_split"):
        bits.append(f"scored {task['scored_split']}")
    rate = task.get("positive_rate")
    if rate is not None:
        bits.append(f"positive rate {rate:.2f}")
    kumo = task.get("models", {}).get("kumo") or {}
    if kumo.get("status") == "ok" and kumo.get("n_scored") not in (None, task.get("n_test")):
        bits.append(f"kumo scored {kumo['n_scored']}/{kumo.get('n_requested')}")
    if kumo.get("status") == "error":
        bits.append(f"kumo: {kumo.get('reason')}")
    return " · ".join(bits)


def _skipped_reasons(report: dict) -> str:
    reasons = []
    for task in report.get("tasks", {}).values():
        kumo = task.get("models", {}).get("kumo") or {}
        if kumo.get("status") == "skipped" and kumo.get("reason"):
            reasons.append(kumo["reason"])
            break
    return reasons[0] if reasons else ""
