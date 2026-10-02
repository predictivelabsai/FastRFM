"""RelBench Formula 1, the smallest public database in the benchmark.

``rel-f1`` is about a megabyte, CC BY-SA 4.0, hosted by the Stanford STAR
project on the Hugging Face Hub. No account and no token. The Hub parquet
includes test labels, so the default is to score the test split. If a target
column is empty, the loader falls back to validation and says so.

Driver tasks (from the dataset manifest):

- ``driver-dnf`` — will the driver fail to finish a race in the next 30 days? (AUROC)
- ``driver-top3`` — will they qualify in the top 3 in the next 30 days? (AUROC)
- ``driver-position`` — mean finishing position over the next 60 days (MAE, lower is better)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .features import _Grouped, _days

HUB = "https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/main/rel-f1"
TASK_NAMES = ("driver-dnf", "driver-top3", "driver-position")
FILES = (
    "manifest.yaml",
    "db/circuits.parquet",
    "db/constructors.parquet",
    "db/constructor_results.parquet",
    "db/constructor_standings.parquet",
    "db/drivers.parquet",
    "db/qualifying.parquet",
    "db/races.parquet",
    "db/results.parquet",
    "db/standings.parquet",
    *(
        f"tasks/{task}/{name}"
        for task in TASK_NAMES
        for name in ("manifest.yaml", "train.parquet", "val.parquet", "test.parquet")
    ),
)

# yaml is not a hard dependency if the user never fetches. The manifests are
# tiny; parse them with a minimal reader when PyYAML is absent.
def _read_manifest(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml as _yaml
    except ImportError:
        return _tiny_yaml(text)
    loaded = _yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a mapping")
    return loaded


def _tiny_yaml(text: str) -> dict:
    """Enough of the RelBench task manifest: top-level scalars and a sql block."""
    out: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith(" ") or line.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if value in ("|-", "|"):
            block: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or lines[i] == ""):
                block.append(lines[i])
                i += 1
            out[key] = "\n".join(block)
            continue
        out[key] = value.strip("'\"")
        i += 1
    return out


def fetch(dest: Path) -> Path:
    """Download rel-f1 parquet and manifests. Skips files that are already present."""
    dest = Path(dest)
    for rel in FILES:
        target = dest / rel
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HUB}/{rel}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fastrfm/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
        except Exception as exc:  # noqa: BLE001 — surface the URL, not a stack
            target.unlink(missing_ok=True)
            raise RuntimeError(f"failed to download {url}: {exc}") from exc
        if target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded empty file from {url}")
    return dest


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    db = root / "db"
    if not (db / "results.parquet").exists():
        raise FileNotFoundError(
            f"no rel-f1 database at {root}. Run `fastrfm fetch rel-f1 -o {root}`."
        )
    tables = {}
    for path in sorted(db.glob("*.parquet")):
        tables[path.stem] = pd.read_parquet(path)
    return tables


def load_task(root: Path, task: str, split: str = "val") -> tuple[dict, pd.DataFrame]:
    """Load a task manifest and a labeled split.

    Prefers ``split``. If that file has no target values, falls back to
    validation and records the fact on the manifest under ``scored_split``.
    """
    if task not in TASK_NAMES:
        raise ValueError(f"unknown rel-f1 task {task!r}; expected one of {', '.join(TASK_NAMES)}")
    manifest = _read_manifest(Path(root) / "tasks" / task / "manifest.yaml")
    frame, used = _labeled_split(Path(root), task, split, str(manifest.get("target_col", "")))
    manifest = dict(manifest)
    manifest["scored_split"] = used
    manifest["requested_split"] = split
    return manifest, frame


def _labeled_split(root: Path, task: str, split: str, target_col: str) -> tuple[pd.DataFrame, str]:
    order = [split] if split != "val" else ["val"]
    if split == "test":
        order = ["test", "val"]
    elif split == "train":
        order = ["train"]
    last_error = ""
    for name in order:
        path = root / "tasks" / task / f"{name}.parquet"
        if not path.exists():
            last_error = f"missing {path}"
            continue
        frame = pd.read_parquet(path)
        if target_col and target_col in frame.columns and frame[target_col].notna().any():
            return frame.loc[frame[target_col].notna()].reset_index(drop=True), name
        last_error = f"{name} has no values in {target_col or '(target)'}"
    raise FileNotFoundError(last_error or f"no labeled split for {task}")


def task_kind(manifest: dict) -> tuple[str, str]:
    task_type = str(manifest.get("task_type", ""))
    if task_type == "binary_classification":
        return "binary", "auroc"
    if task_type == "regression":
        return "regression", "mae"
    raise ValueError(f"unsupported rel-f1 task_type {task_type!r}")


def driver_matrix(
    tables: dict[str, pd.DataFrame],
    entity_ids: np.ndarray,
    anchors: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Driver history before each anchor.

    ``flat`` uses only that driver's own results. ``rfm`` adds the
    constructor's results for other drivers, the driver's standings, and
    qualifying.
    """
    results = tables["results"]
    res_days = _days(results["date"])
    status = pd.to_numeric(results["statusId"], errors="coerce").to_numpy(np.float64)
    dnf = np.where(np.isnan(status), np.nan, (status != 1).astype(np.float64))
    if "positionOrder" in results:
        position = pd.to_numeric(results["positionOrder"], errors="coerce").to_numpy(np.float64)
    else:
        position = status
    points = (
        pd.to_numeric(results["points"], errors="coerce").to_numpy(np.float64)
        if "points" in results
        else np.zeros(len(results))
    )
    by_driver = _Grouped(
        results["driverId"].to_numpy(np.int64),
        res_days,
        {
            "dnf": dnf,
            "position": position,
            "points": points,
            "constructor": results["constructorId"].to_numpy(np.int64),
        },
    )
    by_constructor = _Grouped(
        results["constructorId"].to_numpy(np.int64),
        res_days,
        {
            "dnf": dnf,
            "position": position,
            "driver": results["driverId"].to_numpy(np.int64),
        },
    )
    standings = tables.get("standings")
    by_standing = None
    if standings is not None and "driverId" in standings.columns:
        by_standing = _Grouped(
            standings["driverId"].to_numpy(np.int64),
            _days(standings["date"]),
            {
                "position": pd.to_numeric(standings["position"], errors="coerce").to_numpy(np.float64),
                "points": pd.to_numeric(standings["points"], errors="coerce").to_numpy(np.float64),
            },
        )
    qualifying = tables.get("qualifying")
    by_quali = None
    if qualifying is not None and "driverId" in qualifying.columns:
        by_quali = _Grouped(
            qualifying["driverId"].to_numpy(np.int64),
            _days(qualifying["date"]),
            {"position": pd.to_numeric(qualifying["position"], errors="coerce").to_numpy(np.float64)},
        )

    width = 5 if mode == "flat" else 10
    out = np.zeros((len(entity_ids), width), dtype=np.float64)
    for i, (driver, anchor) in enumerate(zip(entity_ids.tolist(), anchors.tolist(), strict=True)):
        out[i] = _driver_vector(
            by_driver, by_constructor, by_standing, by_quali, int(driver), pd.Timestamp(anchor), mode
        )
    return out


def _anchor_day(anchor: pd.Timestamp) -> int:
    # Same clock as ``features._days``: floors from 2023-01-01. Race dates
    # before that origin are negative, and the day index still sorts.
    delta = pd.Timestamp(anchor) - pd.Timestamp("2023-01-01")
    return int(np.floor(delta / pd.Timedelta(days=1)))


def _driver_vector(by_driver, by_constructor, by_standing, by_quali, driver: int, anchor: pd.Timestamp, mode: str) -> np.ndarray:
    day = _anchor_day(anchor)
    sl = by_driver.before(driver, day)
    days = by_driver.days
    if sl.stop > sl.start:
        recent = days[sl] >= day - 365
        dnf = by_driver.cols["dnf"][sl][recent]
        pos = by_driver.cols["position"][sl][recent]
        pts = by_driver.cols["points"][sl][recent]
        n = float(len(dnf))
        dnf_rate = float(np.nanmean(dnf)) if n else 0.0
        mean_pos = float(np.nanmean(pos)) if n else 0.0
        points = float(np.nansum(pts)) if n else 0.0
        recency = float(day - int(days[sl][recent].max())) if n else 365.0
        constructor = int(by_driver.cols["constructor"][sl][-1])
    else:
        n = dnf_rate = mean_pos = points = 0.0
        recency = 365.0
        constructor = -1
    flat = np.array([n, dnf_rate, mean_pos, recency, points], dtype=np.float64)
    if mode == "flat":
        return flat

    c_dnf = c_pos = c_n = 0.0
    if constructor >= 0:
        csl = by_constructor.before(constructor, day)
        if csl.stop > csl.start:
            own = by_constructor.cols["driver"][csl] != driver
            recent_c = by_constructor.days[csl] >= day - 365
            peer = own & recent_c
            peer_dnf = by_constructor.cols["dnf"][csl][peer]
            peer_pos = by_constructor.cols["position"][csl][peer]
            c_n = float(peer.sum())
            c_dnf = float(np.nanmean(peer_dnf)) if c_n else 0.0
            c_pos = float(np.nanmean(peer_pos)) if c_n else 0.0
    standing_pos = 0.0
    standing_pts = 0.0
    if by_standing is not None:
        ssl = by_standing.before(driver, day)
        if ssl.stop > ssl.start:
            standing_pos = float(by_standing.cols["position"][ssl][-1])
            standing_pts = float(by_standing.cols["points"][ssl][-1])
    quali = 0.0
    if by_quali is not None:
        qsl = by_quali.before(driver, day)
        if qsl.stop > qsl.start:
            recent_q = by_quali.days[qsl] >= day - 365
            if recent_q.any():
                quali = float(np.nanmean(by_quali.cols["position"][qsl][recent_q]))
    extra = np.array([c_n, c_dnf, c_pos, standing_pos, standing_pts], dtype=np.float64)
    # quali replaces nothing; fold it into the constructor position slot's neighbor by appending
    # Keep the width stable at 10: five flat + five relational. Qualifying
    # mean is more informative than raw constructor count, so it takes that slot.
    extra[0] = quali
    return np.concatenate([flat, extra])


def examples_from_task(manifest: dict, frame: pd.DataFrame) -> pd.DataFrame:
    entity_col = str(manifest.get("entity_col", "driverId"))
    target_col = str(manifest.get("target_col", ""))
    time_col = str(manifest.get("time_col", "date"))
    for col in (entity_col, target_col, time_col):
        if col not in frame.columns:
            raise ValueError(
                f"task table is missing {col!r}; columns are {list(frame.columns)}"
            )
    return pd.DataFrame(
        {
            "entity_id": frame[entity_col].to_numpy(np.int64),
            "anchor": pd.to_datetime(frame[time_col]),
            "label": frame[target_col].to_numpy(np.float64),
        }
    )
