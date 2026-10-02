"""FastRFM command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .evaluate import evaluate_relbench, evaluate_warehouse
from .kumo import SIGNUP_URL, key_status
from .relbench import TASK_NAMES, fetch
from .report import format_report, write_json
from .tasks import TASKS
from .warehouse import generate

SYNTHETIC_TASKS = tuple(TASKS)
DEFAULT_DATA = Path("data")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastrfm",
        description=(
            "Evaluate relational prediction. flat is a recency/frequency/monetary "
            "baseline. rfm is a local in-context model over the foreign-key graph. "
            "kumo calls hosted NVIDIA Kumo Relational (ex-KumoRFM) when KUMO_API_KEY is set."
        ),
    )
    parser.add_argument("--version", action="version", version=f"fastrfm {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("keys", help="Say which models need an API key.")
    sub.add_parser("sources", help="List datasets and the tasks defined on each.")

    gen = sub.add_parser("generate", help="Write a synthetic warehouse to parquet.")
    gen.add_argument("-o", "--output", type=Path, default=DEFAULT_DATA / "warehouse")
    gen.add_argument("--customers", type=int, default=400)
    gen.add_argument("--seed", type=int, default=0)

    fet = sub.add_parser("fetch", help="Download an open relational dataset.")
    fet.add_argument("name", choices=("rel-f1",))
    fet.add_argument("-o", "--output", type=Path, default=DEFAULT_DATA / "rel-f1")

    ev = sub.add_parser("eval", help="Score flat, rfm, and optionally kumo on one source.")
    _add_eval_args(ev)

    ex = sub.add_parser("explain", help="Show one customer the two local models disagree on.")
    _add_eval_args(ex)
    ex.add_argument("--task", default="churn", help="Task to narrate (default: churn).")
    return parser


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        default="synthetic",
        help="synthetic, rel-f1, or a path written by `fastrfm generate`.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names. Default: every task for the source.",
    )
    parser.add_argument(
        "--models",
        default="flat,rfm,kumo",
        help="Comma-separated: flat, rfm, kumo. kumo is skipped cleanly without a key.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--customers", type=int, default=400, help="Synthetic warehouse size.")
    parser.add_argument("--context-size", type=int, default=64, help="Labeled rows the rfm model may retrieve.")
    parser.add_argument("--k", type=int, default=31, help="Neighbors averaged by the rfm model.")
    parser.add_argument("--kumo-max-calls", type=int, default=8, help="Cap on hosted queries for one task.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write the report JSON here.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="RelBench split to score (default: test). Ignored for the synthetic warehouse.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA / "rel-f1",
        help="Where rel-f1 was fetched (default: data/rel-f1).",
    )


def main(argv: list[str] | None = None) -> int:
    from .nvidia import load_env

    load_env()
    args = _build_parser().parse_args(argv)
    if args.cmd == "keys":
        print(json.dumps(key_status(), indent=2))
        print(f"\nNVIDIA API key for Kumo Relational: {SIGNUP_URL}")
        print("Open data and the local models need no key.")
        return 0
    if args.cmd == "sources":
        print(_sources_text())
        return 0
    if args.cmd == "generate":
        warehouse = generate(n_customers=args.customers, seed=args.seed)
        warehouse.save(args.output)
        counts = {name: len(frame) for name, frame in warehouse.tables.items()}
        print(f"wrote {args.output}")
        for name, n in counts.items():
            print(f"  {name:<12} {n}")
        return 0
    if args.cmd == "fetch":
        path = fetch(args.output)
        print(f"rel-f1 is in {path}  (CC BY-SA 4.0, Stanford STAR / Hugging Face)")
        return 0
    if args.cmd in ("eval", "explain"):
        return _eval(args)
    return 1


def _eval(args: argparse.Namespace) -> int:
    models = _split_csv(args.models)
    unknown = [m for m in models if m not in ("flat", "rfm", "kumo")]
    if unknown:
        print(f"unknown models: {', '.join(unknown)}", file=sys.stderr)
        return 2
    source = args.source
    if source == "synthetic":
        tasks = _split_csv(args.tasks) or list(SYNTHETIC_TASKS)
        bad = [t for t in tasks if t not in TASKS]
        if bad:
            print(f"unknown synthetic tasks: {', '.join(bad)}", file=sys.stderr)
            return 2
        warehouse = generate(n_customers=args.customers, seed=args.seed)
        report = evaluate_warehouse(
            warehouse,
            tasks,
            models,
            context_size=args.context_size,
            k=args.k,
            seed=args.seed,
            kumo_max_calls=args.kumo_max_calls,
        )
    elif source == "rel-f1" or (source != "synthetic" and _is_relbench(source)):
        root = args.data_dir if source == "rel-f1" else Path(source)
        if not (root / "db" / "results.parquet").exists():
            if source == "rel-f1":
                print(f"fetching rel-f1 into {root}")
                fetch(root)
            else:
                print(f"no rel-f1 database at {root}", file=sys.stderr)
                return 2
        tasks = _split_csv(args.tasks) or list(TASK_NAMES)
        bad = [t for t in tasks if t not in TASK_NAMES]
        if bad:
            print(f"unknown rel-f1 tasks: {', '.join(bad)}", file=sys.stderr)
            return 2
        report = evaluate_relbench(
            root,
            tasks,
            models,
            split=args.split,
            context_size=args.context_size,
            k=args.k,
            seed=args.seed,
            kumo_max_calls=args.kumo_max_calls,
        )
    else:
        path = Path(source)
        if not (path / "source.json").exists():
            print(
                f"unknown source {source!r}. Use synthetic, rel-f1, or a warehouse directory.",
                file=sys.stderr,
            )
            return 2
        from .warehouse import Warehouse

        warehouse = Warehouse.load(path)
        tasks = _split_csv(args.tasks) or list(SYNTHETIC_TASKS)
        report = evaluate_warehouse(
            warehouse,
            tasks,
            models,
            context_size=args.context_size,
            k=args.k,
            seed=args.seed,
            kumo_max_calls=args.kumo_max_calls,
        )
        report["source"] = str(path)

    if args.cmd == "explain":
        story = report["tasks"].get(args.task, {}).get("story")
        if not story:
            print(f"no story for {args.task}. It is produced for synthetic churn and fraud.", file=sys.stderr)
            return 2
        print(story["reading"])
        if story.get("snapshot"):
            for key, value in story["snapshot"].items():
                print(f"  {key:<22} {value}")
    else:
        print(format_report(report), end="")
    if args.output:
        write_json(report, args.output)
        print(f"wrote {args.output}")
    return 0


def _is_relbench(source: str) -> bool:
    path = Path(source)
    return (path / "db" / "results.parquet").exists() and (path / "tasks").is_dir()


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _sources_text() -> str:
    return "\n".join(
        [
            "synthetic   no key",
            "  customers, orders, payments, tickets, restaurants, devices",
            "  tasks: " + ", ".join(SYNTHETIC_TASKS),
            "  the order clock is regular; churn and fraud are planted in the joins",
            "",
            "rel-f1      no key    CC BY-SA 4.0",
            "  Stanford RelBench Formula 1 database, fetched from the Hugging Face Hub",
            "  tasks: " + ", ".join(TASK_NAMES),
            "  the Hub parquet includes test labels; those are what `eval` scores",
            "",
            "kumo model  KUMO_API_KEY",
            f"  hosted NVIDIA Kumo Relational (ex-KumoRFM), NVIDIA API key from {SIGNUP_URL}",
            "  weights are not downloaded",
            "",
            "OpenRFM (arXiv:2606.04320) and RDB-PFN (github.com/MuLabPKU/RDBPFN)",
            "publish training recipes, not a pip inference model. This CLI does not call them.",
            "",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
