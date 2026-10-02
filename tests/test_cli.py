"""The console entry point."""

import json

from fastrfm.cli import main
from fastrfm.warehouse import Warehouse


def test_keys_and_sources(capsys):
    assert main(["keys"]) == 0
    text = capsys.readouterr().out
    assert "KUMO_API_KEY" in text
    assert "kumorfm.ai" in text
    assert main(["sources"]) == 0
    text = capsys.readouterr().out
    assert "rel-f1" in text
    assert "no key" in text


def test_eval_writes_json(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = main(
        [
            "eval",
            "--source",
            "synthetic",
            "--tasks",
            "churn",
            "--models",
            "flat,rfm",
            "--customers",
            "80",
            "--seed",
            "0",
            "--context-size",
            "32",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "churn" in printed
    report = json.loads(out.read_text())
    assert report["tasks"]["churn"]["models"]["rfm"]["status"] == "ok"
    assert "kumo" not in report["tasks"]["churn"]["models"]


def test_generate_roundtrip(tmp_path, capsys):
    dest = tmp_path / "warehouse"
    assert main(["generate", "-o", str(dest), "--customers", "25", "--seed", "5"]) == 0
    capsys.readouterr()
    loaded = Warehouse.load(dest)
    assert len(loaded.tables["customers"]) == 25
    assert "test" in loaded.anchors


def test_explain(capsys):
    code = main(
        ["explain", "--tasks", "churn", "--models", "flat,rfm", "--customers", "80", "--seed", "0"]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "P(churn)" in text
    assert "tickets" in text
