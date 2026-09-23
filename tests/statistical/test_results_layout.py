import hashlib
import json
from pathlib import Path

from speculative_eval import cli


ROOT = Path(__file__).resolve().parents[2]


def test_published_input_checksums():
    root = ROOT / "results"
    manifest = json.loads((root / "validation/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_reproduction_output_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_audit", lambda root: ([], []))
    monkeypatch.setattr(cli, "write_csv", lambda path, rows: None)
    calls = []
    monkeypatch.setattr(cli, "reproduce_designs", lambda *a, **k: {})
    monkeypatch.setattr(cli, "reproduce_table", lambda *a, **k: calls.append((a, k)) or [])
    args = cli.build_parser().parse_args(["--repo-root", str(tmp_path), "reproduce"])
    args.func(args)
    assert calls[0][0][2] == tmp_path / "results/recomputed"
    assert calls[0][1]["tables_output"] == tmp_path / "results/tables"
