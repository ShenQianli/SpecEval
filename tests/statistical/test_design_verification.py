from pathlib import Path
import shutil

import pytest

from speculative_eval import cli
from speculative_eval.verification import verify_designs, verify_published_parameters


ROOT = Path(__file__).resolve().parents[2]


def test_published_parameters():
    verify_published_parameters(ROOT)


def test_design_comparison_and_mismatch(tmp_path):
    reference = ROOT / "results/design"
    shutil.copytree(reference, tmp_path / "design")
    observed = tmp_path / "design"
    verify_designs(reference, observed)
    path = observed / "hbn_schedule.csv"
    text = path.read_text()
    lines = text.splitlines()
    fields = lines[1].split(",")
    fields[2] = str(int(fields[2]) + 1)
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError, match="m_star"):
        verify_designs(reference, observed)


def test_replay_uses_saved_designs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_audit", lambda root: ([], []))
    monkeypatch.setattr(cli, "write_csv", lambda *a: None)
    checked, loaded, calls = [], [], []
    monkeypatch.setattr(cli, "verify_published_parameters", lambda root: checked.append(root))
    monkeypatch.setattr(cli, "load_schedules", lambda path: loaded.append(path) or {})
    monkeypatch.setattr(cli, "reproduce_table", lambda *a, **k: calls.append((a, k)))
    args = cli.build_parser().parse_args(["--repo-root", str(tmp_path), "replay"])
    args.func(args)
    assert checked == [tmp_path]
    assert loaded == [tmp_path / "results/design"]
    assert calls[0][1]["draws"] == 8192
    assert calls[0][0][2] == tmp_path / "results/recomputed"
