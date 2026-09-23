import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rebuild", ROOT / "tools/rebuild_benchmarks.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_source_coverage_and_pins():
    sources = json.loads((ROOT / "data/benchmark_sources.json").read_text())["sources"]
    benchmarks = json.loads((ROOT / "data/benchmarks/manifest.json").read_text())["benchmarks"]
    assert set(sources) == {m.source_key(b) for b in benchmarks}
    assert all(len(s["revision"]) == 40 and s["files"] for s in sources.values())
    assert sources["livecodebench"]["files"] == ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"]
    assert len(sources["mmlu"]["files"]) == 3


def test_unicode_and_prompt():
    rows = m.render([{"id": 60, "problem": "a\u2028b\n", "answer": "0"}], "aime24")
    assert rows[0]["task_id"] == "60"
    assert rows[0]["user_prompt"] == "Problem: a\u2028b\n\nMark your solution with \\boxed\nAnswer:"


def test_gpqa_shuffle_uses_child_index():
    source = [dict(zip(("Record ID", "High-level domain", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"),
                       (str(i), domain, "Q", "correct", "one", "two", "three")))
              for i, domain in enumerate(("Physics", "Biology", "Biology"))]
    rows = m.render(source, "gpqa_diamond__domain_biology")
    import random
    for i, row in enumerate(rows):
        choices = ["correct", "one", "two", "three"]
        random.Random(42+i).shuffle(choices)
        assert ", ".join(f"{chr(65+j)}) {v}" for j, v in enumerate(choices)) in row["user_prompt"]


def test_hle_filter_and_system():
    rows = m.render([{"id": "x", "question": "Q", "image": ""}, {"id": "y", "question": "R", "image": "image"}], "hle__text_only_sample100")
    assert len(rows) == 1 and rows[0]["system_prompt"] == m.HLE_SYSTEM


def test_mmlu_source_index_survives_filter():
    rows = [dict(question="Q", choices=["a", "b", "c", "d"], answer=0, subject=s)
            for s in ("college_physics", "college_biology")]
    result = m.render(rows, "mmlu_science__subject_college_biology")
    assert result[0]["task_id"].startswith("source-row:1:sha256:")


def test_sampling_is_ordered_and_capped():
    rows = [dict(id=str(i), question="Q", image="") for i in range(120)]
    chosen = m.select_rows(rows, "hle__text_only_sample100")
    assert len(chosen) == 100
    assert chosen == m.select_rows(list(reversed(rows)), "hle__text_only_sample100")


def test_no_network_in_source_dir_mode(tmp_path):
    args = m.parser().parse_args(["--source-dir", str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="never downloads"):
        m.Access(args).fetch("gpqa", {}, "gpqa_diamond.csv")
    path = tmp_path / "gpqa/gpqa_diamond.csv"
    path.parent.mkdir()
    path.write_text("local")
    assert m.Access(args).fetch("gpqa", {}, path.name) == path


def test_noninteractive_requires_terms_acknowledgement():
    access = m.Access(m.parser().parse_args(["--non-interactive"]))
    with pytest.raises(ValueError, match="acknowledge-terms"):
        access.acknowledge({})


def test_interactive_terms_decline(monkeypatch):
    access = m.Access(m.parser().parse_args([]))
    access.interactive = True
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(ValueError, match="before downloading"):
        access.acknowledge({})


def mock_hub(monkeypatch, download):
    class Error(Exception):
        def __init__(self, status):
            self.response = types.SimpleNamespace(status_code=status)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    hub.get_token = lambda: None
    utils = types.ModuleType("huggingface_hub.utils")
    utils.HfHubHTTPError = Error
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return Error


def test_access_retry_hidden_token_not_saved(monkeypatch, tmp_path, capsys):
    calls = []
    def download(*a, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise Error(403)
        return str(tmp_path / "file")
    Error = mock_hub(monkeypatch, download)
    access = m.Access(m.parser().parse_args([])); access.interactive = True
    monkeypatch.setattr("builtins.input", lambda _: "t")
    monkeypatch.setattr(m.getpass, "getpass", lambda _: "test-only-secret")
    assert access.fetch("gpqa", {"repo": "owner/repo", "revision": "f"*40}, "file") == tmp_path / "file"
    assert calls[1]["token"] == "test-only-secret"
    assert calls[1]["revision"] == "f"*40
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert "test-only-secret" not in captured.out + captured.err


def test_access_can_pause_without_writing(monkeypatch, tmp_path):
    def download(*a, **kw):
        raise Error(403)
    Error = mock_hub(monkeypatch, download)
    access = m.Access(m.parser().parse_args([])); access.interactive = True
    monkeypatch.setattr("builtins.input", lambda _: "q")
    with pytest.raises(ValueError, match="Paused for access approval"):
        access.fetch("hle", {"repo": "owner/repo", "revision": "f"*40}, "file")
    assert not list(tmp_path.iterdir())


def test_access_noninteractive_failure(monkeypatch):
    def download(*a, **kw):
        raise Error(401)
    Error = mock_hub(monkeypatch, download)
    access = m.Access(m.parser().parse_args(["--non-interactive"]))
    with pytest.raises(ValueError, match="Authorize your HF account"):
        access.fetch("gpqa", {"repo": "owner/repo", "revision": "f"*40}, "file")


def test_hash_mismatch_does_not_write(tmp_path):
    rows = m.render([{"id": 1, "problem": "Q"}], "aime24")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.encode_and_validate(rows, "aime24", {"task_count": 1, "sha256": "0"*64}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_plan_does_not_write_or_require_auth(tmp_path):
    m.build(m.parser().parse_args(["--plan", "--output-dir", str(tmp_path / "absent")]))
    assert not (tmp_path / "absent").exists()


def test_frozen_manifest_collision_stops_before_write(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="nothing was replaced"):
        m.build(m.parser().parse_args(["--output-dir", str(tmp_path)]))
    assert manifest.read_text() == "{}" and len(list(tmp_path.iterdir())) == 1


def test_unexpected_output_files_stop_before_download(tmp_path):
    (tmp_path / "unexpected.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Unexpected JSONL"):
        m.build(m.parser().parse_args(["--output-dir", str(tmp_path)]))
    assert len(list(tmp_path.iterdir())) == 1


def test_resume_skips_verified_outputs(tmp_path, monkeypatch):
    rows = m.render([{"id": 1, "problem": "Q"}], "aime24")
    data = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows).encode()
    root = tmp_path / "repo"
    (root / "data/benchmarks").mkdir(parents=True)
    (root / "data/profiles").mkdir()
    (root / "data/benchmarks/aime24.jsonl").write_bytes(data)
    entry = dict(file="aime24.jsonl", task_count=1, sha256=hashlib.sha256(data).hexdigest())
    (root / "data/benchmarks/manifest.json").write_text(json.dumps({"benchmarks": {"aime24": entry}}))
    (root / "data/benchmark_sources.json").write_text(json.dumps({"sources": {"aime24": {}}}))
    for i in range(6):
        (root / f"data/profiles/benchmark=aime24__model={i}.json").write_text(json.dumps(dict(task_ids=["1"], task_indices=[0])))
    profiles = root / "data/profiles"
    (profiles / "manifest.json").write_text(json.dumps({"artifacts": [dict(file=p.name, sha256=m.digest(p)) for p in profiles.glob("benchmark=*.json")]}))
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m.Access, "fetch", lambda *a: pytest.fail("unexpected download"))
    m.build(m.parser().parse_args(["--non-interactive"]))
