import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from async_hbn.config import sha256_file
from async_hbn.release import load_experiment, read_result
from async_hbn.tasks import load_fixed_tasks, render_local_vllm_prompt
from test_release import config_file


ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not all((ROOT / "data/benchmarks" / e["file"]).is_file()
            for e in json.loads((ROOT / "data/benchmarks/manifest.json").read_text())["benchmarks"].values()),
    reason="Input integration tests require make benchmarks; make systems-input-test enforces preparation",
)


def test_complete_fixed_input_grid():
    spec = importlib.util.spec_from_file_location("check_inputs", ROOT / "tools/check_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_inputs(ROOT) == (18, 108, 1418)


@pytest.mark.parametrize("profile", sorted(read_result("configurations/profiles.json")))
def test_all_profiles_load_without_benchmark_adapters(tmp_path, profile):
    config, _, _ = load_experiment(config_file(tmp_path, profile=profile))
    tasks = load_fixed_tasks(config)
    assert len(tasks) == config.design.n_tasks
    assert all("answer_json" not in task and "scorer_payload_json" not in task for task in tasks.values())


def test_corrupt_input_is_rejected(tmp_path):
    config, _, _ = load_experiment(config_file(tmp_path))
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(config.benchmark.path.read_bytes() + b"\n")
    config = replace(config, benchmark=replace(config.benchmark, path=path))
    with pytest.raises(ValueError, match="SHA256"):
        load_fixed_tasks(config)


def test_reordered_tasks_are_rejected_even_with_matching_file_hash(tmp_path):
    config, _, _ = load_experiment(config_file(tmp_path))
    rows = list(load_fixed_tasks(config).values())[::-1]
    for index, row in enumerate(rows):
        row["task_index"] = index
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = replace(config, benchmark=replace(config.benchmark, path=path, input_sha256=sha256_file(path)))
    with pytest.raises(ValueError, match="task order"):
        load_fixed_tasks(config)


def test_unicode_line_separator_in_prompt_is_preserved(tmp_path):
    config, _, _ = load_experiment(config_file(tmp_path))
    rows = list(load_fixed_tasks(config).values())
    rows[0]["user_prompt"] = "first\u2028second\u0085third"
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    config = replace(config, benchmark=replace(config.benchmark, path=path, input_sha256=sha256_file(path)))
    assert load_fixed_tasks(config)[rows[0]["task_id"]]["user_prompt"] == rows[0]["user_prompt"]


def test_render_uses_model_options_and_generation_suffix(tmp_path):
    config, _, _ = load_experiment(config_file(tmp_path))
    model = replace(config.model, generation_prompt_suffix="suffix")
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
            return "rendered"
    assert render_local_vllm_prompt({"user_prompt": "question"}, model, tokenizer=Tokenizer()) == "renderedsuffix"


def test_changed_tokenizer_is_rejected_before_generation(tmp_path, monkeypatch):
    from async_hbn import tasks
    config, _, _ = load_experiment(config_file(tmp_path))
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "different prompt"
        def encode(self, text):
            return [0]
    monkeypatch.setattr(tasks, "_load_tokenizer", lambda path: Tokenizer())
    with pytest.raises(ValueError, match="tokenizer"):
        tasks.load_and_render_tasks(config)


def test_changed_model_configuration_is_rejected_before_generation(tmp_path):
    from async_hbn.config import verify_static_inputs
    config, _, _ = load_experiment(config_file(tmp_path))
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text("{}")
    config = replace(config, model=replace(config.model, path=path))
    with pytest.raises(ValueError, match="Model configuration"):
        verify_static_inputs(config)
