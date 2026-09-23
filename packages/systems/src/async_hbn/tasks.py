"""Fixed benchmark inputs and model-specific prompt rendering."""

from functools import lru_cache
import json
import hashlib
from pathlib import Path

from .config import ExperimentConfig, ModelConfig, sha256_file
from .resources import resource_dir


def data_dir():
    return resource_dir("data")


def benchmark_catalog():
    return json.loads((data_dir() / "benchmarks/manifest.json").read_text())["benchmarks"]


def load_fixed_tasks(config: ExperimentConfig):
    benchmark = config.benchmark
    if not benchmark.path.is_file():
        raise FileNotFoundError("Benchmark input is missing. Run make benchmarks from the source checkout; "
                                "for an installed wheel, set benchmark_path to the generated JSONL.")
    actual = sha256_file(benchmark.path)
    expected = benchmark.input_sha256
    if not expected or actual != expected:
        raise ValueError("Benchmark input SHA256 mismatch")
    rows = [json.loads(line) for line in benchmark.path.read_text().split("\n") if line.strip()]
    required = {"benchmark_id", "task_id", "task_index", "prompt", "user_prompt", "system_prompt"}
    if len(rows) != benchmark.expected_tasks:
        raise ValueError("Benchmark task count mismatch")
    tasks = {}
    for index, row in enumerate(rows):
        if set(row) != required or row["benchmark_id"] != benchmark.id or row["task_index"] != index:
            raise ValueError("Benchmark schema or task order mismatch")
        task_id = row["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("Invalid or duplicate task ID")
        if not all(isinstance(row[k], str) for k in ("prompt", "user_prompt")):
            raise ValueError("Prompt fields must be strings")
        if row["system_prompt"] is not None and not isinstance(row["system_prompt"], str):
            raise ValueError("System prompt must be a string or null")
        tasks[task_id] = row
    from .profile_rewards import load_frozen_profile
    if tuple(tasks) != load_frozen_profile(config).task_ids:
        raise ValueError("Profile and benchmark task order mismatch")
    return tasks


@lru_cache(maxsize=8)
def _load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)


def render_local_vllm_prompt(task, model: ModelConfig, *, tokenizer):
    messages = []
    if task.get("system_prompt"):
        messages.append({"role": "system", "content": task["system_prompt"]})
    messages.append({"role": "user", "content": task["user_prompt"]})
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    kwargs.update(model.chat_template_kwargs)
    return str(tokenizer.apply_chat_template(messages, **kwargs)) + model.generation_prompt_suffix


def load_and_render_tasks(config):
    tasks = load_fixed_tasks(config)
    tokenizer = _load_tokenizer(str(config.model.path))
    rendered = {
        key: {**task, "rendered_prompt": render_local_vllm_prompt(task, config.model, tokenizer=tokenizer)}
        for key, task in tasks.items()
    }
    text_hash, token_hash = hashlib.sha256(), hashlib.sha256()
    for key, task in rendered.items():
        text = task["rendered_prompt"]
        text_hash.update((json.dumps([key, text], ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        token_hash.update((json.dumps([key, tokenizer.encode(text)], separators=(",", ":")) + "\n").encode())
    from .release import read_result
    key = f"{config.benchmark.id}__model={config.model.alias}"
    expected = read_result("validation/prompt_targets.json")[key]
    if (text_hash.hexdigest() != expected["rendered_sha256"]
            or token_hash.hexdigest() != expected["token_ids_sha256"]):
        raise ValueError("Rendered prompt or tokenizer differs from the published input")
    return rendered
