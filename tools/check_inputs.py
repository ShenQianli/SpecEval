"""Validate the complete fixed benchmark/profile input boundary (standard library only)."""

import hashlib
import json
from pathlib import Path


def check_inputs(root):
    data = Path(root) / "data"
    catalog = json.loads((data / "benchmarks/manifest.json").read_text())["benchmarks"]
    if len(catalog) != 18:
        raise ValueError("Expected 18 benchmarks")
    tasks = {}
    fields = {"benchmark_id", "task_id", "task_index", "prompt", "user_prompt", "system_prompt"}
    if {p.name for p in (data / "benchmarks").glob("*.jsonl")} != {e["file"] for e in catalog.values()}:
        raise ValueError("Benchmark file set mismatch")
    for name, entry in catalog.items():
        path = data / "benchmarks" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Benchmark checksum mismatch: {name}")
        rows = [json.loads(line) for line in path.read_text().split("\n") if line]
        if len(rows) != entry["task_count"]:
            raise ValueError(f"Task count mismatch: {name}")
        if any(set(r) != fields or not isinstance(r["task_id"], str)
               or not all(isinstance(r[k], str) for k in ("prompt", "user_prompt"))
               or (r["system_prompt"] is not None and not isinstance(r["system_prompt"], str)) for r in rows):
            raise ValueError(f"Task schema mismatch: {name}")
        ids = [r["task_id"] for r in rows]
        if len(set(ids)) != len(ids) or any(
            r["task_index"] != i or r["benchmark_id"] != name for i, r in enumerate(rows)
        ):
            raise ValueError(f"Task identity/order mismatch: {name}")
        tasks[name] = ids
    manifest = json.loads((data / "profiles/manifest.json").read_text())
    entries = manifest["artifacts"]
    if len(entries) != 108 or manifest["artifact_count"] != 108:
        raise ValueError("Expected 108 profiles")
    if {p.name for p in (data / "profiles").glob("benchmark=*.json")} != {e["file"] for e in entries}:
        raise ValueError("Profile file set mismatch")
    seen = set()
    for entry in entries:
        path = data / "profiles" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Profile checksum mismatch: {path.name}")
        profile = json.loads(path.read_text())
        key = profile["benchmark_id"], profile["model_alias"]
        if key in seen or key != (entry["benchmark_id"], entry["model_alias"]):
            raise ValueError(f"Profile identity mismatch: {path.name}")
        seen.add(key)
        ids = tasks[key[0]]
        if profile["task_ids"] != ids or profile["task_indices"] != list(range(len(ids))):
            raise ValueError(f"Profile task order mismatch: {path.name}")
        if profile["tested_k"] != 1024 or profile["task_count"] != len(ids):
            raise ValueError(f"Profile size mismatch: {path.name}")
        counts = profile["pass_counts"]
        if len(counts) != len(ids) or any(type(c) is not int or not 0 <= c <= 1024 for c in counts):
            raise ValueError(f"Invalid counts: {path.name}")
    models = {model for _, model in seen}
    if len(models) != 6 or seen != {(b, m) for b in tasks for m in models}:
        raise ValueError("Incomplete benchmark/model grid")
    return len(tasks), len(seen), sum(map(len, tasks.values()))


if __name__ == "__main__":
    benchmarks, profiles, tasks = check_inputs(Path(__file__).resolve().parents[1])
    print(f"Validated {benchmarks} benchmarks / {tasks} tasks / {profiles} profiles")
