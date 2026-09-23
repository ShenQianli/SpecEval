# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["huggingface-hub==0.34.4", "pyarrow==21.0.0"]
# ///
"""Rebuild fixed benchmark prompts from pinned upstream files, without scoring."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MATH_PROMPT = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
CHOICE_PROMPT = (
    "Return your final response within \\boxed{{}} and only include the letter choice "
    "({labels}) as your final response.\nProblem: {problem}\nOptions: {options}\nAnswer:"
)
HLE_SYSTEM = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
STDIN = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
FUNCTION = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
LCB_FIELDS = {"question_id", "question_content", "difficulty", "contest_date", "public_test_cases", "metadata"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_key(benchmark):
    for prefix, key in (("gpqa_", "gpqa"), ("supergpqa_", "supergpqa"),
                        ("mmlu_", "mmlu"), ("hle__", "hle"),
                        ("livecodebench_", "livecodebench"), ("math_", "math")):
        if benchmark.startswith(prefix):
            return key
    return benchmark


def read_rows(path, source):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        file = pq.ParquetFile(path)
        columns = ["id", "question", "image"] if source == "hle" else None
        for batch in file.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                if source == "mmlu":
                    row.setdefault("subject", path.parent.name)
                yield row
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield {k: v for k, v in row.items() if k in LCB_FIELDS} if source == "livecodebench" else row


def stable_id(row, key, source_index):
    if key == "mmlu":
        identity = {k: row[k] for k in ("question", "choices", "answer", "subject")}
        return f"source-row:{source_index}:sha256:{hashlib.sha256(canonical(identity).encode()).hexdigest()}"
    field = {"gpqa": "Record ID", "supergpqa": "uuid", "hle": "id", "livecodebench": "question_id"}[key]
    if row.get(field) is None or str(row[field]) == "":
        raise ValueError(f"Missing stable ID: {field}")
    return str(row[field])


def select_rows(rows, benchmark):
    key = source_key(benchmark)
    if key == "math":
        if len(rows) != 500:
            raise ValueError("MATH source must contain 500 rows")
        return [(None, rows[i]) for i in sorted(random.Random(42).sample(range(500), 100))]
    if key not in {"gpqa", "supergpqa", "mmlu", "hle", "livecodebench"}:
        return [(None, row) for row in rows]
    if key == "supergpqa":
        rows = [r for r in rows if r.get("discipline") == "Science" and r.get("field") in {"Physics", "Chemistry", "Biology"}]
    if key == "livecodebench":
        rows = [r for r in rows if datetime.fromisoformat(str(r["contest_date"]).replace("Z", "+00:00")).date().isoformat() > "2024-03-30"]
        if len(rows) != 441:
            raise ValueError("LiveCodeBench v5 date-filtered pool must contain 441 rows")
    pool = []
    for i, row in enumerate(rows):
        include = {
            "gpqa": lambda: row["High-level domain"].lower() == benchmark.rsplit("_", 1)[1],
            "supergpqa": lambda: row["field"].lower() == benchmark.rsplit("_", 1)[1],
            "mmlu": lambda: row["subject"] == benchmark.split("__subject_", 1)[1],
            "hle": lambda: row.get("image") in (None, ""),
            "livecodebench": lambda: row["difficulty"] == benchmark.rsplit("_", 1)[1],
        }[key]()
        if include:
            pool.append((stable_id(row, key, i), row))
    if len({i for i, _ in pool}) != len(pool):
        raise ValueError(f"Duplicate source IDs: {benchmark}")
    if len(pool) > 100:
        def rank(item):
            text = f"sha256_rank_without_replacement_v1|seed=42|child={benchmark}|task={item[0]}"
            return hashlib.sha256(text.encode()).hexdigest(), item[0]
        pool = sorted(pool, key=rank)[:100]
    return pool


def render(rows, benchmark):
    key = source_key(benchmark)
    output = []
    for index, (task_id, row) in enumerate(select_rows(rows, benchmark)):
        system = None
        if key in {"aime24", "aime25", "aime26", "hmmt", "math"}:
            problem = str(row["problem"])
            if key == "math":
                identity = {k: row.get(k) for k in ("problem", "answer")}
                task_id = row.get("unique_id") or "sha256:" + hashlib.sha256(canonical(identity).encode()).hexdigest()
            else:
                task_id = row.get("id", row.get("problem_idx", index))
            user = MATH_PROMPT.format(problem=problem)
        elif key in {"gpqa", "supergpqa", "mmlu"}:
            problem = str(row["Question"] if key == "gpqa" else row.get("question") or row.get("problem") or "")
            if key == "gpqa":
                choices = [row[name] for name in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
                random.Random(42 + index).shuffle(choices)
            else:
                choices = list(row.get("options") or row.get("choices") or [])
            if not 1 <= len(choices) <= 10:
                raise ValueError(f"Invalid option count: {benchmark}")
            options = ", ".join(f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices))
            labels = "A, B, C, or D" if key == "gpqa" else f"A through {chr(64+len(choices))}"
            user = CHOICE_PROMPT.format(problem=problem, options=options, labels=labels)
        elif key == "hle":
            problem = str(row["question"])
            user, system = problem, HLE_SYSTEM
        else:
            problem = str(row["question_content"])
            public = json.loads(row["public_test_cases"])
            types = {t["testtype"] for t in public}
            if len(types) > 1 or types - {"stdin", "functional"}:
                raise ValueError("Unsupported LiveCodeBench test types")
            metadata = json.loads(row.get("metadata") or "{}")
            kind = next(iter(types)) if types else ("functional" if metadata.get("func_name") else "stdin")
            user = (STDIN if kind == "stdin" else FUNCTION) + problem
        output.append(dict(benchmark_id=benchmark, task_id=str(task_id), task_index=index,
                           prompt=problem, user_prompt=user, system_prompt=system))
    return output


def encode_and_validate(rows, benchmark, entry, profiles):
    if len(rows) != entry["task_count"] or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError(f"Task count/identity mismatch: {benchmark}")
    data = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows).encode()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"JSONL SHA256 mismatch: {benchmark}; no output was replaced")
    files = list(profiles.glob(f"benchmark={benchmark}__model=*.json"))
    if len(files) != 6:
        raise ValueError(f"Expected six profiles: {benchmark}")
    checksums = {e["file"]: e["sha256"] for e in json.loads((profiles / "manifest.json").read_text())["artifacts"]}
    for path in files:
        if digest(path) != checksums.get(path.name):
            raise ValueError(f"Profile checksum mismatch: {path.name}")
        p = json.loads(path.read_text())
        if p["task_ids"] != [r["task_id"] for r in rows] or p["task_indices"] != [r["task_index"] for r in rows]:
            raise ValueError(f"Profile task order mismatch: {path.name}")
    return data


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Access:
    def __init__(self, args):
        self.args = args
        self.interactive = sys.stdin.isatty() and not args.non_interactive
        self.token = None

    def acknowledge(self, sources):
        print("Source terms (review before downloading or using data):")
        for key, source in sources.items():
            print(f"  {key}: https://huggingface.co/datasets/{source['repo']}")
        print("GPQA/HLE require personal access approval and request no public redistribution.\n"
              "HMMT has noncommercial/share-alike terms. Other sources retain their own terms.\n"
              "See data/REDISTRIBUTION.md. This script cannot accept upstream agreements for you.")
        if "livecodebench" in sources:
            print("LiveCodeBench's raw files occupy several GiB even though only prompt metadata is used.")
        if self.args.acknowledge_terms:
            return
        if not self.interactive:
            raise ValueError("Review source terms, then rerun with --acknowledge-terms (not an upstream access grant).")
        if input("Have you reviewed the applicable terms for your intended use? [y/N] ").strip().lower() not in {"y", "yes"}:
            raise ValueError("Stopped before downloading. Rerun the same command after resolving permissions.")

    def fetch(self, key, source, filename):
        if self.args.source_dir:
            path = self.args.source_dir / key / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing source file: {path}; source-dir mode never downloads")
            return path
        from huggingface_hub import hf_hub_download, get_token
        from huggingface_hub.utils import HfHubHTTPError
        while True:
            try:
                return Path(hf_hub_download(source["repo"], filename, repo_type="dataset",
                    revision=source["revision"], cache_dir=self.args.cache_dir,
                    token=self.token or get_token() or False, endpoint="https://huggingface.co",
                    local_files_only=self.args.offline))
            except HfHubHTTPError as error:
                status = getattr(error.response, "status_code", None)
                if status not in (401, 403) or self.args.offline:
                    raise ValueError(f"Download failed for {key}/{filename} (HTTP {status}); rerun to resume.") from None
                print(f"Access needed: https://huggingface.co/datasets/{source['repo']}\n"
                      "Open the page, sign in and personally accept/request access.\n"
                      "Use a read token authorized for this dataset: https://huggingface.co/settings/tokens")
                if not self.interactive:
                    raise ValueError("Authorize your HF account and set HF_TOKEN or run hf auth login, then resume.") from None
                answer = input("[Enter] retry existing login; [t] enter a hidden token; [q] stop and resume later: ").strip().lower()
                if answer == "q":
                    raise ValueError("Paused for access approval; rerun the same command to resume.") from None
                if answer == "t":
                    self.token = getpass.getpass("Read token (not saved by this script): ").strip()


def build(args):
    catalog_bytes = (ROOT / "data/benchmarks/manifest.json").read_bytes()
    catalog = json.loads(catalog_bytes)["benchmarks"]
    registry = json.loads((ROOT / "data/benchmark_sources.json").read_text())["sources"]
    selected = args.benchmarks or list(catalog)
    unknown = set(selected) - set(catalog)
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
    selected = list(dict.fromkeys(selected))
    needed = list(dict.fromkeys(source_key(b) for b in selected))
    if args.plan:
        for key in needed:
            s = registry[key]
            print(f"{key}: {s['repo']} @ {s['revision']} ({len(s['files'])} files)")
            for name in s["files"]:
                print(f"  {key}/{name}")
        return
    manifest = args.output_dir / "manifest.json"
    if manifest.exists() and manifest.read_bytes() != catalog_bytes:
        raise ValueError("Output manifest differs from the frozen catalog; nothing was replaced")
    extras = {p.name for p in args.output_dir.glob("*.jsonl")} - {e["file"] for e in catalog.values()}
    if extras:
        raise ValueError(f"Unexpected JSONL files in output directory: {sorted(extras)}; nothing was replaced")
    pending = []
    for name in selected:
        path = args.output_dir / catalog[name]["file"]
        if path.exists() and not args.rebuild:
            if digest(path) == catalog[name]["sha256"]:
                rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
                encode_and_validate(rows, name, catalog[name], ROOT / "data/profiles")
                print(f"[verified, skip] {name}")
                continue
            raise ValueError(f"Existing file differs: {path}. Move it aside or use --rebuild; replacement occurs only after verification.")
        pending.append(name)
    access = Access(args)
    if pending:
        pending_keys = list(dict.fromkeys(source_key(b) for b in pending))
        access.acknowledge({key: registry[key] for key in pending_keys})
        for key in pending_keys:
            source = registry[key]
            print(f"[source] {key} @ {source['revision']}", flush=True)
            rows = []
            for filename in source["files"]:
                print(f"  {filename}", flush=True)
                path = access.fetch(key, source, filename)
                rows.extend(read_rows(path, key))
            for name in pending:
                if source_key(name) != key:
                    continue
                output = render(rows, name)
                data = encode_and_validate(output, name, catalog[name], ROOT / "data/profiles")
                atomic_write(args.output_dir / catalog[name]["file"], data)
                print(f"[verified, saved] {name}: {len(output)} tasks", flush=True)
    if not manifest.exists():
        atomic_write(manifest, catalog_bytes)
    print(f"Verified {len(selected)}/{len(catalog)} benchmarks in {args.output_dir}")
    if set(selected) == set(catalog):
        print("Complete: 18 JSONL files / 1418 tasks, exact SHA256 and all profile orders verified.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", help="benchmark IDs; default: all 18")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmarks")
    p.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    p.add_argument("--source-dir", type=Path, help="authorized local raw files: SOURCE_KEY/FILENAME; no network")
    p.add_argument("--offline", action="store_true", help="use only the Hugging Face cache")
    p.add_argument("--acknowledge-terms", action="store_true", help="confirm you reviewed applicable terms; does not grant gated access")
    p.add_argument("--non-interactive", action="store_true", help="fail with instructions instead of prompting")
    p.add_argument("--rebuild", action="store_true", help="regenerate even matching outputs; verify before replacing")
    p.add_argument("--plan", action="store_true", help="list pinned files without downloading or writing")
    return p


def main():
    try:
        build(parser().parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted. Verified outputs are kept; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        # Do not expose upstream HTTP errors, which may contain signed URLs.
        message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        print(f"Stopped: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
