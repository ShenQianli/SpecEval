# Fixed experiment inputs

The reproducibility boundary starts with the files in this directory.
It does not require the original response corpus or benchmark scoring software.
Profiles are provided; benchmark tasks/prompts are generated locally from pinned
upstream files.

Third-party benchmark text is subject to upstream terms, not the software's
MIT license. See the [source and access information](SOURCES.md) before sharing
these files.

## One-command preparation

From the repository root, with Python and [uv](https://docs.astral.sh/uv/) available:

```bash
make benchmarks
make audit-inputs
```

The first command installs its small, separately locked script environment,
shows source terms, and downloads exact revisions from Hugging Face. If a
dataset requires access, it shows the official page and waits while you sign
in and personally accept/request permission. Press Enter to retry an existing
Hugging Face login, `t` to enter a read token without terminal echo, or `q` to
stop. Entered tokens stay in process memory and are not saved by the script.
Existing `HF_TOKEN` or `hf auth login` credentials are also supported.
Never paste tokens into issues, logs, or chat messages.

Completed files are hash-verified before writing; rerunning the command skips
verified outputs and reuses the download cache. Invalid existing outputs are
not overwritten unless `--rebuild` is supplied, and even then only after the
replacement passes validation. An interruption does not invalidate completed
files. No revision fallback, answer scoring, model generation or private-test
execution occurs.
LiveCodeBench's upstream JSONL files contain test payloads and occupy several
GiB; the builder discards those payloads without decoding or executing them.

Additional options use the same entry point:

```bash
make benchmarks BENCHMARK_ARGS="--plan"
make benchmarks BENCHMARK_ARGS="--benchmarks aime24 --output-dir ../spec-eval-inputs"
make benchmarks BENCHMARK_ARGS="--cache-dir ../benchmark-cache --acknowledge-terms --non-interactive"
make benchmarks BENCHMARK_ARGS="--offline --acknowledge-terms --non-interactive"
make benchmarks BENCHMARK_ARGS="--source-dir ../authorized-sources --acknowledge-terms --non-interactive"
```

`--acknowledge-terms` confirms that you reviewed the applicable source terms;
it neither accepts upstream agreements nor bypasses access control. In
noninteractive mode, missing permissions produce an actionable error instead
of waiting for input. `--offline` uses only the Hugging Face cache. Alternatively,
`--source-dir` reads raw files you already obtained with permission, arranged as
`SOURCE_KEY/FILENAME`; `--plan` lists the exact layout. That mode never downloads.
Local files undergo the same final JSONL and profile checks as downloaded files.

After all inputs are ready, `make systems-input-test` runs the complete input
integration tests and fails if preparation is incomplete. General CPU tests can
run without downloading benchmark text; that integration module is skipped
when inputs are absent. Tables 1--4 and statistical replay need only the provided
profiles/results and do not require this preparation step.

The source revision and ordered file list are in `benchmark_sources.json`.
LiveCodeBench v5 uses `test.jsonl` through `test5.jsonl`, excludes later versions,
and keeps dates strictly after 2024-03-30. Its stdin/function prompt choice uses
public test-type metadata, or function-name metadata when no public tests exist;
the complete frozen output hash validates this choice. MMLU source subjects are
concatenated in physics, chemistry, biology order before constructing stable IDs.
GPQA option permutations use `random.Random(42 + child_row_index)`.

JSONL encoding is UTF-8 with compact separators, fixed field order, and one LF
after every record. All 18 outputs must match the frozen hashes and the ID/order
of all 108 profiles. Model-specific chat rendering remains a separate runtime
check. Preparation verifies JSONL hashes and task order. Source-artifact
identifiers remain fixed components of the published protocol.

## Benchmark tasks and prompts

After preparation, `benchmarks/` contains one JSONL per benchmark: 18 benchmarks
and 1,418 tasks. These generated files are not included in the source distribution.
Each row contains `benchmark_id`, `task_id`, `task_index`, `prompt`,
`user_prompt`, and nullable `system_prompt`. The first prompt is the task
text; the user prompt includes the benchmark instructions used for generation.
Model-specific chat templates and generation suffixes are applied at runtime.
JSONL records are separated by LF; Unicode separators inside text are preserved.

The task key is `(benchmark_id, task_id)`. Task indices preserve the experimental
order. Answers, reference solutions, and execution-based scoring tests are not
inputs to these experiments and are not included. Examples embedded in the
problem statement remain part of the prompt.

`benchmarks/manifest.json` records file SHA-256, task count, source repository
and split, and the source artifact identity associated with the published
protocol. The JSONL checksum validates the delivered bytes; it is distinct from
the protocol identity used for request seeds.

The benchmarks are AIME 2024/2025/2026, HMMT, MATH-100, Biology/Chemistry/Physics
subsets of GPQA Diamond, SuperGPQA Science and MMLU Science, HLE text-only,
and Easy/Medium/Hard LiveCodeBench subsets after March 30, 2024.
MATH-100 uses a seed-42 sample of MATH-500, preserving sorted source indices.
Subject/difficulty children and the HLE text-only pool are capped at 100 tasks:
larger pools use deterministic SHA-256 ranking of stable task IDs with seed 42;
smaller pools retain all tasks in source order. Exact task membership and order
are fixed by the delivered files, rather than reconstructed during execution.

## Task-probability profiles

`profiles/` contains 108 benchmark--checkpoint JSON files and a checksum
manifest. Each profile stores `task_ids`, `task_indices`, `pass_counts`,
`tested_k=1024`, benchmark/model identity, and the source scorer version.
For task i, the fixed probability is `p_i = pass_counts[i] / 1024`.
The zero-variance profile is excluded by the statistical audit, leaving the
107 profiles evaluated in the paper.

These counts summarize 1,024 generated and scored responses per task.
Generation used temperature 0.6, top-p 0.95 and a 16,384-token response limit;
Qwen3.5 used thinking mode with an 8,192-token thinking budget. Mathematical
answers were normalized against benchmark answers, multiple-choice tasks used
choice extraction, LiveCodeBench used code tests, and HLE used a fixed judge
protocol. Scorer versions are recorded in the
profile files and manifest. This describes input provenance; the repository
does not regenerate or rescore this corpus.

Statistical replay samples from these fixed probabilities. System experiments
perform real generation but reveal deterministic profile-Bernoulli rewards
only after completion. Generated answer content is not scored.

## Validation and source terms

Run `python tools/check_inputs.py` from the repository root to check all
file hashes, task IDs, ordering, counts, and the complete six-model grid.
System input checks repeat the selected benchmark/profile validation before
generation. Source protocol identities remain fixed when local paths change.

Source repositories and their licenses/terms continue to apply to benchmark
text. The repository's MIT code license does not relicense that text.
Models and the original response corpus are not bundled.
