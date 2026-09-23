# Single-run system experiments

This package executes one model / benchmark / budget / strategy configuration.
The strategies are `uniform`, `hbn-sync`, and `hbn-async`.

## Reproduce Tables 2--4 without generation

From the repository root, with Python 3.11 or newer:

```bash
python tools/reproduce_tables.py --output-dir results/tables
```

This uses only the Python standard library. It verifies input-file hashes
and the complete 107-profile, four-budget, three-strategy grid plus the 428
rounded-equivalent Uniform measurements, then writes
unrounded CSV and JSON tables. The default output directory is `results/tables/`.
It aggregates the published measurements. See [measurement documentation](../../results/systems/README.md) for schema,
provenance, aggregation, and limitations. Table 1 reproduction remains in the
repository-root README.

## Configure one experiment

Run these commands from the repository root, with `uv` installed:

```bash
make benchmarks
uv sync --project packages/systems --extra gpu --extra test --locked
uv run --project packages/systems --extra gpu --locked async-hbn list
cp configs/systems/example.yaml configs/systems/my.local.yaml
# Edit model_path and select model/benchmark/budget/strategy.
uv run --project packages/systems --extra gpu --locked async-hbn show --config configs/systems/my.local.yaml
uv run --project packages/systems --extra gpu --locked async-hbn check --config configs/systems/my.local.yaml
uv run --project packages/systems --extra gpu --locked async-hbn run --config configs/systems/my.local.yaml
```

`make benchmarks` guides you through source terms and any required Hugging Face
authorization, then rebuilds and verifies the fixed inputs. It skips files that
already pass verification. See [data preparation](../../data/README.md) for details.

`show` resolves settings without loading models. `check` verifies dataset and
profile hashes, the model configuration hash, task order, and rendered-text/token-ID digests; it loads a
tokenizer but does not start GPU generation. `run` executes exactly one strategy
once, including input checks, model loading, warmup, timed evaluation, and shutdown.
Use idle GPUs; the guard rejects unrelated GPU processes.

All user-relative paths are resolved against the YAML file's directory:

| Field | Meaning |
|---|---|
| `model` | One of the six published aliases, e.g. `qwen3_5_4b` |
| `benchmark` | Exact benchmark ID, e.g. `aime24`; `list` shows all supported pairs |
| `budget` | `8`, `16`, `32`, or `64`; for rounded-equivalent Uniform, the actual integer budget in the target table |
| `reference_budget` | Optional; selects a rounded-equivalent Uniform experiment relative to HBN at `8`, `16`, `32`, or `64` |
| `strategy` | `uniform`, `hbn-sync`, or `hbn-async` |
| `model_path` | Local checkpoint directory including tokenizer and weights |
| `benchmark_path` | Optional path to the JSONL produced by `make benchmarks`; required when using an installed wheel outside the source checkout |
| `profile_data_dir` | Frozen profiles; defaults to repository `data/profiles/` |
| `output_dir` | Parent directory for uniquely named run outputs |
| `gpu_ids` | Optional; defaults to eight GPUs `[0,...,7]` |
| `max_inflight_per_engine` | Optional; defaults to 32, hence total concurrency 256 |

The published catalog selects the full-precision ex-ante pilot size and stage
weight for the chosen task count and budget. It also supplies benchmark-specific
prompt settings, token limits, model settings, and frozen Bernoulli rewards.
Changing the strategy does not change pilot identities or rewards. Profiles are
included. Run `make benchmarks` from the repository root to prepare task/prompt
JSONL files; the command guides you through any source access requirements.
Model checkpoints must also be supplied externally. Both input file hashes and
task ordering are checked. Benchmark text is not packaged in wheels; when using
an installed wheel, set `benchmark_path` to the prepared file.

The public interface intentionally fixes seed, sampling, and statistical design
to the published protocol; unknown keys are rejected. To study a different
method/protocol, explicitly version a new configuration rather than overriding
these fields under the published identity. Changing GPU count or concurrency is
supported, but those timings are not the published hardware setting.

## Uniform at rounded-down equivalent budgets

This experiment measures Uniform at budgets derived from HBN's fixed-profile
variance ratios. For profile `j` and reference budget `b`, let `r[j,b]` be
the unrounded ratio in `results/replay/hbn_pair_results.csv`. Uniform receives
`floor(b/r[j,b])` rollouts per task. Its theoretical variance relative to HBN
is `b/(floor(b/r[j,b])*r[j,b])`, which is at least one.

The complete 428-row mapping is
`results/systems/uniform_equivalent_budgets.csv`. To regenerate it without
models, GPU dependencies, or replay:

```bash
make systems-equivalent-targets
```

The command writes `results/recomputed/uniform_equivalent_budgets.csv` and
refuses to overwrite an existing file. Target generation checks the fixed
replay checksum and computes the floor from its decimal representation without
intermediate rounding. Recomputed replay draws do not redefine these targets.

Every setting is measured afresh, including the 64 settings whose rounded
budget equals the reference budget. No measured time is reused. The reference
budget supplies the original pilot/continuation numbering, model, prompts,
reward protocol, and serving configuration. Uniform has no pilot barrier;
all requests are eligible from the beginning. Only the total budget and its
derived continuation count change. A distinct, path-independent protocol
fingerprint identifies every equivalent experiment, including unchanged-budget
settings. It also changes generation seeds; token-identical output across runs
is not assumed.

For AIME24 / Qwen3.5-4B at reference budget 16, the target is 17 rollouts per
task. From the repository root:

```bash
cp configs/systems/uniform_equivalent.yaml configs/systems/equivalent.local.yaml
# Set model_path and output_dir; use the target-table budget for the chosen profile/reference_budget.
uv run --project packages/systems --extra gpu --locked async-hbn show --config configs/systems/equivalent.local.yaml
uv run --project packages/systems --extra gpu --locked async-hbn check --config configs/systems/equivalent.local.yaml
uv run --project packages/systems --extra gpu --locked async-hbn run --config configs/systems/equivalent.local.yaml
```

`reference_budget` is supported only for `strategy: uniform`. The loader
rejects an actual budget that differs from its target. The run manifest records
the reference and actual budgets and target fingerprint. A single invocation
executes one strategy once; no task distribution service is required.

Export these runs separately from the original three-strategy grid:

```bash
uv run --project packages/systems --locked async-hbn measurements --equivalent \
  --runs /path/to/equivalent_run1 /path/to/equivalent_run2 \
  --output results/recomputed/uniform_equivalent_measurements.csv
```

The exporter checks target budgets, new protocol identities, every accepted
request's task, seed and frozen reward, full serving settings, input hashes,
request accounting, GPU monitoring and the run UUID. It rejects original runs
even when the budget is unchanged, retries, incomplete runs, and duplicate
observations. Measurements use eight replicas and concurrency 256.

With all 428 new observations, generate the measured Table 4 comparison:

```bash
make systems-equivalent-tables \
  EQUIVALENT_MEASUREMENTS=results/recomputed/uniform_equivalent_measurements.csv
```

This writes `results/recomputed/equivalent_tables/`. Tables 2 and 3 retain
their original inputs. Table 4 sums the freshly measured rounded-budget
Uniform times, divides by the summed original Uniform times at each reference
budget, and subtracts one. HBN-async uses the same denominator. The difference
is reported in percentage points. Average continuous and rounded budgets are
weighted by each profile's task count. The rounded/continuous budget percentage
is the ratio of their total rollout counts. The input must cover all 428 settings.

`make paper-tables` uses the published equivalent measurement file by default.
It needs only Python's standard library, not a model service.

## What the run measures

Every request performs real generation. Its reward is a frozen Bernoulli draw
from the included profile, revealed only after generation completes; this path
does not run actual benchmark scoring. The main timer includes dispatch,
pilot-dependent posterior inference, allocation, and queue management, but
excludes loading, warmup, offline design, static quadrature, and reset.
Same seeds do not guarantee token-identical output under different schedules.

Run outputs include `experiment_manifest.json`, `summary_runs.csv`, and
`block=00/policy=.../` with `summary.json`, `allocation.json` where applicable,
`requests.parquet`, `events.parquet`, and `engine_samples.parquet`.
Artifact policy names are `uniform_flat`, `hbn_sync`, and
`hbn_spec_fill256_partial_plugin`.

## Generate measurements from new runs

From the repository root, supply the individual run directories printed by
`async-hbn run`:

```bash
uv run --project packages/systems --locked async-hbn measurements \
  --runs /path/to/run1 /path/to/run2 /path/to/run3 \
  --output results/recomputed/measurements.csv
```

The output has the same schema as `results/systems/measurements.csv`. It records
result-ready time and estimates useful and wasted FLOPs from request token counts
and the model coefficients in `results/systems/architectures.json`. Queued
cancellations contribute no compute; running aborts contribute their observed
partial token counts. The exporter validates identities, allocation, profile and
model fingerprints, request accounting, and GPU monitoring. It rejects retries,
incomplete runs, duplicate observations, and existing output files.

Once the measurements cover all 107 profiles, four budgets, and three strategies:

```bash
python tools/reproduce_tables.py \
  --measurements results/recomputed/measurements.csv \
  --equivalent-measurements results/recomputed/uniform_equivalent_measurements.csv \
  --output-dir results/recomputed/tables
```

Supply the complete 428-run equivalent measurement file exported above alongside
the 1,284-run original-budget grid. Both time columns in Table 4 then use the
new original-budget Uniform times as their denominator. Omit both measurement
options to reproduce the paper from the published, checksum-verified inputs.

Table 4 measurements are bound to the frozen HBN replay that defines their
target budgets. The optional `--replay` argument accepts only an identical copy
of that file. Different replay ratios require a new target grid and matching
Uniform measurements; `make reproduce` does not redefine the published targets.

## Implementation and tests

- `release.py` / `cli.py`: portable fixed-protocol configuration and single-run CLI.
- `suite.py` / `runner.py`: lifecycle, event loop, timing, and output.
- `controller.py` / `protocol.py`: admission and fixed-ID recovery.
- `allocation.py` / `partial_priority.py`: complete and partial pilot allocation.
- `pool.py` / `worker.py`: persistent single-GPU replicas and reset.
- `ledger.py` / `profile_rewards.py`: request accounting and frozen rewards.

Tests cover the 428 published profile/budget identity targets, allocation,
frozen rewards, partial-allocation admission, final selection, request accounting,
and measurement export. Logical IDs and seeds are independent of local asset
paths. The numerical allocation implementation is tested against
`packages/statistical/src/speculative_eval/core.py`.

Statistical code uses Python 3.12 and NumPy 1.26; systems use Python 3.11 and
NumPy 2. CPU tests do not require model weights or GPU dependencies.

```bash
make systems-setup systems-test  # CPU only, run from repository root
```
