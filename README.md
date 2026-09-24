# Speculative Evaluation

This release starts from fixed benchmark tasks/prompts and fixed task-probability
profiles. It provides statistical replay, single-run real-generation system
experiments, and CPU-only reproduction of main-text Tables 1--4 from published results.

Benchmark inputs are downloaded and reconstructed locally and are not included
in this repository. Their use and redistribution remain subject to the
[source and access information](data/SOURCES.md).

## Reproduce the paper tables

To reaggregate all four paper tables from published data, without GPU generation
or statistical replay, run `make paper-tables`. Outputs are `results/tables/table1.csv`
through `table4.csv`. Table 1 uses the paper's one-decimal percentages and also
writes `table1_unrounded.csv`. See [replay data](results/replay/README.md) and
[system measurements](results/systems/README.md) for the input schemas.

No dependencies, GPUs, checkpoints, or experiment-run artifacts are
needed to aggregate the published measurements (Python 3.11+):

```bash
make paper-tables
```

See [packages/systems/README.md](packages/systems/README.md) to configure and execute one
Uniform, HBN-sync, or HBN-async experiment, export request-level measurements,
and generate tables from new runs. The system and statistical packages
use separate locked Python environments; do not merge their NumPy versions.

The system package also supports [fresh Uniform measurements at rounded-down
equivalent budgets](packages/systems/README.md#uniform-at-rounded-down-equivalent-budgets).
`make systems-equivalent-targets` reproduces the 428 target budgets from fixed
HBN replay. `make systems-equivalent-tables EQUIVALENT_MEASUREMENTS=...` builds
the measured-time Table 4 comparison from a complete new measurement file,
without GPU generation.
The default `make paper-tables` includes the 428 published rounded-equivalent
Uniform measurements, task-count-weighted average continuous and rounded budgets,
and the rounded/continuous rollout-budget ratio. To write tables elsewhere, use
`make paper-tables TABLES_OUTPUT=/path/to/output`.

The statistical pipeline starts from frozen, scored rollout counts and evaluates Oracle, tuned
Empirical Neyman (EN), tuned Independent Bayesian Neyman (IBN), and
Hierarchical Bayesian Neyman (HBN). Profiles are included in `data/profiles/`;
benchmark tasks/prompts are rebuilt with `make benchmarks`;
see [input definitions and provenance](data/README.md).

## Reported table

All entries are mean Uniform-normalized variance across 107
benchmark--checkpoint profiles (Uniform = 100%). Lower is better.

| Budget b | Oracle | Tuned EN | Tuned IBN | HBN |
|---:|---:|---:|---:|---:|
| 8  | 40.7% | 94.3% | 89.6% | **87.2%** |
| 16 | 39.9% | 92.6% | 81.9% | **80.1%** |
| 32 | 39.5% | 91.9% | 74.2% | **73.0%** |
| 64 | 39.3% | 91.5% | 67.0% | **66.4%** |

The methods use the following tuning protocols:

- Tuned EN exhausts every feasible integer pilot size on the 107 evaluation
  profiles and uses the analytic risk-minimizing constant stage weight. It is
  a post-hoc empirical sensitivity result, not an ex-ante deployable policy.
- For every positive alpha on the 23-point grid, IBN jointly designs its pilot
  and constant weight under the matched iid Beta(alpha, alpha) prior. Alpha is
  then selected post hoc by mean risk on the 107 evaluation profiles.
- HBN jointly selects its pilot and constant weight before evaluation under
  the hierarchical reference prior. No evaluation profile enters its design.
- Oracle uses the exact fixed-profile integer Neyman allocation and is an
  unattainable full-information reference.

## Recompute statistical designs and replay

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
make setup
make audit
make test
make reproduce
make verify
make verify-design
```

`make reproduce` starts from the frozen profiles and recomputes all designs
and replays at the published Monte Carlo precision. The design stage uses:

- EN: 8192 draws per profile and every `m=1,...,b-1`;
- IBN: 8 independent replicates of 8192 matched-prior draws for every positive
  alpha and every `m=1,...,b-1`;
- HBN: 8 independent replicates of 8192 reference-prior draws and every
  feasible pilot size `m=1,...,b-1` (through 63 when `b=64`);
- final fixed-profile replay: 8192 draws per profile and candidate.

The run writes only inputs needed to audit the table:

```text
results/recomputed/data_audit.csv
results/recomputed/design/{en,ibn,hbn}_{candidates,schedule}.csv
results/recomputed/design/metadata.json
results/recomputed/{oracle,en,hbn}_pair_results.csv
results/recomputed/ibn_pair_grid.csv
results/recomputed/ibn_grid_summary.csv
results/recomputed/ibn_tuned_pair_results.csv
results/tables/table1.csv
results/tables/table1_unrounded.csv
results/recomputed/summary.json
```

`make verify` compares all table entries and selected IBN alpha values against
`results/validation/table1_targets.json`. Exact Oracle values use a strict numerical
tolerance; Monte Carlo policies must match the stored numerical acceptance
tolerances and the paper's displayed percentages.

The full-precision design inputs are also included in
[results/design](results/design/README.md). Run `make replay WORKERS=4` to
recompute replay without redoing design, then `make verify`. This verifies
Table 1 and checks that published replay/system parameters match the saved
schedules. After a full `make reproduce`, `make verify-design` additionally
compares all recomputed candidate risks, schedules and scientific metadata.

## Fixed experiment inputs

Rebuild the 18 benchmark JSONL files (1,418 tasks/prompts) with one command:

```bash
make benchmarks
```

This uses a separate locked script environment; no GPU or statistical package
installation is needed. The script guides you through source terms and missing
Hugging Face access, downloads pinned revisions, and verifies every generated
file against the frozen SHA-256 and all corresponding profiles. It never
accepts upstream agreements on your behalf. Repeating the command resumes
downloads and skips verified outputs. GPQA/HLE require your own access approval.
See [data preparation](data/README.md) for offline and noninteractive options.
Generated question text is not part of the source distribution or wheel.

`data/profiles/` contains 108 benchmark--checkpoint pass-count profiles,
with 107 nonzero-variance profiles used for evaluation. Task IDs and order match
across these inputs; no benchmark test cases or scoring service is required.

```bash
python tools/check_inputs.py
```

The reproduction boundary starts at these hash-validated inputs. Their
construction and pinned upstream sources are specified in `data/README.md`.
Profiles remain fixed inputs, not regenerated rollout counts.
System rewards are deterministic Bernoulli draws from the fixed
profiles, revealed after real generation completes. Model checkpoints remain
external inputs.

## Repository layout

```text
packages/
  statistical/     Python 3.12 package, lockfile, and unit tests
  systems/         Python 3.11 package, lockfile, and unit tests
configs/systems/   Single-run configuration examples
data/              Shared profiles and benchmark input definitions
results/           Published designs, replay, measurements, and validation
tools/             Input preparation and table reproduction
tests/             Repository integration and cross-package tests
```

The two packages have independent environments. Run repository-level commands
from this directory; `make test` and `make systems-test` each include the
corresponding integration tests.

## Code map

```text
data/profiles/manifest.json + data/profiles/benchmark=*.json
  -> data.py          schema, SHA-256 audit, zero-variance exclusion
  -> core.py          posterior scores and exact integer allocation
  -> design.py        joint EN, IBN, and HBN design
  -> experiments.py   independent fixed-profile replay and Table 1 aggregation
  -> cli.py           audit, reproduce, and verify commands

data/benchmarks/*.jsonl + data/profiles/*.json
  -> packages/systems/src/async_hbn/tasks.py       fixed input validation and prompt rendering
  -> packages/systems/src/async_hbn/runner.py      real generation with frozen rewards
  -> packages/systems/src/async_hbn/measurements.py  time and FLOP measurements
  -> tools/reproduce_tables.py   Tables 2--4
```

Candidate pilot sizes share nested random paths within each design or replay.
The design and replay seed namespaces are distinct. All 107 profiles are
equally weighted, and each fixed probability is `pass_count / 1024`.

## Scope

Included are fixed profiles, benchmark reconstruction and validation, all three design
procedures, exact Oracle, fixed-profile replay, single-run system experiments,
and all four main-text tables. Reaggregation of published results must reproduce the
paper tables; Monte Carlo replay uses the documented numerical tolerances.
Fresh system runs follow the fixed protocol but need not produce identical text
or timings. The raw 1024-response-per-task corpus and model weights are not
included.

## License and data note

Code is released under the MIT License. Benchmark task text and derived
numerical profiles retain their source terms; the code license does not
relicense third-party benchmark content. Benchmark sources are listed in
`data/benchmarks/manifest.json`.
