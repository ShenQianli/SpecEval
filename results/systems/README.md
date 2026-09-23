# Published system measurements

- `measurements.csv`: 1,284 real-generation observations, one per
  profile × budget × strategy. No repeated runs are implied.
- `architectures.json`: checkpoint parameter counts and full-attention coefficients
  for the FLOP proxy, with model configuration hashes.
- `uniform_equivalent_budgets.csv`: 428 rounded-down Uniform target budgets,
  deterministically derived from the fixed HBN replay, with source and target fingerprints.
- `uniform_equivalent_measurements.csv`: 428 fresh Uniform timings at these
  budgets, audited against request-level outputs. Its checksum and row count
  are recorded in `../validation/uniform_equivalent_measurements.json`.

Experiment settings are in [configurations](../configurations/README.md).
Checksums and identity targets are in [validation](../validation/README.md).

New single-run artifacts can be converted to this measurement schema with
`async-hbn measurements`; see the [systems instructions](../../packages/systems/README.md#generate-measurements-from-new-runs).

## Measurement schema

| Column | Meaning |
|---|---|
| `profile`, `model`, `budget`, `strategy` | Unique observation key and checkpoint |
| `time_seconds` | Measured result-ready wall time, not model startup or warmup |
| `useful_eflop` | Estimated FLOPs of accepted requests, divided by 10^18 |
| `wasted_eflop` | Estimated FLOPs of completed-discarded and running-aborted work |
| `accepted_rollouts` | Exactly N × b |
| `extra_rollouts` | Completed-discarded plus running-aborted request count |

Uniform and HBN-sync have zero speculative waste. Aborted requests use observed
partial token counts, so waste can underestimate compute before the first output.
The proxy per request is
`2*core*(P+C) + 2*lm*C + 2*full_layers*attention_width*(P+C)*(P+C+1)`.
It is not a GPU hardware counter and omits memory traffic, padding, and detailed
GDN recurrent-state costs. Values here were computed from recorded request traces;
they are not synthetic timings or predictions.

## Aggregation

Table 2 uses ratios of summed FLOPs/counts within each budget. Table 3 uses
`sum(strategy_time)/sum(uniform_time)-1`. For normalized time, first compute
`strategy_time * uniform_useful_flops / strategy_useful_flops` per profile,
then aggregate over the same 107 profiles. Delta columns are differences in
percentage points, not relative percentage savings.

Table 4 uses fresh measured Uniform times at `floor(b/r)` rollouts per task,
where `r` is the fixed HBN variance ratio for that profile and reference budget.
Sum these times, divide by summed original-budget Uniform time, and subtract
one. HBN-async uses the same denominator. The rounded/continuous budget column
is `sum(N * floor(b/r)) / sum(N * b/r)` expressed as a percentage; it is not
an average of per-profile ratios.
Average continuous- and rounded-equivalent budgets divide their respective
total rollout counts by the total task count across profiles, not by 107.

Frozen rewards align sync/async allocation, not generated text or token lengths.
These tables support reaggregation, not a substitute for raw request-level audit.
Data retain the repository's derived-statistics licensing note; underlying models
and benchmarks retain their own terms. No prompts or generated text are included.

## Fresh rounded-equivalent Uniform measurements

The target table distinguishes `reference_budget` (the HBN budget) from
`budget` (Uniform's actual rollout count per task). It includes the continuous
equivalent budget, the fixed HBN variance ratio, and the theoretical ratio
of rounded-budget Uniform variance to HBN variance. All 428 settings require
new measurements, including unchanged-budget settings.

`async-hbn measurements --equivalent` exports the measurement fields above plus
`reference_budget`, `target_fingerprint`, `protocol_fingerprint`, and `run_uuid`.
The unique comparison key is `(profile, reference_budget)`. Absolute run paths,
host identifiers, prompts, and generated text are not included in the CSV.

The table generator uses the published equivalent measurements by default;
`--equivalent-measurements` supplies a replacement measured grid.
It requires all 428 settings and validates their targets, identities,
counts, positive times and unique run UUIDs. Both time increases retain summed
original Uniform time as the denominator. Tables 2 and 3 remain unchanged.
See the [single-run instructions](../../packages/systems/README.md#uniform-at-rounded-down-equivalent-budgets).
