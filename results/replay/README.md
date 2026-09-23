# Published Table 1 replay results

The four CSV files are the final per-profile
results, with 107 profiles × four budgets each. Oracle is exact; EN, IBN and
HBN use the analytic continuation-variance replay described in the paper.
IBN contains the final selected alpha at each budget, not a new tuning run.

`python tools/reproduce_table1.py` averages variance ratios equally over the
107 profiles and converts them to percentages. `results/tables/table1.csv` matches
the paper's four method columns and one-decimal precision; alpha is omitted.
`table1_unrounded.csv` preserves unrounded percentage means.

This is reaggregation of published results, not execution of new pilot replay.
The full statistical pipeline writes the same table schema. Unrounded variance
ratios and selected tuning parameters are retained in `results/recomputed/summary.json`.
