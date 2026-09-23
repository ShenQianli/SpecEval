# Results

| Directory | Contents |
|---|---|
| `tables/` | Generated Table 1--4 CSV files, unrounded Table 1, and system-table JSON |
| `replay/` | Published per-profile Oracle, EN, IBN, and HBN statistical results |
| `design/` | Published candidate risks, selected EN/IBN/HBN schedules and design metadata |
| `systems/` | Published real-generation measurements and FLOP coefficients |
| `configurations/` | Fixed experiment templates and ex-ante designs |
| `validation/` | Numerical targets, identity digests, source digests, data checksums |
| `recomputed/` | New statistical design/replay results, system measurements, and tables |

Run `make paper-tables` from the repository root to create `tables/` without
generation or replay. Use `TABLES_OUTPUT=/path/to/output` for a separate output
directory. These are the four main-text tables. Published inputs are version-controlled; `tables/` and
`recomputed/` are generated outputs. `make reproduce` writes statistical
intermediates to `recomputed/` and Table 1 to `tables/`. Its CLI accepts
`--output` for intermediates and `--tables-output` for tables.
