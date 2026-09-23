# Statistical designs

Saved full-precision candidate risks and selected schedules for EN, IBN and HBN.
`metadata.json` specifies the design priors, grids and Monte Carlo precision.
Each schedule is indexed by task count and budget; IBN additionally uses alpha.
The selected pilot size is `m_star`, and the deterministic stage weight is
`weight_star`. IBN alpha selection occurs during fixed-profile replay, not here.

From the repository root:

- `make replay`: use these schedules and recompute statistical replay.
- `make reproduce`: recompute both designs and replay.
- `make verify`: check recomputed Table 1 and the published parameter identities.
- `make verify-design`: additionally compare recomputed candidates, schedules
  and scientific metadata with these files. Worker count is execution metadata
  and may differ; numerical values have absolute tolerance `1e-12`.

New designs are written to `results/recomputed/design/`; these inputs are not
overwritten. SHA-256 checksums are recorded in `results/validation/manifest.json`.
