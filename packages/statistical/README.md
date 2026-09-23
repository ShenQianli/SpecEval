# Statistical evaluation

The `speculative-evaluation` package provides fixed-profile design, replay, and
verification for Speculative Evaluation. It requires Python 3.12 and uses its own
locked environment. The package version is 0.0.1; its import name is
`speculative_eval` and its command is `spec-eval`.

From the repository root:

```bash
make setup
make audit
make test
make reproduce
make verify
```

`make replay` uses the published designs without recomputing them.
Commands resolve data and results relative to the current working directory;
`spec-eval --repo-root /path/to/spec_eval ...` selects a different repository root.
See the [repository instructions](../../README.md) for statistical protocols,
input preparation, output schemas, and numerical acceptance tolerances.
