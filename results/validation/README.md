# Validation targets

- `table1_targets.json`: high-precision statistical means and selected IBN alpha.
- `expected_identity.json`: allocation and accepted-ID/seed/reward digests for
  every published profile/budget.
- `core_source_sha256.json`: checksums for fixed numerical and worker components.
  Scheduling behavior is validated by identity and behavioral tests.
- `prompt_targets.json`: rendered-text and token-ID digests for all 107 profiles,
  checked with the six local model tokenizers during input preparation.
- `manifest.json`: published input checksums. File keys
  are relative to `results/`.
- `uniform_equivalent_measurements.json`: checksum and audited observation count
  for the 428 fresh rounded-equivalent Uniform measurements used in Table 4.
