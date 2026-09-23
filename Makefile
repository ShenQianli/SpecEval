UV ?= uv
WORKERS ?= 4
BENCHMARK_ARGS ?=
EQUIVALENT_MEASUREMENTS ?= results/systems/uniform_equivalent_measurements.csv
TABLES_OUTPUT ?= results/tables

.PHONY: benchmarks
benchmarks:
	$(UV) run --script --locked tools/rebuild_benchmarks.py $(BENCHMARK_ARGS)

.PHONY: setup audit test reproduce verify

setup:
	$(UV) sync --project packages/statistical --extra test --locked

audit:
	$(UV) run --project packages/statistical --locked spec-eval audit-data

.PHONY: audit-inputs systems-input-test
audit-inputs:
	python tools/check_inputs.py

systems-input-test: audit-inputs
	$(UV) run --project packages/systems --extra test --locked python -m pytest -c packages/systems/pyproject.toml tests/systems/test_fixed_inputs.py

test:
	$(UV) run --project packages/statistical --extra test --locked python -m pytest -c packages/statistical/pyproject.toml packages/statistical/tests tests/statistical

reproduce:
	$(UV) run --project packages/statistical --locked spec-eval reproduce --workers $(WORKERS)

.PHONY: replay verify-design
replay:
	$(UV) run --project packages/statistical --locked spec-eval replay --workers $(WORKERS)

verify-design:
	$(UV) run --project packages/statistical --locked spec-eval verify-design --design-dir results/recomputed/design

verify:
	$(UV) run --project packages/statistical --locked spec-eval verify
	$(UV) run --project packages/statistical --locked spec-eval verify-design

.PHONY: systems-setup systems-test systems-tables
systems-setup:
	$(UV) sync --project packages/systems --extra test --locked

systems-test:
	$(UV) run --project packages/systems --extra test --locked python -m pytest -c packages/systems/pyproject.toml packages/systems/tests tests/systems

systems-tables:
	python tools/reproduce_tables.py --output-dir results/tables

.PHONY: systems-equivalent-targets systems-equivalent-tables
systems-equivalent-targets:
	python tools/equivalent_budgets.py --output results/recomputed/uniform_equivalent_budgets.csv

systems-equivalent-tables:
	python tools/reproduce_tables.py --equivalent-measurements $(EQUIVALENT_MEASUREMENTS) --output-dir results/recomputed/equivalent_tables

.PHONY: paper-tables
paper-tables:
	python tools/reproduce_table1.py --output-dir $(TABLES_OUTPUT)
	python tools/reproduce_tables.py --output-dir $(TABLES_OUTPUT)
