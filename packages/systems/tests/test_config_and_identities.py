from __future__ import annotations

from pathlib import Path

from async_hbn.config import POLICY_NAMES, load_config
from async_hbn.identities import Phase, RequestKey, physical_request_id, request_seed


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests/fixtures/single_run.yaml"


def test_frozen_config_has_expected_budget_and_design() -> None:
    config = load_config(CONFIG)
    assert config.design.n_tasks == 30
    assert config.design.budget_per_task == 32
    assert config.design.pilot_per_task == 10
    assert config.design.pilot_total == 300
    assert config.design.continuation_total == 660
    assert config.design.accepted_total == 960
    assert config.design.stage_weight == 0.28592995791015385
    assert config.execution.policies == POLICY_NAMES
    assert config.engine.batch_invariant is False
    assert config.engine.reproducibility_mode == "seeded_non_batch_invariant"
    assert config.engine.max_inflight_per_engine == 32


def test_seed_depends_only_on_logical_identity() -> None:
    key = RequestKey("60", Phase.CONTINUATION, 1)
    logical = key.logical_id("protocol")
    assert request_seed(42, logical) == request_seed(42, logical)
    assert request_seed(42, logical) != request_seed(43, logical)
    first = physical_request_id("run-a", logical, 0)
    retry = physical_request_id("run-a", logical, 1)
    other_run = physical_request_id("run-b", logical, 0)
    assert len({first, retry, other_run}) == 3
    # Retry/run namespace changes the physical ID but never the sampling seed.
    assert request_seed(42, logical) == request_seed(42, logical)
