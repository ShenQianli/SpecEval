from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest

from async_hbn.config import RewardConfig, load_config, sha256_file
from async_hbn.messages import EventType
from async_hbn.protocol import Policy
from async_hbn.runner import ExperimentRunner, balanced_policy_orders
from async_hbn.suite import effective_config_manifest, smoke_config


ROOT = Path(__file__).resolve().parents[1]


class FakePool:
    def __init__(self, answers: dict[str, int]) -> None:
        self.answers = answers
        self.events: deque[dict] = deque()
        self.clock = time.monotonic_ns()

    def _time(self) -> int:
        self.clock += 1
        return self.clock

    def submit(self, engine_id: int, requests: list[dict]) -> None:
        for request in requests:
            base = {
                "engine_id": engine_id,
                "run_uuid": request["run_uuid"],
                "logical_request_id": request["logical_request_id"],
                "physical_request_id": request["physical_request_id"],
            }
            self.events.append({
                **base,
                "type": EventType.STARTED.value,
                "timestamp_ns": self._time(),
                "attempt": request["attempt"],
            })
            self.events.append({
                **base,
                "type": EventType.COMPLETED.value,
                "timestamp_ns": self._time(),
                "raw_output": rf"\boxed{{{self.answers[request['task_id']]}}}",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 2,
            })

    def abort(self, engine_id: int, physical_request_ids: list[str]) -> None:
        targets = set(physical_request_ids)
        retained = deque()
        metadata = {}
        while self.events:
            row = self.events.popleft()
            physical_id = row.get("physical_request_id")
            if physical_id in targets:
                metadata[physical_id] = row
            else:
                retained.append(row)
        self.events = retained
        for physical_id in targets:
            row = metadata[physical_id]
            self.events.append({
                "type": EventType.ABORTED.value,
                "engine_id": engine_id,
                "timestamp_ns": self._time(),
                "run_uuid": row["run_uuid"],
                "logical_request_id": row["logical_request_id"],
                "physical_request_id": physical_id,
                "generated_tokens_before_abort": 0,
            })

    def next_event(self, timeout: float = 1.0) -> dict:
        assert self.events, "fake router stalled"
        return self.events.popleft()


def _tasks() -> dict[str, dict]:
    return {
        str(task_id): {
            "task_id": str(task_id),
            "answer_json": 0,
            "scorer": "aime",
            "rendered_prompt": f"problem-{task_id}",
        }
        for task_id in range(60, 90)
    }


def test_all_policies_finish_with_fixed_accepted_budget() -> None:
    config = load_config(ROOT / "tests/fixtures/single_run.yaml")
    tasks = _tasks()
    outputs = {}
    for policy in Policy:
        runner = ExperimentRunner(
            config,
            tasks,
            FakePool({task_id: 0 for task_id in tasks}),
        )
        outputs[policy] = runner.run_policy(
            policy, run_uuid=f"fake-{policy.value}-{uuid.uuid4().hex}", block_index=0
        )
        assert outputs[policy].summary["accepted_logical"] == 960
        assert outputs[policy].summary["planned_valid"] == 960
        assert outputs[policy].summary["pilot_ready_seconds"] >= 0

    partial = outputs[Policy.HBN_ASYNC].summary
    assert partial["actual_rollouts"] >= partial["valid_rollouts"] == 960
    assert partial["speculative_admission"] == "partial_allocation_gated"
    assert sum(outputs[Policy.HBN_SYNC].allocation["allocation"].values()) == 660


def test_runner_uses_profile_rewards_instead_of_output_scoring(tmp_path: Path) -> None:
    config = load_config(ROOT / "tests/fixtures/single_run.yaml")
    tasks = _tasks()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "schema_version": 1,
        "benchmark_id": "aime24",
        "model_alias": "qwen3_5_4b",
        "tested_k": 1024,
        "task_count": 30,
        "task_ids": list(tasks),
        "pass_counts": [0] * 30,
    }))
    config = replace(config, reward=RewardConfig(
        mode="profile_bernoulli",
        profile_path=profile_path,
        profile_sha256=sha256_file(profile_path),
        tested_k=1024,
        seed=42,
    ))
    # FakePool emits answers that would score as correct under the AIME task
    # metadata.  The accepted rewards must nevertheless come from the profile.
    runner = ExperimentRunner(
        config,
        tasks,
        FakePool({task_id: 0 for task_id in tasks}),
    )
    output = runner.run_policy(
        Policy.UNIFORM_FLAT,
        run_uuid=f"fake-profile-{uuid.uuid4().hex}",
        block_index=0,
    )
    assert output.summary["reward_mode"] == "profile_bernoulli"
    assert output.summary["reward_fingerprint"]
    assert {row["reward"] for row in output.requests if row["state"] == "accepted"} == {0}


def test_cyclic_blocks_balance_every_policy_position() -> None:
    policy_count = len(tuple(Policy))
    orders = balanced_policy_orders(
        tuple(Policy), blocks=policy_count, seed=20260831
    )
    assert len(set(orders)) == policy_count
    for position in range(policy_count):
        assert {order[position] for order in orders} == set(Policy)


def test_five_three_policy_blocks_have_at_most_one_position_imbalance() -> None:
    selected = (
        Policy.UNIFORM_FLAT,
        Policy.HBN_SYNC,
        Policy.HBN_ASYNC,
    )
    orders = balanced_policy_orders(selected, blocks=5, seed=20260831)
    assert all(set(order) == set(selected) for order in orders)
    for policy in selected:
        counts = [
            sum(order[position] is policy for order in orders)
            for position in range(len(selected))
        ]
        assert max(counts) - min(counts) <= 1


def test_reduced_smoke_preserves_a_small_two_stage_design() -> None:
    config = load_config(ROOT / "tests/fixtures/single_run.yaml")
    smoke = smoke_config(config)
    assert smoke.design.n_tasks == 8
    assert smoke.design.pilot_per_task == 2
    assert smoke.design.uniform_continuations_per_task == 2
    assert smoke.design.speculative_depth == 4
    assert smoke.sampling.max_new_tokens == 64
    manifest = effective_config_manifest(smoke)
    assert manifest["design"]["budget_per_task"] == 4
    assert manifest["sampling"]["max_new_tokens"] == 64


def test_runner_records_event_level_engine_samples() -> None:
    config = load_config(ROOT / "tests/fixtures/single_run.yaml")
    output = ExperimentRunner(
        config, _tasks(), FakePool({str(task_id): 0 for task_id in range(60, 90)})
    ).run_policy(
        Policy.UNIFORM_FLAT,
        run_uuid=f"fake-samples-{uuid.uuid4().hex}",
        block_index=0,
    )
    assert output.engine_samples
    assert {row["event_type"] for row in output.engine_samples} >= {
        "router_submit",
        EventType.STARTED.value,
        EventType.COMPLETED.value,
    }
    assert sum(
        row["completion_tokens_delta"] for row in output.engine_samples
    ) == 960 * 2
    assert all(row["active_requests"] >= 0 for row in output.engine_samples)
