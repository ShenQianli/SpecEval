"""Convert validated single-run artifacts to paper measurement rows."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from .config import canonical_json
from .identities import Phase, RequestKey, request_seed
from .release import STRATEGIES, read_result, results_dir

FIELDS = (
    "profile", "model", "budget", "strategy", "time_seconds", "useful_eflop",
    "wasted_eflop", "accepted_rollouts", "extra_rollouts",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def flop_eflop(rows, arch, *, aborted=False):
    columns = ("prompt_tokens_before_abort", "generated_tokens_before_abort") if aborted else (
        "prompt_tokens", "completion_tokens"
    )
    costs = []
    for p, c in rows[list(columns)].itertuples(index=False, name=None):
        require(all(pd.notna(x) and math.isfinite(x) and x >= 0 and int(x) == x
                    for x in (p, c)), "Invalid token counts")
        p, c = int(p), int(c)
        length = p + c
        costs.append(2 * arch["core"] * length + 2 * arch["lm"] * c
                     + 2 * arch["full_layers"] * arch["attention_width"] * length * (length + 1))
    return math.fsum(costs) / 1e18


def measurement(root, *, equivalent=False):
    root = Path(root)
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    config = manifest["effective_config"]
    model, benchmark = config["model"]["alias"], config["benchmark"]["id"]
    profile = f"{benchmark}__model={model}"
    budget = config["design"]["budget_per_task"]
    entry = read_result("configurations/profiles.json")[profile]
    supplement = config.get("equivalent_uniform")
    require(bool(supplement) == equivalent, "Use the matching standard/equivalent measurement exporter")
    reference = supplement["reference_budget"] if equivalent else budget
    design = entry["budgets"][str(reference)]
    if equivalent:
        from .equivalent import load_targets, derived_design
        target = load_targets(results_dir())[profile, reference]
        derived, fingerprint, metadata = derived_design(entry, target)
        require(supplement == metadata, "Equivalent target metadata mismatch")
        design = dict(design=derived, protocol_fingerprint=fingerprint)
    expected = read_result("validation/expected_identity.json")[f"{profile}__b={reference:03d}"]
    arch = read_result("systems/architectures.json")[model]
    require(not manifest["reduced_smoke"] and manifest["requested_blocks"] == 1,
            "Expected one full-size run")
    policies = manifest["selected_policies"]
    require(len(policies) == 1 and policies[0] in STRATEGIES.values(), "Expected one paper strategy")
    policy = policies[0]
    strategy = next(s for s, p in STRATEGIES.items() if p == policy)
    require(not equivalent or strategy == "uniform", "Equivalent runs must use Uniform")
    require(config["design"] == design["design"], "Design mismatch")
    for field in ("sampling", "master_seed"):
        require(config[field] == entry["template"][field], f"{field} mismatch")
    require(manifest["protocol_fingerprint"] == design["protocol_fingerprint"], "Protocol mismatch")
    require(config["benchmark"]["sha256"] == entry["template"]["benchmark"]["sha256"],
            "Benchmark mismatch")
    from .tasks import benchmark_catalog
    input_sha256 = benchmark_catalog()[benchmark]["sha256"]
    require(config["benchmark"].get("input_sha256") == input_sha256, "Benchmark input mismatch")
    require(manifest.get("dataset", {}).get("sha256") == input_sha256, "Measured dataset mismatch")
    configs = {f["name"]: f["sha256"] for f in manifest["model"]["files"] if "sha256" in f}
    require(configs.get("config.json") == arch["config_sha256"], "Model architecture mismatch")
    guard = json.loads((root / "gpu_process_guard.json").read_text())
    require(guard["contamination"] is None and guard["monitor_error"] is None,
            "GPU monitoring did not validate this run")
    path = root / "block=00" / f"policy={policy}"
    summary = json.loads((path / "summary.json").read_text())
    require(summary["policy"] == policy and summary["block_index"] == 0, "Run identity mismatch")
    require(summary["protocol_fingerprint"] == manifest["protocol_fingerprint"], "Protocol mismatch")
    require(summary["reward_profile_sha256"] == entry["template"]["reward"]["profile_sha256"],
            "Profile mismatch")
    require(summary["reward_fingerprint"] == expected["expected_reward_fingerprint"], "Reward mismatch")
    if equivalent:
        # Compare the full serving configuration, while allowing device-index remapping.
        from .config import config_from_mapping
        from .profile_rewards import ProfileBernoulliRewards, load_frozen_profile
        parsed = config_from_mapping(config, source=root / "measurement.yaml")
        template = entry["template"]
        for k, v in template["engine"].items():
            if k != "gpu_ids":
                require(config["engine"][k] == v, f"Serving configuration mismatch: {k}")
        require(len(config["engine"]["gpu_ids"]) == 8, "Expected eight GPU replicas")
        for k in ("family", "alias", "chat_template_kwargs", "thinking_budget_tokens", "tensor_parallel_size"):
            require(config["model"][k] == template["model"][k], f"Model configuration mismatch: {k}")
        frozen = load_frozen_profile(parsed)
        sampler = ProfileBernoulliRewards(parsed, task_ids=frozen.task_ids)
        require(bool(summary.get("run_uuid")), "Equivalent run UUID missing")
    require(summary["retry_attempts"] == 0, "Retry compute needs separate accounting")
    requests = pd.read_parquet(path / "requests.parquet")
    require(not requests.logical_request_id.duplicated().any(), "Duplicate request identity")
    require(set(requests.state) <= {"accepted", "completed_discarded", "aborted_running", "cancelled_queued"},
            "Incomplete request states")
    useful = requests[requests.state == "accepted"]
    discarded = requests[requests.state == "completed_discarded"]
    aborted = requests[requests.state == "aborted_running"]
    count = config["design"]["n_tasks"] * budget
    require(len(useful) == summary["accepted_logical"] == count, "Accepted count mismatch")
    extra = len(discarded) + len(aborted)
    require(count + extra == summary["actual_rollouts"], "Actual count mismatch")
    if strategy != "uniform":
        allocation = json.loads((path / "allocation.json").read_text())["allocation"]
        require(digest(allocation) == expected["expected_allocation_sha256"], "Allocation mismatch")
        identities = sorted((r.logical_request_id, int(r.seed), int(r.reward))
                            for r in useful.itertuples())
        require(digest(identities) == expected["expected_accepted_sha256"], "Accepted identity mismatch")
    else:
        require(useful.groupby("task_id").size().eq(budget).all()
                and useful.task_id.nunique() == config["design"]["n_tasks"], "Uniform allocation mismatch")
        m = config["design"]["pilot_per_task"]
        identities = {
            RequestKey(task, phase, i).logical_id(manifest["protocol_fingerprint"])
            for task in useful.task_id.unique()
            for phase, count_per_task in ((Phase.PILOT, m), (Phase.CONTINUATION, budget - m))
            for i in range(1, count_per_task + 1)
        }
        require(set(useful.logical_request_id) == identities, "Uniform request identity mismatch")
        require(all(r.seed == request_seed(config["master_seed"], r.logical_request_id)
                    for r in useful.itertuples()), "Uniform seed mismatch")
        if equivalent:
            require(set(useful.task_id) == set(frozen.task_ids), "Uniform task universe mismatch")
            rewards = {
                key.logical_id(manifest["protocol_fingerprint"]): sampler.reward(key)
                for task in frozen.task_ids
                for phase, size in ((Phase.PILOT, m), (Phase.CONTINUATION, budget - m))
                for i in range(1, size + 1)
                for key in [RequestKey(task, phase, i)]
            }
            require(all(r.reward == rewards[r.logical_request_id] for r in useful.itertuples()),
                    "Uniform per-request reward mismatch")
    require(strategy == "hbn-async" or extra == 0, "Non-speculative strategy has waste")
    elapsed = float(summary["result_seconds"])
    require(math.isfinite(elapsed) and elapsed > 0, "Invalid elapsed time")
    useful_cost = flop_eflop(useful, arch)
    require(useful_cost > 0, "Useful FLOPs must be positive")
    row = dict(profile=profile, model=model, budget=budget, strategy=strategy,
                time_seconds=elapsed, useful_eflop=useful_cost,
                wasted_eflop=flop_eflop(discarded, arch) + flop_eflop(aborted, arch, aborted=True),
                accepted_rollouts=count, extra_rollouts=extra)
    if equivalent:
        row.update(reference_budget=reference, target_fingerprint=target["target_fingerprint"],
                   protocol_fingerprint=manifest["protocol_fingerprint"], run_uuid=summary["run_uuid"])
    return row


def export_measurements(run_dirs, output, *, equivalent=False):
    rows = [measurement(root, equivalent=equivalent) for root in run_dirs]
    keys = [(r["profile"], r.get("reference_budget", r["budget"]), r["strategy"]) for r in rows]
    require(len(set(keys)) == len(keys), "Duplicate profile/budget/strategy observations")
    require(bool(rows), "No measurements supplied")
    if equivalent:
        require(len({r["run_uuid"] for r in rows}) == len(rows), "Duplicate equivalent run UUID")
    rows.sort(key=lambda r: (r["profile"], r["budget"], r["strategy"]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if equivalent else FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output
