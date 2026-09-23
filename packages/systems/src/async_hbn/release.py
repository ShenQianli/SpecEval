"""Portable selection of one published model/profile/budget/strategy."""

import copy
import json
from pathlib import Path

import yaml

from .config import ExperimentConfig, config_from_mapping, sha256_file
from .resources import resource_dir

STRATEGIES = {
    "uniform": "uniform_flat",
    "hbn-sync": "hbn_sync",
    "hbn-async": "hbn_spec_fill256_partial_plugin",
}


def results_dir():
    return resource_dir("results")


def read_result(name):
    root = results_dir()
    manifest = json.loads((root / "validation/manifest.json").read_text())
    if sha256_file(root / name) != manifest["files"][name]:
        raise ValueError(f"Reference fingerprint mismatch: {name}")
    return json.loads((root / name).read_text())


class PublishedConfig(ExperimentConfig):
    @property
    def protocol_fingerprint(self):
        return self.raw["_published_protocol_fingerprint"]


def load_experiment(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    required = {
        "model",
        "benchmark",
        "budget",
        "strategy",
        "model_path",
        "output_dir",
    }
    optional = {"benchmark_path", "profile_data_dir", "gpu_ids", "max_inflight_per_engine", "reference_budget"}
    if (
        not isinstance(spec, dict)
        or required - spec.keys()
        or spec.keys() - required - optional
    ):
        raise ValueError(
            f"Config requires {sorted(required)}; optional: {sorted(optional)}"
        )
    if spec["strategy"] not in STRATEGIES:
        raise ValueError(f"Strategy must be one of {tuple(STRATEGIES)}")
    if type(spec["budget"]) is not int or spec["budget"] < 1:
        raise ValueError("budget must be a positive integer")
    reference = spec.get("reference_budget", spec["budget"])
    if type(reference) is not int:
        raise ValueError("reference_budget must be an integer")
    if "reference_budget" in spec and spec["strategy"] != "uniform":
        raise ValueError("reference_budget is only supported for equivalent Uniform runs")
    key = f"{spec['benchmark']}__model={spec['model']}"
    catalog = read_result("configurations/profiles.json")
    if key not in catalog or str(reference) not in catalog[key]["budgets"]:
        raise ValueError("Unknown published profile or budget; use async-hbn list")
    entry = catalog[key]
    raw = copy.deepcopy(entry["template"])
    design = entry["budgets"][str(reference)]
    raw["design"] = design["design"]
    raw["_published_protocol_fingerprint"] = design["protocol_fingerprint"]
    raw["experiment_name"] = f"{key}__b={int(spec['budget']):03d}__{spec['strategy']}"
    if "reference_budget" in spec:
        from .equivalent import load_targets, derived_design
        target = load_targets(results_dir())[key, reference]
        if spec["budget"] != target["budget"]:
            raise ValueError(f"Equivalent Uniform budget must be {target['budget']}")
        raw["design"], raw["_published_protocol_fingerprint"], raw["_equivalent_uniform"] = derived_design(entry, target)
        raw["experiment_name"] += f"__equivalent_ref={reference:03d}"
    resolve = lambda p: (path.parent / p).resolve()
    raw["model"]["path"] = str(resolve(spec["model_path"]))
    from .tasks import benchmark_catalog, data_dir
    benchmark = benchmark_catalog()[spec["benchmark"]]
    if benchmark["source_artifact_sha256"] != raw["benchmark"]["sha256"]:
        raise ValueError("Benchmark protocol identity mismatch")
    raw["benchmark"]["path"] = str(resolve(spec["benchmark_path"]) if "benchmark_path" in spec
                                    else data_dir() / "benchmarks" / benchmark["file"])
    raw["benchmark"]["input_sha256"] = benchmark["sha256"]
    default_data = data_dir() / "profiles"
    data = (
        resolve(spec["profile_data_dir"])
        if "profile_data_dir" in spec
        else default_data
    )
    raw["reward"]["profile_path"] = str(data / raw["reward"]["profile_path"])
    if "gpu_ids" in spec:
        raw["engine"]["gpu_ids"] = spec["gpu_ids"]
    if "max_inflight_per_engine" in spec:
        raw["engine"]["max_inflight_per_engine"] = spec["max_inflight_per_engine"]
    config = config_from_mapping(raw, source=path)
    if sha256_file(config.reward.profile_path) != config.reward.profile_sha256:
        raise ValueError("Frozen profile fingerprint mismatch")
    return (
        PublishedConfig(**vars(config)),
        STRATEGIES[spec["strategy"]],
        resolve(spec["output_dir"]),
    )
