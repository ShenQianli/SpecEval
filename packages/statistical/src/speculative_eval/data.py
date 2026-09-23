"""Input schema and cryptographic audit for frozen pass-count profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Profile:
    path: Path
    pair_key: str
    benchmark: str
    model: str
    tested_k: int
    scorer_version: str
    sha256: str
    probabilities: np.ndarray

    @property
    def n_tasks(self) -> int:
        return int(self.probabilities.size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(benchmark: str, model: str) -> str:
    return f"{benchmark}__model={model}".replace("/", "_")


def audit_data(data_dir: Path) -> tuple[list[Profile], list[dict[str, object]]]:
    """Validate every artifact against manifest.json and return valid profiles."""

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get("artifact_count") != len(
        artifacts
    ):
        raise ValueError("manifest artifact count mismatch")

    by_file = {item["file"]: item for item in artifacts}
    files = sorted(data_dir.glob("benchmark=*.json"))
    if {path.name for path in files} != set(by_file):
        raise ValueError("manifest and local artifact filenames differ")

    profiles: list[Profile] = []
    statuses: list[dict[str, object]] = []
    seen_pairs: set[str] = set()
    for path in files:
        expected = by_file[path.name]
        payload = json.loads(path.read_text())
        problems: list[str] = []
        for field in ("benchmark_id", "model_alias", "tested_k", "task_count"):
            if payload.get(field) != expected.get(field):
                problems.append(f"manifest_{field}_mismatch")
        if payload.get("schema_version") != 1:
            problems.append("unsupported_schema")

        tested_k = payload.get("tested_k")
        counts = payload.get("pass_counts")
        task_count = payload.get("task_count")
        if not isinstance(tested_k, int) or tested_k < 1024:
            problems.append("tested_k_below_1024")
        if not isinstance(counts, list) or len(counts) != task_count:
            problems.append("invalid_pass_counts_length")
        elif any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > tested_k
            for value in counts
        ):
            problems.append("invalid_pass_count")

        digest = sha256_file(path)
        if digest != expected.get("sha256"):
            problems.append("sha256_mismatch")
        benchmark = payload.get("benchmark_id", "")
        model = payload.get("model_alias", "")
        key = pair_key(benchmark, model)
        if key in seen_pairs:
            problems.append("duplicate_pair")
        seen_pairs.add(key)

        probabilities = (
            np.asarray(counts, dtype=np.float64) / tested_k
            if isinstance(counts, list) and isinstance(tested_k, int)
            else np.asarray([], dtype=np.float64)
        )
        total_variance = float(np.sum(probabilities * (1.0 - probabilities)))
        status = "valid"
        reason = ""
        if problems:
            status = "invalid"
            reason = ";".join(problems)
        elif not np.isfinite(probabilities).all():
            status = "invalid"
            reason = "nonfinite_probability"
        elif total_variance <= 0.0:
            status = "excluded"
            reason = "zero_total_variance"

        statuses.append(
            {
                "file": path.name,
                "pair_key": key,
                "status": status,
                "reason": reason,
                "sha256": digest,
            }
        )
        if status == "valid":
            profiles.append(
                Profile(
                    path=path,
                    pair_key=key,
                    benchmark=benchmark,
                    model=model,
                    tested_k=tested_k,
                    scorer_version=payload.get("scorer_version", ""),
                    sha256=digest,
                    probabilities=probabilities,
                )
            )

    invalid = [row for row in statuses if row["status"] == "invalid"]
    if invalid:
        raise ValueError(f"data audit failed: {invalid[:3]}")
    profiles.sort(key=lambda item: item.pair_key)
    return profiles, statuses
