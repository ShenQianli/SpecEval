from __future__ import annotations

import json
from pathlib import Path

from speculative_eval.data import audit_data


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_data_manifest_and_hashes() -> None:
    profiles, statuses = audit_data(ROOT / "data" / "profiles")
    assert len(statuses) == 108
    assert len(profiles) == 107
    excluded = [row for row in statuses if row["status"] == "excluded"]
    assert excluded == [
        {
            "file": "benchmark=gpqa_diamond__domain_biology__model=qwen2_5_3b.json",
            "pair_key": "gpqa_diamond__domain_biology__model=qwen2_5_3b",
            "status": "excluded",
            "reason": "zero_total_variance",
            "sha256": "1ff13e505c446e70f1cb3ecac6473c02a1fa98f059b65e1292614050e8e4a883",
        }
    ]
    assert sorted({profile.n_tasks for profile in profiles}) == [19, 30, 86, 93, 100]


def test_table1_targets_have_only_current_columns() -> None:
    targets = json.loads((ROOT / "results/validation/table1_targets.json").read_text())
    rows = targets["table1"]
    assert [int(row["b"]) for row in rows] == [8, 16, 32, 64]
    assert all(
        set(row)
        == {"b", "oracle", "tuned_en", "tuned_ibn", "tuned_alpha", "hbn"}
        for row in rows
    )
