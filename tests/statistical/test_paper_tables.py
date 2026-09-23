"""Exercise the published table CLI without installed scientific packages."""

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_paper_table_scripts_standard_library_only(tmp_path):
    for script in ("tools/reproduce_table1.py", "tools/reproduce_tables.py"):
        subprocess.run(
            [sys.executable, "-I", "-S", str(ROOT / script),
             "--output-dir", str(tmp_path)],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )
    for number in (1, 2, 3, 4):
        with (tmp_path / f"table{number}.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert [int(r["budget"]) for r in rows] == [8, 16, 32, 64]
    columns = (
        "avg_continuous_equivalent_budget", "avg_rounded_equivalent_budget",
        "rounded_continuous_budget_pct", "equivalent_uniform_time_pct",
        "async_actual_time_pct", "delta_pp",
    )
    expected = [
        (9.37, 8.87, 94.63, 3.79, 11.15, -7.36),
        (20.88, 20.39, 97.66, 13.64, 2.85, 10.79),
        (47.04, 46.56, 98.98, 27.95, 5.49, 22.46),
        (107.22, 106.75, 99.56, 42.81, 6.56, 36.25),
    ]
    assert [tuple(round(float(r[c]), 2) for c in columns) for r in rows] == expected
