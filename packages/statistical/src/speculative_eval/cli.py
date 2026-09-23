"""Command-line interface for the Table 1 reproduction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import audit_data
from .design import MAX_FEASIBLE_PILOT, reproduce_designs, write_csv
from .experiments import load_schedules, reproduce_table
from .verification import verify_designs, verify_published_parameters


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path.cwd().resolve()


def _audit(root: Path) -> tuple[list, list[dict[str, object]]]:
    profiles, statuses = audit_data(root / "data" / "profiles")
    counts: dict[str, int] = {}
    for row in statuses:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    print(
        f"data audit: files={len(statuses)} valid={len(profiles)} "
        + " ".join(
            f"{key}={value}"
            for key, value in sorted(counts.items())
            if key != "valid"
        )
    )
    return profiles, statuses


def command_audit(args: argparse.Namespace) -> None:
    _audit(_root(args.repo_root))


def command_replay(args: argparse.Namespace) -> None:
    root = _root(args.repo_root)
    design = Path(args.design_dir).resolve() if args.design_dir else root / "results/design"
    if not args.design_dir:
        verify_published_parameters(root)
    output = Path(args.output).resolve() if args.output else root / "results/recomputed"
    profiles, statuses = _audit(root)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "data_audit.csv", statuses)
    reproduce_table(
        profiles, load_schedules(design), output,
        draws=args.replay_draws, workers=args.workers,
        tables_output=Path(args.tables_output).resolve() if args.tables_output else root / "results/tables",
    )
    print(f"wrote replay results to {output}")


def command_verify_design(args: argparse.Namespace) -> None:
    root = _root(args.repo_root)
    verify_published_parameters(root)
    if args.design_dir:
        verify_designs(root / "results/design", Path(args.design_dir).resolve())
    print("design verification passed")


def command_reproduce(args: argparse.Namespace) -> None:
    root = _root(args.repo_root)
    output = Path(args.output).resolve() if args.output else root / "results/recomputed"
    profiles, statuses = _audit(root)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "data_audit.csv", statuses)
    schedules = reproduce_designs(
        profiles,
        output / "design",
        workers=args.workers,
        en_draws=args.en_design_draws,
        hbn_draws=args.hbn_design_draws,
        hbn_replicates=args.hbn_design_replicates,
        ibn_draws=args.ibn_design_draws,
        ibn_replicates=args.ibn_design_replicates,
        en_max_pilot=args.en_max_pilot,
        hbn_max_pilot=args.hbn_max_pilot,
        ibn_max_pilot=args.ibn_max_pilot,
    )
    print("replaying Oracle, tuned EN, tuned IBN, and HBN")
    table = reproduce_table(
        profiles,
        schedules,
        output,
        draws=args.replay_draws,
        workers=args.workers,
        tables_output=Path(args.tables_output).resolve() if args.tables_output else root / "results/tables",
    )
    print("Table 1")
    for row in table:
        print(
            f"b={int(row['b']):>2} "
            f"Oracle={100 * float(row['oracle']):.1f}% "
            f"Tuned EN={100 * float(row['tuned_en']):.1f}% "
            f"Tuned IBN={100 * float(row['tuned_ibn']):.1f}% "
            f"HBN={100 * float(row['hbn']):.1f}%"
        )
    print(f"wrote Table 1 reproduction to {output}")


def _assert_close(
    label: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{label}: observed {observed:.16g}, expected {expected:.16g}, "
            f"tolerance {tolerance:g}"
        )


def command_verify(args: argparse.Namespace) -> None:
    root = _root(args.repo_root)
    results = Path(args.results).resolve() if args.results else root / "results/recomputed"
    targets = json.loads((root / "results/validation/table1_targets.json").read_text())
    observed = json.loads((results / "summary.json").read_text())
    expected_rows = {int(row["b"]): row for row in targets["table1"]}
    observed_rows = {int(row["b"]): row for row in observed["table1"]}
    if set(observed_rows) != set(expected_rows):
        raise AssertionError("Table 1 budget grid mismatch")
    expected_fields = {"b", "oracle", "tuned_en", "tuned_ibn", "tuned_alpha", "hbn"}
    for budget, expected in expected_rows.items():
        row = observed_rows[budget]
        if set(row) != expected_fields:
            raise AssertionError(f"unexpected Table 1 fields at b={budget}: {set(row)}")
        _assert_close(
            f"b={budget} Oracle",
            float(row["oracle"]),
            float(expected["oracle"]),
            args.oracle_tolerance,
        )
        _assert_close(
            f"b={budget} tuned alpha",
            float(row["tuned_alpha"]),
            float(expected["tuned_alpha"]),
            1e-12,
        )
        for field in ("tuned_en", "tuned_ibn", "hbn"):
            _assert_close(
                f"b={budget} {field}",
                float(row[field]),
                float(expected[field]),
                args.mc_tolerance,
            )
            if f"{100 * float(row[field]):.1f}" != f"{100 * float(expected[field]):.1f}":
                raise AssertionError(
                    f"b={budget} {field} does not reproduce the published "
                    "one-decimal percentage"
                )
    print(
        "verification passed: all Table 1 entries and tuned alpha values "
        "match their acceptance targets"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec-eval",
        description="Reproduce Speculative Evaluation Table 1",
    )
    parser.add_argument(
        "--repo-root",
        help="repository root (defaults to the current working directory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-data", help="validate frozen inputs")
    audit.set_defaults(func=command_audit)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="design all methods and reproduce Table 1",
    )
    reproduce.add_argument("--output")
    reproduce.add_argument("--tables-output")
    reproduce.add_argument("--replay-draws", type=int, default=8192)
    reproduce.add_argument("--en-design-draws", type=int, default=8192)
    reproduce.add_argument("--hbn-design-draws", type=int, default=8192)
    reproduce.add_argument("--hbn-design-replicates", type=int, default=8)
    reproduce.add_argument("--ibn-design-draws", type=int, default=8192)
    reproduce.add_argument("--ibn-design-replicates", type=int, default=8)
    reproduce.add_argument(
        "--en-max-pilot", type=int, default=MAX_FEASIBLE_PILOT
    )
    reproduce.add_argument(
        "--hbn-max-pilot", type=int, default=MAX_FEASIBLE_PILOT
    )
    reproduce.add_argument(
        "--ibn-max-pilot", type=int, default=MAX_FEASIBLE_PILOT
    )
    reproduce.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    reproduce.set_defaults(func=command_reproduce)

    replay = subparsers.add_parser("replay", help="replay using saved designs")
    replay.add_argument("--design-dir")
    replay.add_argument("--output")
    replay.add_argument("--tables-output")
    replay.add_argument("--replay-draws", type=int, default=8192)
    replay.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    replay.set_defaults(func=command_replay)

    design = subparsers.add_parser("verify-design", help="validate published parameters and optionally recomputed designs")
    design.add_argument("--design-dir")
    design.set_defaults(func=command_verify_design)

    verify = subparsers.add_parser(
        "verify",
        help="compare Table 1 with the acceptance targets",
    )
    verify.add_argument("--results")
    verify.add_argument("--oracle-tolerance", type=float, default=5e-12)
    verify.add_argument("--mc-tolerance", type=float, default=2e-4)
    verify.set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
