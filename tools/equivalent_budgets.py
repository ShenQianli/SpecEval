"""Generate the 428 Uniform target budgets from fixed HBN replay (no generation)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/systems/src"))
from async_hbn.equivalent import write_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(write_targets(args.results_dir, args.output))


if __name__ == "__main__":
    main()
