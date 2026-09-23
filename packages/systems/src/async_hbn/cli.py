"""Public single-experiment interface; no distributed orchestration."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List supported benchmark/checkpoint profiles")
    export = commands.add_parser("measurements", help="Export paper measurements from single-run directories")
    export.add_argument("--runs", nargs="+", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--equivalent", action="store_true", help="Export freshly measured rounded-equivalent Uniform runs")
    for name in ("show", "check", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "measurements":
        from .measurements import export_measurements

        print(export_measurements(args.runs, args.output, equivalent=args.equivalent))
        return
    from .release import load_experiment, read_result

    if args.command == "list":
        print("\n".join(sorted(read_result("configurations/profiles.json"))))
        return
    config, policy, output = load_experiment(args.config)
    if args.command == "show":
        print(
            json.dumps(
                {
                    "config": config.raw,
                    "selected_policy": policy,
                    "output_dir": str(output),
                },
                indent=2,
            )
        )
        return
    from .config import verify_static_inputs

    verify_static_inputs(config)
    if args.command == "check":
        from .tasks import load_and_render_tasks
        from .profile_rewards import ProfileBernoulliRewards

        tasks = load_and_render_tasks(config)
        ProfileBernoulliRewards(config, task_ids=tasks)
        print(
            f"Validated {len(tasks)} tasks and rendered prompts; no GPU generation started."
        )
        return
    from .protocol import Policy
    from .suite import run_suite

    result = run_suite(
        config,
        output_root=output,
        blocks=1,
        policies=(Policy(policy),),
        reduced_smoke=False,
    )
    print(result)


if __name__ == "__main__":
    main()
