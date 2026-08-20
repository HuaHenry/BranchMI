from __future__ import annotations

import argparse
import json
from pathlib import Path

from branchmi_pilot.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="branchmi-pilot",
        description="Run and analyze the BranchMI counterfactual pilot study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run/resume an experiment and analyze it")
    run_parser.add_argument("--config", "-c", required=True, help="YAML configuration path")
    run_parser.add_argument("--limit", type=int, help="Override dataset.limit")
    run_parser.add_argument("--run-name", help="Override run_name")
    run_parser.add_argument(
        "--no-resume", action="store_true", help="Fail if the output run already exists"
    )

    analyze_parser = subparsers.add_parser("analyze", help="Recompute analysis only")
    analyze_parser.add_argument("--run-dir", required=True, help="Existing run directory")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        from branchmi_pilot.experiment import PilotExperiment

        cfg = load_config(args.config)
        if args.limit is not None:
            cfg.dataset.limit = args.limit
        if args.run_name:
            cfg.run_name = args.run_name
        if args.no_resume:
            cfg.resume = False
        run_dir = PilotExperiment(cfg).run()
        print(f"Run complete: {run_dir}")
        print(f"Report: {run_dir / 'report.md'}")
        return

    from branchmi_pilot.analysis import analyze_run

    summary = analyze_run(Path(args.run_dir))
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
