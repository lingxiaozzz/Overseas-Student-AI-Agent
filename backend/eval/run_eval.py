"""Run the evaluation suites as one reproducible benchmark.

The route and task suites intentionally remain independent: one verifies
decision quality and the other verifies end-to-end agent behaviour.  This
entry point gives both suites the same run identifier and produces one small,
stable summary that can be compared across implementation changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Expected evaluation report was not created: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_suite(script_name: str, arguments: list[str]) -> None:
    command = [sys.executable, str(EVAL_DIR / script_name), *arguments]
    print(f"\n==> {' '.join(command)}")
    subprocess.run(command, cwd=EVAL_DIR.parent, check=True)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _print_summary(summary: dict[str, Any]) -> None:
    route = summary.get("route")
    task = summary.get("task")
    print("\n================ Agent Evaluation ================")
    print(f"Run ID                     {summary['run_id']}")
    print(f"Route strict accuracy      {_percent(route and route['strict_accuracy'])}")
    print(f"Route lenient accuracy     {_percent(route and route['lenient_accuracy'])}")
    print(f"Task success rate          {_percent(task and task['success_rate'])}")
    print(f"Safety correctness         {_percent(route and route['safety_correctness'])}")
    print(f"Avg steps                  {task['avg_steps']:.2f}" if task else "Avg steps                  n/a")
    print(f"Avg tool calls             {task['avg_tool_calls']:.2f}" if task else "Avg tool calls             n/a")
    print(f"Replan rate                {_percent(task and task['replan_rate'])}")
    print("===================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run route and task evaluation as one benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="eval/reports", help="Report root relative to backend/.")
    parser.add_argument("--label", default="", help="Optional immutable label, e.g. baseline or reranker-v1.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request route-suite timeout in seconds.")
    parser.add_argument("--route-cases-file", default="", help="Optional versioned route dataset JSON.")
    parser.add_argument("--task-cases-file", default="", help="Optional versioned task dataset JSON.")
    parser.add_argument("--skip-route", action="store_true")
    parser.add_argument("--skip-task", action="store_true")
    args = parser.parse_args()

    if args.skip_route and args.skip_task:
        raise SystemExit("At least one suite must be enabled.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    label = "".join(char if char.isalnum() or char in "-_" else "-" for char in args.label).strip("-")
    run_id = f"{timestamp}-{label}" if label else timestamp
    report_root = (EVAL_DIR.parent / args.output_dir).resolve()
    run_dir = report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    common_args = ["--base-url", args.base_url, "--output-dir", str(run_dir)]
    if not args.skip_route:
        route_dataset_args = ["--cases-file", args.route_cases_file] if args.route_cases_file else []
        _run_suite(
            "route_eval.py",
            [
                *common_args,
                *route_dataset_args,
                "--output-prefix",
                "route",
                "--session-prefix",
                f"{run_id}-route",
                "--timeout",
                str(args.timeout),
            ],
        )
    if not args.skip_task:
        task_dataset_args = ["--cases-file", args.task_cases_file] if args.task_cases_file else []
        _run_suite(
            "task_eval.py",
            [*common_args, *task_dataset_args, "--output-prefix", "task", "--session-prefix", f"{run_id}-task"],
        )

    route_report = _load_json(run_dir / "latest.json") if not args.skip_route else None
    task_report = _load_json(run_dir / "task-latest.json") if not args.skip_task else None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": timestamp,
        "label": label or None,
        "base_url": args.base_url,
        "route_dataset_file": args.route_cases_file or None,
        "task_dataset_file": args.task_cases_file or None,
        "route": None,
        "task": None,
    }
    if route_report:
        summary["route"] = {
            "total_cases": route_report["total_cases"],
            "total_turns": route_report["total_turns"],
            "strict_accuracy": route_report["per_turn_strict_accuracy"],
            "lenient_accuracy": route_report["per_turn_lenient_accuracy"],
            "final_route_accuracy": route_report["final_route_strict_accuracy"],
            "safety_correctness": route_report["safety_correctness"],
            "request_error_count": len(route_report["request_errors"]),
        }
    if task_report:
        summary["task"] = {
            "total_tasks": task_report["total_tasks"],
            "success_rate": task_report["task_success_rate"],
            "avg_steps": task_report["avg_steps"],
            "avg_tool_calls": task_report["avg_tool_calls"],
            "replan_rate": task_report["replan_rate"],
            "reflection_finish_rate": task_report["reflection_finish_rate"],
            "evaluation_pass_rate": task_report["evaluation_pass_rate"],
        }

    summary_path = run_dir / "summary.json"
    latest_path = report_root / "latest-summary.json"
    payload = json.dumps(summary, indent=2)
    summary_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    _print_summary(summary)
    print(f"\nRun report: {summary_path}")
    print(f"Latest summary: {latest_path}")


if __name__ == "__main__":
    main()
