import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .cache_metrics import fetch_cache_metrics, persist_cache_run
from .dataset_loader import load_cases


DEFAULT_CASES_FILE = Path(__file__).resolve().parent / "datasets" / "task_cases.json"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_reports" / "task"


def post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    request_headers = {"Content-Type": "application/json", "x-persist-experience": "false"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def evaluate_case(case: dict, response: dict) -> dict:
    metrics = response.get("metrics") or {}
    reflection = response.get("reflection") or {}
    evaluation = response.get("evaluation") or {}
    plan = response.get("plan") or {}
    steps = response.get("steps") or []
    used_tools = response.get("used_tools") or []
    sources = response.get("sources") or []
    answer = str(response.get("answer") or "")
    final_route = response.get("route", "unknown")
    step_routes = [item.get("route") for item in steps if item.get("route")]

    steps_used = int(metrics.get("steps_used", len(steps) or 0))
    tool_calls = int(metrics.get("tool_calls", len(used_tools)))
    replanned = bool(metrics.get("replanned", False))
    memory_hits = int(metrics.get("memory_hits", 0))
    reflect_done = bool(reflection.get("done", False))
    reflect_action = reflection.get("next_action", "")

    checks: dict[str, bool] = {}
    checks["non_empty_answer"] = (not case.get("require_non_empty_answer", True)) or bool(answer.strip())
    checks["min_steps"] = steps_used >= int(case.get("min_steps", 1))
    checks["max_steps"] = steps_used <= int(case.get("max_steps", 4))
    checks["reflection_finished"] = reflect_done and reflect_action == "finish"

    if "expected_final_route" in case:
        checks["final_route"] = final_route == case["expected_final_route"]
    if "expected_routes_any" in case:
        expected_any = set(case["expected_routes_any"])
        checks["routes_any"] = bool(expected_any.intersection(step_routes or [final_route]))
    if "expected_routes_all" in case:
        expected_all = set(case["expected_routes_all"])
        checks["routes_all"] = expected_all.issubset(set(step_routes or [final_route]))

    required_tools = case.get("require_tools", [])
    checks["tools"] = set(required_tools).issubset(set(used_tools))

    success = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "success": success,
        "failures": failures,
        "checks": checks,
        "message": case["message"],
        "success_criteria": case.get("success_criteria"),
        "predicted_final_route": final_route,
        "step_routes": step_routes,
        "used_tools": used_tools,
        "sources": sources,
        "steps_used": steps_used,
        "tool_calls": tool_calls,
        "replanned": replanned,
        "memory_hits": memory_hits,
        "subgoals": plan.get("subgoals", []),
        "reflection_lesson": reflection.get("lesson", ""),
        "evaluation": {
            "passed": bool(evaluation.get("passed", False)),
            "score": float(evaluation.get("score", 0.0)),
            "feedback": str(evaluation.get("feedback", "")),
            "source": evaluation.get("source", "rule_fallback"),
            "triggered_replan": bool(evaluation.get("triggered_replan", False)),
        },
        "answer_preview": " ".join(answer.split())[:180],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /agent-chat task-level planning metrics.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of running backend API.")
    parser.add_argument("--session-prefix", default="task-eval", help="Session prefix.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory for task JSON reports. Defaults to data/eval_reports/task.",
    )
    parser.add_argument("--output-prefix", default="task-eval", help="Report filename prefix.")
    parser.add_argument(
        "--cases-file",
        default="",
        help="Optional versioned task dataset JSON. Defaults to datasets/task_cases.json.",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated task case ids for a targeted run. Does not overwrite task-latest.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record failed API requests and continue the run (default: true).",
    )
    parser.add_argument(
        "--from-case",
        default="",
        help="Resume from a task case id. Earlier cases are skipped.",
    )
    args = parser.parse_args()
    is_targeted_run = bool(args.case_ids.strip() or args.from_case.strip())

    dataset_file = args.cases_file or str(DEFAULT_CASES_FILE)
    cases = load_cases(dataset_file, suite="task")
    selected_case_ids = [case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()]
    if selected_case_ids:
        available_case_ids = {case["id"] for case in cases}
        unknown_case_ids = [case_id for case_id in selected_case_ids if case_id not in available_case_ids]
        if unknown_case_ids:
            parser.error(f"Unknown task case id(s): {', '.join(unknown_case_ids)}")
        selected_ids = set(selected_case_ids)
        cases = [case for case in cases if case["id"] in selected_ids]
    if args.from_case:
        case_ids = [case["id"] for case in cases]
        if args.from_case not in case_ids:
            parser.error(f"Unknown task case id: {args.from_case}")
        start_index = case_ids.index(args.from_case)
        cases = cases[start_index:]
        print(f"Resuming from case {args.from_case} ({len(cases)} cases remaining)")
    cache_metrics_before = fetch_cache_metrics(args.base_url)
    endpoint = f"{args.base_url.rstrip('/')}/agent-chat"
    results: list[dict] = []
    request_errors: list[dict[str, str]] = []

    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        session_id = f"{args.session_prefix}-{case_id}"
        print(f"[{index}/{len(cases)}] Running {case_id}...")
        try:
            response = post_json(
                endpoint,
                {"message": case["message"], "session_id": session_id},
                headers={"x-trace-id": f"task-{case_id}"},
            )
            results.append(evaluate_case(case, response))
        except (TimeoutError, RuntimeError, urllib.error.URLError) as exc:
            if not args.continue_on_error:
                raise
            print(f"  WARNING: request failed for {case_id}: {exc}")
            failed_result = evaluate_case(case, {})
            failed_result["request_error"] = str(exc)
            results.append(failed_result)
            request_errors.append({"case_id": case_id, "message": str(exc)})

    successes = [item for item in results if item["success"]]
    task_success_rate = len(successes) / len(results) if results else 0.0
    avg_steps = _safe_mean([float(item["steps_used"]) for item in results])
    avg_tool_calls = _safe_mean([float(item["tool_calls"]) for item in results])
    replan_rate = _safe_mean([1.0 if item["replanned"] else 0.0 for item in results])
    reflection_finish_rate = _safe_mean(
        [1.0 if item["checks"].get("reflection_finished") else 0.0 for item in results]
    )
    memory_hit_rate = _safe_mean([1.0 if item["memory_hits"] > 0 else 0.0 for item in results])
    evaluation_pass_rate = _safe_mean(
        [1.0 if item["evaluation"]["passed"] else 0.0 for item in results]
    )
    avg_evaluation_score = _safe_mean([item["evaluation"]["score"] for item in results])

    by_category: dict[str, dict[str, float]] = {}
    for item in results:
        bucket = by_category.setdefault(item["category"], {"total": 0.0, "success": 0.0})
        bucket["total"] += 1
        bucket["success"] += 1.0 if item["success"] else 0.0
    category_metrics = {
        name: {
            "total": int(stats["total"]),
            "success_rate": (stats["success"] / stats["total"]) if stats["total"] else 0.0,
        }
        for name, stats in by_category.items()
    }

    failures = [item for item in results if not item["success"]]

    print("\nTask Evaluation Summary")
    print(f"- Total tasks: {len(results)}")
    print(f"- Task success rate: {task_success_rate:.2%}")
    print(f"- Avg steps: {avg_steps:.2f}")
    print(f"- Avg tool calls: {avg_tool_calls:.2f}")
    print(f"- Replan rate: {replan_rate:.2%}")
    print(f"- Reflection finish rate: {reflection_finish_rate:.2%}")
    print(f"- Memory hit rate: {memory_hit_rate:.2%}")
    print(f"- Evaluation pass rate: {evaluation_pass_rate:.2%}")
    print(f"- Avg evaluation score: {avg_evaluation_score:.2f}")
    print(f"- Request errors/timeouts: {len(request_errors)}")
    for name, stats in category_metrics.items():
        print(f"- {name}: success={stats['success_rate']:.2%} ({stats['total']} tasks)")
    if failures:
        print("- Failures:")
        for item in failures:
            print(f"  - {item['id']}: {', '.join(item['failures'])}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.output_prefix}-{timestamp}.json"
    latest_path = output_dir / "task-latest.json"
    report = {
        "generated_at_utc": timestamp,
        "base_url": args.base_url,
        "dataset_file": dataset_file,
        "total_tasks": len(results),
        "task_success_rate": task_success_rate,
        "avg_steps": avg_steps,
        "avg_tool_calls": avg_tool_calls,
        "replan_rate": replan_rate,
        "reflection_finish_rate": reflection_finish_rate,
        "memory_hit_rate": memory_hit_rate,
        "evaluation_pass_rate": evaluation_pass_rate,
        "avg_evaluation_score": avg_evaluation_score,
        "request_errors": request_errors,
        "category_metrics": category_metrics,
        "failures": failures,
        "results": results,
        "cases": cases,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not is_targeted_run:
        latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache_report_path = persist_cache_run(
        suite="task",
        base_url=args.base_url,
        before=cache_metrics_before,
        after=fetch_cache_metrics(args.base_url),
    )
    print(f"\nReport saved: {report_path}")
    if not is_targeted_run:
        print(f"Latest report: {latest_path}")
    if cache_report_path:
        print(f"Cache summary saved: {cache_report_path}")


if __name__ == "__main__":
    main()
