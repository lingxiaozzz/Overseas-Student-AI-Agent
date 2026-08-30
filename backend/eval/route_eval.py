import argparse
import json
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .cache_metrics import fetch_cache_metrics, persist_cache_run
from .dataset_loader import load_cases

ROUTE_ORDER = ["chat", "rag", "tool", "unknown"]
LENIENT_AMBIGUOUS_THRESHOLD = 0.3
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180

DEFAULT_CASES_FILE = Path(__file__).resolve().parent / "datasets" / "route_cases.json"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_reports" / "route"


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    request_label: str = "",
) -> dict:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    label = f" ({request_label})" if request_label else ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise TimeoutError(
            f"Request timed out after {timeout_seconds}s{label}. "
            "The agent may be waiting on slow Gemini responses or retries; "
            "check backend logs and retry with a higher --timeout if needed."
        ) from exc
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}{label}: {exc.read().decode('utf-8')}") from exc


def best_expected_route(case: dict) -> str:
    if "ground_truth_best_route" in case:
        return case["ground_truth_best_route"]
    if "expected_route" in case:
        return case["expected_route"]
    expected_routes = case.get("expected_routes", {})
    return max(expected_routes.items(), key=lambda item: item[1])[0] if expected_routes else "unknown"


def lenient_match(predicted: str, case: dict) -> bool:
    if "expected_route" in case:
        return predicted == case["expected_route"]
    return case.get("expected_routes", {}).get(predicted, 0.0) >= LENIENT_AMBIGUOUS_THRESHOLD


def weighted_score(predicted: str, case: dict) -> float:
    if "expected_route" in case:
        return 1.0 if predicted == case["expected_route"] else 0.0
    return float(case.get("expected_routes", {}).get(predicted, 0.0))


def tools_match(expected_tools: list[str], used_tools: list[str]) -> bool:
    return set(expected_tools).issubset(set(used_tools)) if expected_tools else True


def eval_turn(
    endpoint: str,
    session_id: str,
    case_id: str,
    category: str,
    turn_index: int,
    turn_case: dict,
    confusion: dict[str, Counter],
    category_stats: dict[str, dict[str, float]],
    strict_mismatches: list[dict],
    lenient_mismatches: list[dict],
    tool_mismatches: list[dict],
    custom_metrics: dict[str, dict[str, float]],
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, list[str]]:
    response = post_json(
        endpoint,
        {"message": turn_case["message"], "session_id": session_id},
        headers={"x-persist-experience": "false"},
        timeout_seconds=timeout_seconds,
        request_label=f"{case_id} turn {turn_index}",
    )
    predicted = response.get("route", "unknown")
    used_tools = response.get("used_tools", [])
    strict_expected = best_expected_route(turn_case)
    confusion[strict_expected][predicted] += 1

    stats = category_stats[category]
    stats["total"] += 1
    if predicted == strict_expected:
        stats["strict_correct"] += 1
    else:
        strict_mismatches.append(
            {
                "case_id": case_id,
                "category": category,
                "turn_index": turn_index,
                "strict_expected": strict_expected,
                "predicted": predicted,
                "router_reason": response.get("router_reason", ""),
                "intent_label": turn_case.get("intent_label"),
                "expected_reason": turn_case.get("expected_reason"),
                "evaluation_objective": turn_case.get("evaluation_objective"),
                "ground_truth_best_route": turn_case.get("ground_truth_best_route"),
                "safety_evaluation_target": turn_case.get("safety_evaluation_target"),
                "message": turn_case["message"],
            }
        )

    if lenient_match(predicted, turn_case):
        stats["lenient_correct"] += 1
    else:
        lenient_mismatches.append(
            {
                "case_id": case_id,
                "category": category,
                "turn_index": turn_index,
                "predicted": predicted,
                "expected_routes": turn_case.get("expected_routes"),
                "intent_label": turn_case.get("intent_label"),
                "expected_reason": turn_case.get("expected_reason"),
                "evaluation_objective": turn_case.get("evaluation_objective"),
                "ground_truth_best_route": turn_case.get("ground_truth_best_route"),
                "safety_evaluation_target": turn_case.get("safety_evaluation_target"),
                "message": turn_case["message"],
            }
        )

    stats["weighted_score_sum"] += weighted_score(predicted, turn_case)
    expected_tools = turn_case.get("expected_tools", [])
    if not tools_match(expected_tools, used_tools):
        tool_mismatches.append(
            {
                "case_id": case_id,
                "category": category,
                "turn_index": turn_index,
                "expected_tools": expected_tools,
                "used_tools": used_tools,
                "message": turn_case["message"],
            }
        )

    # Metric 1: Context-Sensitivity Rate (context-setting turns should route to chat)
    if "context-setting" in str(turn_case.get("expected_reason", "")).lower():
        custom_metrics["context_sensitivity"]["total"] += 1
        if predicted == "chat":
            custom_metrics["context_sensitivity"]["correct"] += 1

    # Metric 2: Safety Correctness (adversarial turns should match strict expectation)
    if category == "adversarial":
        custom_metrics["safety_correctness"]["total"] += 1
        if predicted == strict_expected:
            custom_metrics["safety_correctness"]["correct"] += 1

    # Metric 3: Ambiguity Precision (ambiguous turns should pick ground-truth best route)
    if category == "ambiguous-intent":
        custom_metrics["ambiguity_precision"]["total"] += 1
        if predicted == turn_case.get("ground_truth_best_route"):
            custom_metrics["ambiguity_precision"]["correct"] += 1

    return predicted, used_tools


def record_failed_turn(
    *,
    case_id: str,
    category: str,
    turn_index: int,
    turn_case: dict,
    error_type: str,
    error_message: str,
    confusion: dict[str, Counter],
    category_stats: dict[str, dict[str, float]],
    strict_mismatches: list[dict],
    lenient_mismatches: list[dict],
    custom_metrics: dict[str, dict[str, float]],
    request_errors: list[dict],
) -> tuple[str, list[str]]:
    predicted = error_type
    used_tools: list[str] = []
    strict_expected = best_expected_route(turn_case)

    request_errors.append(
        {
            "case_id": case_id,
            "category": category,
            "turn_index": turn_index,
            "error_type": error_type,
            "error_message": error_message,
            "message": turn_case.get("message", ""),
        }
    )
    confusion[strict_expected][predicted] += 1

    stats = category_stats[category]
    stats["total"] += 1
    strict_mismatches.append(
        {
            "case_id": case_id,
            "category": category,
            "turn_index": turn_index,
            "strict_expected": strict_expected,
            "predicted": predicted,
            "router_reason": "",
            "intent_label": turn_case.get("intent_label"),
            "expected_reason": turn_case.get("expected_reason"),
            "evaluation_objective": turn_case.get("evaluation_objective"),
            "ground_truth_best_route": turn_case.get("ground_truth_best_route"),
            "safety_evaluation_target": turn_case.get("safety_evaluation_target"),
            "message": turn_case.get("message", ""),
            "error_type": error_type,
            "error_message": error_message,
        }
    )
    lenient_mismatches.append(
        {
            "case_id": case_id,
            "category": category,
            "turn_index": turn_index,
            "predicted": predicted,
            "expected_routes": turn_case.get("expected_routes"),
            "intent_label": turn_case.get("intent_label"),
            "expected_reason": turn_case.get("expected_reason"),
            "evaluation_objective": turn_case.get("evaluation_objective"),
            "ground_truth_best_route": turn_case.get("ground_truth_best_route"),
            "safety_evaluation_target": turn_case.get("safety_evaluation_target"),
            "message": turn_case.get("message", ""),
            "error_type": error_type,
            "error_message": error_message,
        }
    )

    if "context-setting" in str(turn_case.get("expected_reason", "")).lower():
        custom_metrics["context_sensitivity"]["total"] += 1
    if category == "adversarial":
        custom_metrics["safety_correctness"]["total"] += 1
    if category == "ambiguous-intent":
        custom_metrics["ambiguity_precision"]["total"] += 1

    return predicted, used_tools


def run_eval_turn(
    *,
    endpoint: str,
    session_id: str,
    case_id: str,
    category: str,
    turn_index: int,
    turn_case: dict,
    confusion: dict[str, Counter],
    category_stats: dict[str, dict[str, float]],
    strict_mismatches: list[dict],
    lenient_mismatches: list[dict],
    tool_mismatches: list[dict],
    custom_metrics: dict[str, dict[str, float]],
    request_errors: list[dict],
    timeout_seconds: int,
    continue_on_error: bool,
) -> tuple[str, list[str]]:
    try:
        return eval_turn(
            endpoint=endpoint,
            session_id=session_id,
            case_id=case_id,
            category=category,
            turn_index=turn_index,
            turn_case=turn_case,
            confusion=confusion,
            category_stats=category_stats,
            strict_mismatches=strict_mismatches,
            lenient_mismatches=lenient_mismatches,
            tool_mismatches=tool_mismatches,
            custom_metrics=custom_metrics,
            timeout_seconds=timeout_seconds,
        )
    except (TimeoutError, RuntimeError, urllib.error.URLError) as exc:
        if not continue_on_error:
            raise
        print(f"  WARNING: skipped {case_id} turn {turn_index}: {exc}")
        error_type = "timeout" if isinstance(exc, TimeoutError) else "error"
        return record_failed_turn(
            case_id=case_id,
            category=category,
            turn_index=turn_index,
            turn_case=turn_case,
            error_type=error_type,
            error_message=str(exc),
            confusion=confusion,
            category_stats=category_stats,
            strict_mismatches=strict_mismatches,
            lenient_mismatches=lenient_mismatches,
            custom_metrics=custom_metrics,
            request_errors=request_errors,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /agent-chat route selection accuracy.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of running backend API.")
    parser.add_argument("--session-prefix", default="route-eval", help="Session prefix.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory for route JSON reports. Defaults to data/eval_reports/route.",
    )
    parser.add_argument("--output-prefix", default="route-eval", help="Report filename prefix.")
    parser.add_argument(
        "--cases-file",
        default="",
        help="Optional versioned route dataset JSON. Defaults to datasets/route_cases.json.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds for /agent-chat (default: 180).",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip timed-out or failed requests and continue the eval run (default: true).",
    )
    parser.add_argument(
        "--from-case",
        default="",
        help="Resume from a specific case id (e.g. ambiguous-5). Earlier cases are skipped.",
    )
    args = parser.parse_args()

    dataset_file = args.cases_file or str(DEFAULT_CASES_FILE)
    all_cases = load_cases(dataset_file, suite="route")
    selected_cases = all_cases
    if args.from_case:
        case_ids = [case.get("id", f"case-{index}") for index, case in enumerate(all_cases, start=1)]
        if args.from_case not in case_ids:
            raise SystemExit(f"Unknown case id: {args.from_case}")
        start_index = case_ids.index(args.from_case)
        selected_cases = all_cases[start_index:]
        print(f"Resuming from case {args.from_case} ({len(selected_cases)} cases remaining)")

    endpoint = f"{args.base_url.rstrip('/')}/agent-chat"
    cache_metrics_before = fetch_cache_metrics(args.base_url)
    planned_turns = sum(len(case.get("turns", [case])) for case in selected_cases)
    print(
        f"Running route eval: {len(selected_cases)} cases, {planned_turns} turns, "
        f"timeout={args.timeout}s, continue_on_error={args.continue_on_error}"
    )
    confusion: dict[str, Counter] = defaultdict(Counter)
    category_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"total": 0.0, "strict_correct": 0.0, "lenient_correct": 0.0, "weighted_score_sum": 0.0}
    )
    strict_mismatches: list[dict] = []
    lenient_mismatches: list[dict] = []
    tool_mismatches: list[dict] = []
    request_errors: list[dict] = []

    strict_correct = 0
    lenient_correct = 0
    weighted_sum = 0.0
    total_turns = 0
    final_confusion: dict[str, Counter] = defaultdict(Counter)
    final_total = 0
    final_strict_correct = 0
    state_tracking: list[dict] = []
    custom_metrics: dict[str, dict[str, float]] = {
        "context_sensitivity": {"total": 0.0, "correct": 0.0},
        "safety_correctness": {"total": 0.0, "correct": 0.0},
        "ambiguity_precision": {"total": 0.0, "correct": 0.0},
    }

    for index, case in enumerate(selected_cases, start=1):
        case_id = case.get("id", f"case-{index}")
        category = case.get("category", "uncategorized")
        session_id = f"{args.session_prefix}-{case_id}"

        if "turns" in case:
            final_predicted = "unknown"
            final_used_tools: list[str] = []
            expected_state: dict[str, str] = {}
            predicted_state: dict[str, str] = {}
            for turn_index, turn in enumerate(case["turns"], start=1):
                total_turns += 1
                print(f"[{total_turns}/{planned_turns}] {case_id} turn {turn_index}...")
                predicted, used_tools = run_eval_turn(
                    endpoint=endpoint,
                    session_id=session_id,
                    case_id=case_id,
                    category=category,
                    turn_index=turn_index,
                    turn_case=turn,
                    confusion=confusion,
                    category_stats=category_stats,
                    strict_mismatches=strict_mismatches,
                    lenient_mismatches=lenient_mismatches,
                    tool_mismatches=tool_mismatches,
                    custom_metrics=custom_metrics,
                    request_errors=request_errors,
                    timeout_seconds=args.timeout,
                    continue_on_error=args.continue_on_error,
                )
                expected_state[f"turn_{turn_index}"] = best_expected_route(turn)
                predicted_state[f"turn_{turn_index}"] = predicted
                strict_correct += 1 if predicted == best_expected_route(turn) else 0
                lenient_correct += 1 if lenient_match(predicted, turn) else 0
                weighted_sum += weighted_score(predicted, turn)
                final_predicted = predicted
                final_used_tools = used_tools

            if "final_expected_route" in case:
                final_expected = case["final_expected_route"]
                expected_state["final"] = final_expected
                predicted_state["final"] = final_predicted
                state_tracking.append(
                    {
                        "case_id": case_id,
                        "category": category,
                        "intent_label": case.get("intent_label"),
                        "expected_reason": case.get("expected_reason"),
                        "state": {
                            "expected": expected_state,
                            "predicted": predicted_state,
                        },
                    }
                )
                final_total += 1
                final_confusion[final_expected][final_predicted] += 1
                final_strict_correct += 1 if final_predicted == final_expected else 0
                expected_tools = case.get("expected_tools", [])
                if expected_tools and not tools_match(expected_tools, final_used_tools):
                    tool_mismatches.append(
                        {
                            "case_id": case_id,
                            "category": category,
                            "turn_index": "final",
                            "expected_tools": expected_tools,
                            "used_tools": final_used_tools,
                            "intent_label": case.get("intent_label"),
                            "expected_reason": case.get("expected_reason"),
                            "message": case["turns"][-1]["message"],
                        }
                    )
            continue

        total_turns += 1
        print(f"[{total_turns}/{planned_turns}] {case_id}...")
        predicted, _ = run_eval_turn(
            endpoint=endpoint,
            session_id=session_id,
            case_id=case_id,
            category=category,
            turn_index=1,
            turn_case=case,
            confusion=confusion,
            category_stats=category_stats,
            strict_mismatches=strict_mismatches,
            lenient_mismatches=lenient_mismatches,
            tool_mismatches=tool_mismatches,
            custom_metrics=custom_metrics,
            request_errors=request_errors,
            timeout_seconds=args.timeout,
            continue_on_error=args.continue_on_error,
        )
        strict_correct += 1 if predicted == best_expected_route(case) else 0
        lenient_correct += 1 if lenient_match(predicted, case) else 0
        weighted_sum += weighted_score(predicted, case)

    per_turn_strict_accuracy = strict_correct / total_turns if total_turns else 0.0
    per_turn_lenient_accuracy = lenient_correct / total_turns if total_turns else 0.0
    per_turn_weighted_score = weighted_sum / total_turns if total_turns else 0.0
    final_route_strict_accuracy = final_strict_correct / final_total if final_total else 0.0
    context_sensitivity_rate = (
        custom_metrics["context_sensitivity"]["correct"] / custom_metrics["context_sensitivity"]["total"]
        if custom_metrics["context_sensitivity"]["total"]
        else 0.0
    )
    safety_correctness = (
        custom_metrics["safety_correctness"]["correct"] / custom_metrics["safety_correctness"]["total"]
        if custom_metrics["safety_correctness"]["total"]
        else 0.0
    )
    ambiguity_precision = (
        custom_metrics["ambiguity_precision"]["correct"] / custom_metrics["ambiguity_precision"]["total"]
        if custom_metrics["ambiguity_precision"]["total"]
        else 0.0
    )

    category_metrics = {
        name: {
            "total": int(stats["total"]),
            "strict_accuracy": (stats["strict_correct"] / stats["total"]) if stats["total"] else 0.0,
            "lenient_accuracy": (stats["lenient_correct"] / stats["total"]) if stats["total"] else 0.0,
            "weighted_score": (stats["weighted_score_sum"] / stats["total"]) if stats["total"] else 0.0,
        }
        for name, stats in category_stats.items()
    }
    confusion_dict = {
        expected: {pred: int(row[pred]) for pred in ROUTE_ORDER if row[pred] > 0}
        for expected, row in confusion.items()
        if row
    }
    final_confusion_dict = {
        expected: {pred: int(row[pred]) for pred in ROUTE_ORDER if row[pred] > 0}
        for expected, row in final_confusion.items()
        if row
    }

    print("Route Evaluation Summary")
    print(f"- Selected cases: {len(selected_cases)} / {len(all_cases)}")
    print(f"- Total evaluated turns: {total_turns}")
    print(f"- Request errors/timeouts: {len(request_errors)}")
    print(f"- Per-turn strict accuracy: {per_turn_strict_accuracy:.2%}")
    print(f"- Per-turn lenient accuracy: {per_turn_lenient_accuracy:.2%}")
    print(f"- Per-turn weighted score: {per_turn_weighted_score:.3f}")
    print(f"- Final-route cases: {final_total}")
    print(f"- Final-route strict accuracy: {final_route_strict_accuracy:.2%}")
    print(f"- Context-Sensitivity Rate: {context_sensitivity_rate:.2%}")
    print(f"- Safety Correctness: {safety_correctness:.2%}")
    print(f"- Ambiguity Precision: {ambiguity_precision:.2%}")

    if request_errors:
        print("- Errors:")
        for item in request_errors:
            print(f"  - {item['case_id']} turn {item['turn_index']}: {item['error_type']}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.output_prefix}-{timestamp}.json"
    latest_path = output_dir / "latest.json"
    report = {
        "generated_at_utc": timestamp,
        "base_url": args.base_url,
        "from_case": args.from_case or None,
        "continue_on_error": args.continue_on_error,
        "request_timeout_seconds": args.timeout,
        "total_cases": len(selected_cases),
        "total_cases_available": len(all_cases),
        "dataset_file": dataset_file,
        "total_turns": total_turns,
        "request_errors": request_errors,
        "per_turn_strict_accuracy": per_turn_strict_accuracy,
        "per_turn_lenient_accuracy": per_turn_lenient_accuracy,
        "per_turn_weighted_score": per_turn_weighted_score,
        "final_route_total": final_total,
        "final_route_strict_accuracy": final_route_strict_accuracy,
        "context_sensitivity_rate": context_sensitivity_rate,
        "safety_correctness": safety_correctness,
        "ambiguity_precision": ambiguity_precision,
        "category_metrics": category_metrics,
        "confusion_matrix": confusion_dict,
        "final_route_confusion_matrix": final_confusion_dict,
        "strict_mismatches": strict_mismatches,
        "lenient_mismatches": lenient_mismatches,
        "tool_mismatches": tool_mismatches,
        "state_tracking": state_tracking,
        "cases": all_cases,
        "ambiguous_lenient_threshold": LENIENT_AMBIGUOUS_THRESHOLD,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache_report_path = persist_cache_run(
        suite="route",
        base_url=args.base_url,
        before=cache_metrics_before,
        after=fetch_cache_metrics(args.base_url),
    )
    print(f"Report saved: {report_path}")
    print(f"Latest report: {latest_path}")
    if cache_report_path:
        print(f"Cache summary saved: {cache_report_path}")


if __name__ == "__main__":
    main()
