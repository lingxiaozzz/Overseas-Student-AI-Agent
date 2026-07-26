"""Interview-ready demo for /agent-chat hierarchical agent runtime."""

from __future__ import annotations

import argparse
import json
import textwrap
import urllib.error
import urllib.request
from typing import Any


DEMO_CASES = [
    {
        "id": "demo-rag",
        "title": "1) RAG: USYD pre-arrival checklist",
        "message": "I am a new international student at USYD. What should I prepare before arrival?",
        "expect": "Should route to rag and retrieve pre-arrival guidance.",
        "required_route": "rag",
        "required_tools": [],
        "min_steps": 1,
    },
    {
        "id": "demo-tool",
        "title": "2) Tool: weekly budget estimation",
        "message": "Can you estimate my weekly budget if rent is 420 AUD?",
        "expect": "Should call estimate_weekly_budget tool.",
        "required_route": "tool",
        "required_tools": ["estimate_weekly_budget"],
        "min_steps": 1,
    },
    {
        "id": "demo-multi",
        "title": "3) Multi-step: arrival prep + budget",
        "message": "Help me prepare for USYD arrival and estimate weekly budget if rent is 420 AUD.",
        "expect": "Should decompose into multiple steps across rag/tool.",
        "required_routes_any": ["rag", "tool"],
        "required_tools": ["estimate_weekly_budget"],
        "min_steps": 2,
    },
]


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach {url}. Start the API first: python -m uvicorn app.main:app --reload"
        ) from exc


def _preview(text: str, width: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    return textwrap.shorten(cleaned, width=width, placeholder="...")


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    used_tools = set(response.get("used_tools") or [])
    steps = response.get("steps") or []
    step_routes = {step.get("route") for step in steps}
    metrics = response.get("metrics") or {}
    steps_used = int(metrics.get("steps_used", len(steps) or 0))

    if case.get("required_route") and response.get("route") != case["required_route"]:
        failures.append(f"route={response.get('route')} expected={case['required_route']}")

    required_any = set(case.get("required_routes_any") or [])
    if required_any and not required_any.intersection(step_routes or {response.get("route")}):
        failures.append(f"missing routes any of {sorted(required_any)}")

    required_tools = set(case.get("required_tools") or [])
    if not required_tools.issubset(used_tools):
        failures.append(f"tools={sorted(used_tools)} expected to include {sorted(required_tools)}")

    min_steps = int(case.get("min_steps", 1))
    if steps_used < min_steps:
        failures.append(f"steps_used={steps_used} expected>={min_steps}")

    return failures


def print_case_result(
    case: dict[str, Any],
    response: dict[str, Any],
    trace_id: str,
    failures: list[str],
) -> None:
    plan = response.get("plan") or {}
    reflection = response.get("reflection") or {}
    evaluation = response.get("evaluation") or {}
    metrics = response.get("metrics") or {}
    environment = response.get("environment") or {}
    steps = response.get("steps") or []

    print("=" * 72)
    print(case["title"])
    print(f"Expect: {case['expect']}")
    print(f"Result: {'PASS' if not failures else 'FAIL'} {('; '.join(failures) if failures else '')}")
    print(f"trace_id: {trace_id}")
    print(f"route: {response.get('route')} | reason: {response.get('router_reason')}")
    print(f"plan.goal: {plan.get('goal')}")
    print(f"plan.subgoals ({len(plan.get('subgoals', []))}): {plan.get('subgoals', [])}")
    print("steps:")
    for step in steps:
        print(
            f"  - #{step.get('step_index')} [{step.get('route')}] "
            f"reward={step.get('reward')} tools={step.get('used_tools', [])} "
            f"| {step.get('subgoal')}"
        )
    print(
        "reflection: "
        f"action={reflection.get('next_action')} "
        f"achieved={reflection.get('goal_achieved')} "
        f"source={reflection.get('judge_source')} "
        f"lesson={reflection.get('lesson')}"
    )
    print(
        "evaluation: "
        f"passed={evaluation.get('passed')} "
        f"score={evaluation.get('score')} "
        f"source={evaluation.get('source')} "
        f"triggered_replan={evaluation.get('triggered_replan')} "
        f"feedback={evaluation.get('feedback')}"
    )
    print(
        "metrics: "
        f"steps={metrics.get('steps_used')} "
        f"tools={metrics.get('tool_calls')} "
        f"memory_hits={metrics.get('memory_hits')} "
        f"reward={metrics.get('total_reward')} "
        f"replanned={metrics.get('replanned')}"
    )
    print(
        "environment: "
        f"{environment.get('name')} action_space={environment.get('action_space')}"
    )
    if response.get("memory_lessons"):
        print(f"memory_lessons: {response.get('memory_lessons')}")
    if response.get("used_tools"):
        print(f"used_tools: {response.get('used_tools')}")
    if response.get("sources"):
        print(f"sources: {response.get('sources')}")
    print(f"answer: {_preview(str(response.get('answer', '')))}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Overseas Student AI Agent demo scenarios.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--session-id", default="demo-interview", help="Shared session id.")
    parser.add_argument(
        "--persist-experience",
        default="false",
        choices=["true", "false"],
        help="Whether demo writes experience memory.",
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/agent-chat"
    print("Overseas Student AI Agent Demo")
    print(f"Endpoint: {endpoint}")
    print(f"session_id: {args.session_id}")
    print()

    failed_cases = 0
    for case in DEMO_CASES:
        trace_id = f"demo-{case['id']}"
        headers = {
            "Content-Type": "application/json",
            "x-trace-id": trace_id,
            "x-persist-experience": args.persist_experience,
        }
        payload = {
            "message": case["message"],
            "session_id": f"{args.session_id}-{case['id']}",
        }
        print(f"Running {case['id']} ...")
        response = post_json(endpoint, payload, headers)
        failures = evaluate_case(case, response)
        if failures:
            failed_cases += 1
        print_case_result(case, response, trace_id, failures)

    print("Demo complete.")
    print(f"Summary: {len(DEMO_CASES) - failed_cases}/{len(DEMO_CASES)} passed")
    print("Tip: set LOG_LEVEL=TRACE on the server to show plan/act/reflect traces.")
    if failed_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
