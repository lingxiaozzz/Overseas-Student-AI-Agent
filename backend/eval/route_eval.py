import argparse
import json
import urllib.error
import urllib.request
from collections import Counter, defaultdict


TEST_CASES = [
    {"message": "I am a new USYD student. What should I prepare before arrival?", "expected_route": "rag"},
    {"message": "Do I need OSHC before applying for my visa?", "expected_route": "rag"},
    {"message": "Accommodation tips near USYD campus for first month.", "expected_route": "rag"},
    {"message": "What documents should I bring for enrolment at USYD?", "expected_route": "rag"},
    {"message": "Can you estimate my weekly budget if rent is 420 AUD?", "expected_route": "tool"},
    {"message": "Calculate my weekly cost with rent 500, groceries 140, transport 60.", "expected_route": "tool"},
    {"message": "Build me a pre-arrival checklist if I already have visa and OSHC.", "expected_route": "tool"},
    {"message": "How much should I budget weekly in Sydney as a student?", "expected_route": "tool"},
    {"message": "How can I improve time management for study?", "expected_route": "chat"},
    {"message": "I feel homesick. Any coping tips?", "expected_route": "chat"},
    {"message": "Explain the Pomodoro method in simple words.", "expected_route": "chat"},
    {"message": "Give me a weekly study plan for exams.", "expected_route": "chat"},
]


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /agent-chat route selection accuracy.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of running backend API.",
    )
    parser.add_argument(
        "--session-prefix",
        default="route-eval",
        help="Prefix used for session_id during evaluation.",
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/agent-chat"
    confusion: dict[str, Counter] = defaultdict(Counter)
    mismatches: list[dict] = []
    correct = 0

    for index, case in enumerate(TEST_CASES, start=1):
        payload = {
            "message": case["message"],
            "session_id": f"{args.session_prefix}-{index}",
        }
        response = post_json(endpoint, payload)
        predicted = response.get("route", "unknown")
        expected = case["expected_route"]
        confusion[expected][predicted] += 1

        if predicted == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "message": case["message"],
                    "expected": expected,
                    "predicted": predicted,
                    "router_reason": response.get("router_reason", ""),
                }
            )

    total = len(TEST_CASES)
    accuracy = correct / total if total else 0.0

    print("Route Evaluation Summary")
    print(f"- Total cases: {total}")
    print(f"- Correct: {correct}")
    print(f"- Accuracy: {accuracy:.2%}")
    print("")

    print("Confusion Matrix (expected -> predicted count)")
    ordered_routes = ["chat", "rag", "tool", "unknown"]
    for expected in ordered_routes:
        row = confusion.get(expected, Counter())
        if not row:
            continue
        details = ", ".join(f"{pred}:{row[pred]}" for pred in ordered_routes if row[pred] > 0)
        print(f"- {expected}: {details}")
    print("")

    if mismatches:
        print("Mismatches")
        for mismatch in mismatches:
            print(
                f"- expected={mismatch['expected']} predicted={mismatch['predicted']} "
                f"reason={mismatch['router_reason']}"
            )
            print(f"  message={mismatch['message']}")
    else:
        print("No mismatches found.")


if __name__ == "__main__":
    main()
