import json
import tempfile
import unittest
from pathlib import Path

from eval.observability_report import build_report


class ObservabilityReportTests(unittest.TestCase):
    def test_aggregates_runs_cache_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "agent_runs.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_id": "run-1",
                                "outcome": "success",
                                "route": "rag",
                                "elapsed_ms": 100,
                                "tool_calls": 1,
                                "replanned": False,
                                "evaluation": {"passed": True},
                                "cache": {"prompt_tokens": 100, "cache_hit_tokens": 25},
                            }
                        ),
                        json.dumps(
                            {
                                "event_id": "run-2",
                                "outcome": "error",
                                "error_type": "TimeoutError",
                                "elapsed_ms": 300,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            (data_dir / "feedback.jsonl").write_text(
                json.dumps({"event_id": "run-1", "rating": "helpful"}) + "\n",
                encoding="utf-8",
            )

            report = build_report(data_dir)

        self.assertEqual(report["runs"], {"total": 2, "successful": 1, "failed": 1, "success_rate": 0.5})
        self.assertEqual(report["routing"], {"rag": 1})
        self.assertEqual(report["cache"]["hit_rate"], 0.25)
        self.assertEqual(report["feedback"]["helpful_rate"], 1.0)
        self.assertEqual(report["error_types"], {"TimeoutError": 1})
