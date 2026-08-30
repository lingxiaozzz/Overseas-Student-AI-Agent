import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.rate_limit import SlidingWindowRateLimiter, rate_limiter
from app.main import app


class RateLimitTests(unittest.TestCase):
    def test_sliding_window_returns_retry_delay(self) -> None:
        limiter = SlidingWindowRateLimiter()
        limits = ((2, 60),)
        self.assertIsNone(limiter.check("client", limits, now=100.0))
        self.assertIsNone(limiter.check("client", limits, now=101.0))
        self.assertEqual(limiter.check("client", limits, now=102.0), 58)
        self.assertIsNone(limiter.check("client", limits, now=160.0))

    def test_feedback_endpoint_returns_429_after_limit(self) -> None:
        previous_limit = settings.feedback_rate_limit_per_minute
        previous_feedback_path = settings.feedback_log_path
        settings.feedback_rate_limit_per_minute = 1
        rate_limiter.reset()
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings.feedback_log_path = Path(directory) / "feedback.jsonl"
                client = TestClient(app)
                first = client.post("/feedback", json={"event_id": "run-limit-1", "rating": "helpful"})
                second = client.post("/feedback", json={"event_id": "run-limit-2", "rating": "helpful"})
        finally:
            settings.feedback_rate_limit_per_minute = previous_limit
            settings.feedback_log_path = previous_feedback_path
            rate_limiter.reset()

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)
