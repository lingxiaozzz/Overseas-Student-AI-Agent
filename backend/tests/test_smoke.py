import os
import tempfile
import unittest
from pathlib import Path


RUNTIME_DIR = tempfile.TemporaryDirectory()
os.environ["RUNTIME_DATA_PATH"] = RUNTIME_DIR.name

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_feedback_path = settings.feedback_log_path
        settings.feedback_log_path = Path(RUNTIME_DIR.name) / "observability" / "feedback.jsonl"
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        settings.feedback_log_path = cls.previous_feedback_path
        RUNTIME_DIR.cleanup()

    def test_health_and_web_page(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("留学生咨询助手", page.text)
        self.assertIn("styles.css?v=20260830", page.text)

    def test_feedback_is_persisted_without_prompt_content(self) -> None:
        response = self.client.post(
            "/feedback",
            json={"event_id": "run-ci-smoke", "rating": "helpful"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"accepted": True})

        feedback_path = Path(RUNTIME_DIR.name) / "observability" / "feedback.jsonl"
        self.assertIn('"event_id":"run-ci-smoke"', feedback_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
