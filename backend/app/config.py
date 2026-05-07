import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    """Small settings object for Step 1.

    Later we can replace this with pydantic-settings when the project has more
    configuration.
    """

    app_name: str = "Overseas Student AI Agent"
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    memory_max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "6"))
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    retry_initial_seconds: float = float(os.getenv("RETRY_INITIAL_SECONDS", "1.0"))
    retry_max_seconds: float = float(os.getenv("RETRY_MAX_SECONDS", "8.0"))
    knowledge_base_path: Path = PROJECT_ROOT / "data" / "knowledge_base"


settings = Settings()
