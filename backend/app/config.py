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
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_plan_steps: int = int(os.getenv("MAX_PLAN_STEPS", "4"))
    experience_memory_max_items: int = int(os.getenv("EXPERIENCE_MEMORY_MAX_ITEMS", "200"))
    experience_memory_top_k: int = int(os.getenv("EXPERIENCE_MEMORY_TOP_K", "3"))
    experience_memory_min_score: float = float(os.getenv("EXPERIENCE_MEMORY_MIN_SCORE", "0.2"))
    experience_memory_enabled: bool = os.getenv("EXPERIENCE_MEMORY_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    long_term_memory_enabled: bool = os.getenv("LONG_TERM_MEMORY_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    long_term_memory_max_items: int = int(os.getenv("LONG_TERM_MEMORY_MAX_ITEMS", "100"))
    long_term_memory_top_k: int = int(os.getenv("LONG_TERM_MEMORY_TOP_K", "5"))
    long_term_memory_min_score: float = float(os.getenv("LONG_TERM_MEMORY_MIN_SCORE", "0.1"))
    long_term_memory_max_facts_per_write: int = int(
        os.getenv("LONG_TERM_MEMORY_MAX_FACTS_PER_WRITE", "3")
    )
    evaluation_pass_score: float = float(os.getenv("EVALUATION_PASS_SCORE", "0.6"))
    knowledge_base_path: Path = PROJECT_ROOT / "data" / "knowledge_base"
    experience_memory_path: Path = PROJECT_ROOT / "data" / "memory" / "experiences.json"
    long_term_memory_path: Path = PROJECT_ROOT / "data" / "memory" / "long_term.json"


settings = Settings()
