import os
from pathlib import Path

from dotenv import load_dotenv

# config.py -> core -> app -> backend -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Small settings object for Step 1.

    Later we can replace this with pydantic-settings when the project has more
    configuration.
    """

    app_name: str = "Overseas Student AI Agent"
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_thinking: bool = os.getenv("DEEPSEEK_THINKING", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }
    memory_max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "6"))
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    retry_initial_seconds: float = float(os.getenv("RETRY_INITIAL_SECONDS", "1.0"))
    retry_max_seconds: float = float(os.getenv("RETRY_MAX_SECONDS", "8.0"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_plan_steps: int = int(os.getenv("MAX_PLAN_STEPS", "4"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "6"))
    max_tool_calls: int = int(os.getenv("MAX_TOOL_CALLS", "3"))
    max_agent_runtime_seconds: float = float(os.getenv("MAX_AGENT_RUNTIME_SECONDS", "90"))
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
    long_term_memory_ttl_days: int = int(os.getenv("LONG_TERM_MEMORY_TTL_DAYS", "180"))
    evaluation_pass_score: float = float(os.getenv("EVALUATION_PASS_SCORE", "0.6"))
    official_fetch_enabled: bool = os.getenv("OFFICIAL_FETCH_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    official_fetch_timeout_seconds: float = float(os.getenv("OFFICIAL_FETCH_TIMEOUT_SECONDS", "8.0"))
    official_fetch_max_chars: int = int(os.getenv("OFFICIAL_FETCH_MAX_CHARS", "4000"))
    official_fetch_max_pages: int = int(os.getenv("OFFICIAL_FETCH_MAX_PAGES", "2"))
    knowledge_base_path: Path = PROJECT_ROOT / "data" / "knowledge_base"
    # Keep mutable production data separate from the versioned knowledge base.
    # A platform disk can mount here without hiding data/knowledge_base in the image.
    runtime_data_path: Path = Path(os.getenv("RUNTIME_DATA_PATH", str(PROJECT_ROOT / "data")))
    agent_run_log_path: Path = runtime_data_path / "observability" / "agent_runs.jsonl"
    feedback_log_path: Path = runtime_data_path / "observability" / "feedback.jsonl"
    experience_memory_path: Path = runtime_data_path / "memory" / "experiences.json"
    long_term_memory_path: Path = runtime_data_path / "memory" / "long_term.json"


settings = Settings()
