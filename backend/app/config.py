import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Small settings object for Step 1.

    Later we can replace this with pydantic-settings when the project has more
    configuration.
    """

    app_name: str = "Overseas Student AI Agent"
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


settings = Settings()
