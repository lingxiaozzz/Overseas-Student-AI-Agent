from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import settings


class MissingApiKeyError(RuntimeError):
    pass


_llm_provider_override: ContextVar[str | None] = ContextVar("llm_provider_override", default=None)
_llm_model_override: ContextVar[str | None] = ContextVar("llm_model_override", default=None)


@contextmanager
def llm_override(provider: str | None = None, model: str | None = None) -> Iterator[None]:
    provider_token = _llm_provider_override.set((provider or "").strip().lower() or None)
    model_token = _llm_model_override.set((model or "").strip() or None)
    try:
        yield
    finally:
        _llm_provider_override.reset(provider_token)
        _llm_model_override.reset(model_token)


def resolve_llm_provider() -> str:
    override = _llm_provider_override.get()
    if override in {"gemini", "deepseek"}:
        return override
    model = (_llm_model_override.get() or "").lower()
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("gemini"):
        return "gemini"
    return settings.llm_provider if settings.llm_provider in {"gemini", "deepseek"} else "deepseek"


def resolve_chat_model_name(provider: str | None = None) -> str:
    override = _llm_model_override.get()
    if override:
        return override
    resolved = provider or resolve_llm_provider()
    if resolved == "deepseek":
        return settings.deepseek_model
    return settings.gemini_model


def require_chat_api_key(provider: str | None = None) -> None:
    resolved = provider or resolve_llm_provider()
    if resolved == "deepseek":
        if not settings.deepseek_api_key:
            raise MissingApiKeyError("DEEPSEEK_API_KEY is not set.")
        return
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")


def create_chat_model(*, temperature: float = 0.0) -> BaseChatModel:
    provider = resolve_llm_provider()
    require_chat_api_key(provider)
    model_name = resolve_chat_model_name(provider)
    if provider == "deepseek":
        return ChatOpenAI(
            model=model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
            extra_body={
                "thinking": {"type": "enabled" if settings.deepseek_thinking else "disabled"}
            },
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )
