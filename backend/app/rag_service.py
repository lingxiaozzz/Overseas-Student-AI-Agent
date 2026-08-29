from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.content_utils import content_to_text
from app.llm_service import MissingApiKeyError, create_chat_model
from app.official_fetch import fetch_official_pages_for_query
from app.prompt_utils import cache_friendly_messages
from app.retry_service import with_retry
from app.schemas import RetrievedContext


RAG_SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Answer using the provided knowledge base context first.
Each context source has a bracketed number. Cite factual claims using that number, for example [1].
If official webpage excerpts are provided, prefer them for current policy details and cite their bracketed number.
If the context does not contain enough information, say what is missing and give cautious general guidance.
Always remind users to verify visa, enrolment, health cover, and legal requirements with official sources.
Use prior conversation history when it helps maintain continuity."""


class KnowledgeBaseNotFoundError(RuntimeError):
    pass


def _load_knowledge_base_documents() -> list[Document]:
    knowledge_base_path = settings.knowledge_base_path
    if not knowledge_base_path.exists():
        raise KnowledgeBaseNotFoundError(f"Knowledge base folder not found: {knowledge_base_path}")

    documents: list[Document] = []
    for file_path in sorted(knowledge_base_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_path.name},
            )
        )

    if not documents:
        raise KnowledgeBaseNotFoundError(f"No markdown files found in: {knowledge_base_path}")

    return documents


@lru_cache(maxsize=1)
def _build_vector_store() -> FAISS:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(_load_knowledge_base_documents())
    candidate_models = [
        settings.embedding_model,
        "models/gemini-embedding-001",
        "models/gemini-embedding-2",
        "models/gemini-embedding-2-preview",
    ]
    # Keep order but remove duplicates.
    unique_candidate_models = list(dict.fromkeys(candidate_models))
    last_error: Exception | None = None

    for model_name in unique_candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=settings.google_api_key,
            )
            return FAISS.from_documents(chunks, embeddings)
        except Exception as exc:
            # Some Gemini API versions do not support older embedding model names.
            # If a model is not found, try the known working fallback model.
            if "NOT_FOUND" in str(exc):
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise RuntimeError(
            "Failed to build FAISS index. Set GEMINI_EMBEDDING_MODEL to a supported embedding model "
            "(for example: models/gemini-embedding-001)."
        ) from last_error

    raise RuntimeError("Failed to build FAISS index due to an unknown embedding configuration error.")


def _format_context(documents: list[Document], *, start_index: int = 1) -> str:
    return "\n\n".join(
        f"[{index}] Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for index, document in enumerate(documents, start=start_index)
    )


def _preview_text(text: str, max_length: int = 350) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_length:
        return compact_text

    return f"{compact_text[:max_length].rstrip()}..."


def _build_retrieved_contexts(
    scored_documents: list[tuple[Document, float]],
) -> list[RetrievedContext]:
    return [
        RetrievedContext(
            rank=rank,
            source=document.metadata.get("source", "unknown"),
            score=float(score),
            content_preview=_preview_text(document.page_content),
        )
        for rank, (document, score) in enumerate(scored_documents, start=1)
    ]


def _append_source_citations(answer: str, sources: list[str]) -> str:
    """Always expose the source-number mapping even if the model omits inline citations."""
    if not sources:
        return answer.strip()
    source_lines = "\n".join(f"[{index}] {source}" for index, source in enumerate(sources, start=1))
    return f"{answer.strip()}\n\nSources:\n{source_lines}".strip()


async def generate_rag_response(
    message: str, chat_history: str = ""
) -> tuple[str, list[str], list[RetrievedContext]]:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    vector_store = _build_vector_store()
    scored_documents = vector_store.similarity_search_with_score(message, k=3)
    retrieved_documents = [document for document, _score in scored_documents]
    retrieved_contexts = _build_retrieved_contexts(scored_documents)
    official_pages = await fetch_official_pages_for_query(message)
    context = _format_context(retrieved_documents)
    official_block = ""
    if official_pages:
        official_block = "\n\nOfficial webpage excerpts:\n" + "\n\n".join(
            f"[{index}] URL: {page.url}\nTitle: {page.title}\n{page.text}"
            for index, page in enumerate(official_pages, start=len(retrieved_contexts) + 1)
        )
        start_rank = len(retrieved_contexts) + 1
        retrieved_contexts.extend(
            RetrievedContext(
                rank=start_rank + index,
                source=page.url,
                score=1.0,
                content_preview=_preview_text(page.text),
            )
            for index, page in enumerate(official_pages)
        )

    model = create_chat_model(temperature=0.2)
    messages = cache_friendly_messages(
        RAG_SYSTEM_PROMPT,
        chat_history,
        f"Question:\n{message}\n\nKnowledge base context:\n{context}{official_block}",
    )
    response = await with_retry(lambda: model.ainvoke(messages))
    sources: list[str] = []
    for context_item in retrieved_contexts:
        if context_item.source not in sources:
            sources.append(context_item.source)
    answer = _append_source_citations(content_to_text(response.content), sources)
    return answer, sources, retrieved_contexts
