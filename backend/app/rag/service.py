from collections import Counter
from functools import lru_cache
from math import log
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.api.schemas import RetrievedContext
from app.core.config import settings
from app.core.llm import MissingApiKeyError, create_chat_model
from app.core.prompts import cache_friendly_messages
from app.core.retry import with_retry
from app.rag.official_fetch import fetch_official_pages_for_query
from app.utils.content import content_to_text, response_language_instruction


RAG_SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Answer using the provided knowledge base context first.
Each context source has a bracketed number. Cite factual claims using that number, for example [1].
If official webpage excerpts are provided, prefer them for current policy details and cite their bracketed number.
If the context does not contain enough information, say what is missing and give cautious general guidance.
Always remind users to verify visa, enrolment, health cover, and legal requirements with official sources.
Use prior conversation history when it helps maintain continuity."""


class KnowledgeBaseNotFoundError(RuntimeError):
    pass


_CHINESE_CHARACTER_RE = re.compile(r"[\u4e00-\u9fff]")
_RETRIEVAL_TOP_K = 3
_RETRIEVAL_CANDIDATE_K = 80
_ENGLISH_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CHINESE_TEXT_RE = re.compile(r"[\u4e00-\u9fff]+")
_BM25_K1 = 1.2
_BM25_B = 0.75
_RERANK_STOPWORDS = {
    "about", "after", "and", "are", "can", "could", "do", "for", "from", "have", "how",
    "i", "if", "in", "is", "it", "my", "need", "of", "on", "or", "should", "the", "to",
    "what", "when", "while", "why", "with", "you", "your",
    # Corpus-wide identity terms are not useful for selecting a support topic.
    "consider", "international", "new", "options", "student", "students", "sydney", "university",
    "usyd",
}


def _document_language(file_name: str) -> str:
    return "zh-CN" if file_name.lower().endswith(".zh-cn.md") else "en"


def _query_language(message: str) -> str:
    return "zh-CN" if _CHINESE_CHARACTER_RE.search(message) else "en"


@lru_cache(maxsize=1)
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
                metadata={
                    "source": file_path.name,
                    "language": _document_language(file_path.name),
                },
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


def _bm25_tokens(text: str) -> list[str]:
    """Tokenise English words and Chinese character bigrams without extra dependencies."""
    english_tokens = [
        token
        for token in _ENGLISH_TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in _RERANK_STOPWORDS
    ]
    chinese_bigrams = [
        segment[index : index + 2]
        for segment in _CHINESE_TEXT_RE.findall(text)
        for index in range(max(len(segment) - 1, 0))
    ]
    return [*english_tokens, *chinese_bigrams]


def _bm25_scores(query: str, source_texts: dict[str, str]) -> dict[str, float]:
    """Calculate source-level BM25 scores for mixed Chinese and English text."""
    query_tokens = list(dict.fromkeys(_bm25_tokens(query)))
    tokenized_sources = {source: _bm25_tokens(text) for source, text in source_texts.items()}
    if not query_tokens or not tokenized_sources:
        return {source: 0.0 for source in source_texts}

    document_count = len(tokenized_sources)
    average_length = sum(len(tokens) for tokens in tokenized_sources.values()) / document_count
    document_frequency = Counter(
        token for tokens in tokenized_sources.values() for token in set(tokens)
    )
    scores: dict[str, float] = {}
    for source, tokens in tokenized_sources.items():
        term_frequency = Counter(tokens)
        length_normalizer = _BM25_K1 * (
            1 - _BM25_B + _BM25_B * len(tokens) / max(average_length, 1)
        )
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency[token]
            if not frequency:
                continue
            inverse_frequency = log(
                1 + (document_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
            )
            score += inverse_frequency * (frequency * (_BM25_K1 + 1)) / (frequency + length_normalizer)
        scores[source] = score
    return scores


def _select_language_aware_documents(
    scored_documents: list[tuple[Document, float]],
    *,
    preferred_language: str,
    query: str,
    top_k: int = _RETRIEVAL_TOP_K,
) -> list[tuple[Document, float]]:
    """Rerank source-level candidates while preserving language preference.

    FAISS provides semantic recall. This second stage groups chunks by source,
    combines their vector rank with bilingual BM25 keyword relevance, then retains the
    best vector chunk for each source. Source-level deduplication also keeps
    context citation indices aligned with the public ``sources`` list.
    """
    source_candidates: dict[str, list[tuple[int, Document, float]]] = {}
    for rank, (document, score) in enumerate(scored_documents, start=1):
        source = str(document.metadata.get("source", "unknown"))
        source_candidates.setdefault(source, []).append((rank, document, score))

    knowledge_base_texts = {
        str(document.metadata.get("source", "unknown")): document.page_content
        for document in _load_knowledge_base_documents()
    }
    source_texts = {
        source: knowledge_base_texts.get(
            source,
            " ".join(document.page_content for _, document, _ in source_items),
        )
        for source, source_items in source_candidates.items()
    }
    bm25_scores = _bm25_scores(query, source_texts)
    max_bm25_score = max(bm25_scores.values(), default=0.0)
    reranked: list[tuple[int, float, Document, float]] = []
    for source, source_items in source_candidates.items():
        best_rank, best_document, best_score = source_items[0]
        # FAISS rank remains a tie-breaker. Exact domain terms should be able
        # to correct an otherwise plausible but wrong semantic first result.
        vector_rank_score = 0.1 / best_rank
        normalized_bm25_score = bm25_scores[source] / max_bm25_score if max_bm25_score else 0.0
        hybrid_score = vector_rank_score + normalized_bm25_score
        language_priority = int(
            best_document.metadata.get("language", "en") != preferred_language
        )
        reranked.append((language_priority, hybrid_score, best_document, best_score))

    reranked.sort(key=lambda item: (item[0], -item[1]))
    selected: list[tuple[Document, float]] = []
    for _, _, document, score in reranked:
        selected.append((document, score))
        if len(selected) == top_k:
            break
    return selected


def _append_source_citations(answer: str, sources: list[str], language: str = "en") -> str:
    """Always expose the source-number mapping even if the model omits inline citations."""
    if not sources:
        return answer.strip()
    source_lines = "\n".join(f"[{index}] {source}" for index, source in enumerate(sources, start=1))
    label = "参考来源" if language == "zh-CN" else "Sources"
    return f"{answer.strip()}\n\n{label}:\n{source_lines}".strip()


async def generate_rag_response(
    message: str, chat_history: str = "", response_language: str | None = None
) -> tuple[str, list[str], list[RetrievedContext]]:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    vector_store = _build_vector_store()
    preferred_language = _query_language(message)
    language = response_language or preferred_language
    candidates = vector_store.similarity_search_with_score(message, k=_RETRIEVAL_CANDIDATE_K)
    scored_documents = _select_language_aware_documents(
        candidates,
        preferred_language=preferred_language,
        query=message,
    )
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
        f"{RAG_SYSTEM_PROMPT}\n\n{response_language_instruction(language)}",
        chat_history,
        f"Question:\n{message}\n\nKnowledge base context:\n{context}{official_block}",
    )
    response = await with_retry(lambda: model.ainvoke(messages))
    sources: list[str] = []
    for context_item in retrieved_contexts:
        if context_item.source not in sources:
            sources.append(context_item.source)
    answer = _append_source_citations(content_to_text(response.content), sources, language)
    return answer, sources, retrieved_contexts
