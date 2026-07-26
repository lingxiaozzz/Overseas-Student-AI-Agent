from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chat_service import MissingApiKeyError
from app.config import settings
from app.content_utils import content_to_text
from app.retry_service import with_retry
from app.schemas import RetrievedContext


RAG_SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Answer using the provided knowledge base context first.
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


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for document in documents
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


async def generate_rag_response(
    message: str, chat_history: str = ""
) -> tuple[str, list[str], list[RetrievedContext]]:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    vector_store = _build_vector_store()
    scored_documents = vector_store.similarity_search_with_score(message, k=3)
    retrieved_documents = [document for document, _score in scored_documents]
    context = _format_context(retrieved_documents)
    retrieved_contexts = _build_retrieved_contexts(scored_documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nQuestion:\n{message}\n\nKnowledge base context:\n{context}",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    chain = prompt | model
    response = await with_retry(
        lambda: chain.ainvoke({"message": message, "context": context, "chat_history": chat_history})
    )
    sources = sorted({document.metadata.get("source", "unknown") for document in retrieved_documents})
    return content_to_text(response.content), sources, retrieved_contexts
