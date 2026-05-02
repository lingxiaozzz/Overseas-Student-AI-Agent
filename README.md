# Overseas-Student-AI-Agent
A production-ready AI agent system for international student support, integrating LLM-based reasoning, retrieval-augmented generation (RAG), and tool calling.  Features include multi-step planning, long/short-term memory, vector search, and external API integration.  Built with LangChain, FAISS, FastAPI, and modern full-stack technologies.

## Step 1: Minimal LangChain Chat API

This first version is intentionally small:

- `FastAPI` provides the web API.
- `LangChain` connects the app to a Gemini chat model.
- `LangGraph` routes user questions through a simple agent workflow.
- `/chat` accepts a student question and returns an AI answer.
- `/rag-chat` retrieves relevant local knowledge base documents before answering.
- `/tool-chat` enables tool calling for budgeting and checklist tasks.
- `/agent-chat` uses LangGraph to choose between direct chat and RAG.

Later steps will add tools, memory, and LangGraph.

## Project Structure

```text
backend/
  app/
    main.py          # FastAPI app and API routes
    chat_service.py  # LangChain chat chain
    rag_service.py   # LangChain RAG chain with FAISS
    tool_service.py  # LangChain tool calling service
    graph_service.py # LangGraph route -> (chat|rag|tool) workflow
    config.py        # Environment variable loading
    schemas.py       # Request and response models
  requirements.txt
data/
  knowledge_base/    # Local markdown files used by RAG
```

## Setup

From the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root by copying `.env.example`, then add your real Google AI Studio API key:

```text
GOOGLE_API_KEY=your_real_google_ai_studio_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

You can create a Gemini API key from Google AI Studio:

```text
https://aistudio.google.com/app/apikey
```

## Run the API

From the `backend` folder:

```powershell
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Try the `POST /chat` endpoint with:

```json
{
  "message": "I am a new international student at USYD. What should I prepare before arrival?"
}
```

Try the `POST /rag-chat` endpoint with:

```json
{
  "message": "What should I prepare before arriving at USYD?"
}
```

The RAG response includes:

- `answer`: Gemini's answer grounded in the local knowledge base.
- `sources`: the markdown files retrieved from `data/knowledge_base`.
- `retrieved_contexts`: ranked snippets retrieved from FAISS, including source file, similarity score, and a short preview.

Try the `POST /agent-chat` endpoint with:

```json
{
  "message": "I need help with USYD arrival checklist and OSHC."
}
```

The agent response includes:

- `route`: `chat`, `rag`, or `tool`, selected by LangGraph route node.
- `answer`: final answer from the selected route.
- `sources`: retrieved files (empty for direct chat).
- `retrieved_contexts`: retrieved ranked snippets (empty for direct chat).
- `used_tools`: tool names used during tool-calling flow (empty for non-tool routes).

Try the `POST /tool-chat` endpoint with:

```json
{
  "message": "My rent is 420 AUD per week, can you estimate my weekly budget?"
}
```

Tool chat response includes:

- `answer`: final LLM response after tool execution.
- `used_tools`: list of executed tools, such as `estimate_weekly_budget`.
