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
- `/agent-chat` uses LangGraph with an LLM router (and keyword fallback) to choose between direct chat, RAG, and tool calling.
- Short-term memory keeps recent turns per `session_id` across endpoints.

Later steps will add tools, memory, and LangGraph.

## Project Structure

```text
backend/
  app/
    main.py           # FastAPI app and API routes
    chat_service.py   # LangChain chat chain
    rag_service.py    # LangChain RAG chain with FAISS
    tool_service.py   # LangChain tool calling service
    graph_service.py  # LangGraph plan-act-reflect workflow
    environment.py    # Observation-Action environment abstraction
    memory_service.py # Working + experience memory
    config.py         # Environment variable loading
    schemas.py        # Request and response models
  requirements.txt
data/
  knowledge_base/     # Local markdown files used by RAG (world knowledge)
  memory/             # Persisted experience memory artifacts
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
MEMORY_MAX_TURNS=6
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_SECONDS=1.0
RETRY_MAX_SECONDS=8.0
LOG_LEVEL=INFO
MAX_PLAN_STEPS=4
EXPERIENCE_MEMORY_MAX_ITEMS=200
EXPERIENCE_MEMORY_TOP_K=3
EXPERIENCE_MEMORY_MIN_SCORE=0.2
EXPERIENCE_MEMORY_ENABLED=true
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
  "message": "I am a new international student at USYD. What should I prepare before arrival?",
  "session_id": "student-001"
}
```

Try the `POST /rag-chat` endpoint with:

```json
{
  "message": "What should I prepare before arriving at USYD?",
  "session_id": "student-001"
}
```

The RAG response includes:

- `answer`: Gemini's answer grounded in the local knowledge base.
- `sources`: the markdown files retrieved from `data/knowledge_base`.
- `retrieved_contexts`: ranked snippets retrieved from FAISS, including source file, similarity score, and a short preview.

Try the `POST /agent-chat` endpoint with:

```json
{
  "message": "I need help with USYD arrival checklist and OSHC.",
  "session_id": "student-001"
}
```

The agent response includes:

- `route`: last-step route (`chat`, `rag`, or `tool`) from hierarchical execution.
- `router_reason`: short explanation from the last act/route decision.
- `answer`: final synthesized answer across completed plan steps.
- `plan`: overall `goal` and ordered `subgoals` from the planner.
- `steps`: per-step route, reason, tools, and answer preview.
- `reflection`: progress, next action (`continue`/`replan`/`finish`), and lesson.
- `metrics`: `steps_used`, `tool_calls`, whether replanning happened, and `memory_hits`.
- `memory_lessons`: retrieved episodic lessons used by the planner (if any).
- `environment`: environment name and action space (`chat`/`rag`/`tool`).
- `sources`: retrieved files aggregated across steps (empty for direct chat).
- `retrieved_contexts`: retrieved ranked snippets aggregated across steps.
- `used_tools`: tool names used during tool-calling steps.

Hierarchical planning behavior:

- `/agent-chat` runs `plan -> act -> execute -> reflect (-> replan) -> finalize`.
- Simple requests usually stay single-step; complex goals can expand to multiple subgoals.
- `MAX_PLAN_STEPS` caps the planning horizon (default `4`).

Environment abstraction:

- Execution goes through an Observation-Action interface (`environment.py`).
- Current adapter: `student_support` maps actions to chat/rag/tool backends.
- Each step returns reward signal (`last_reward` / `total_reward`) for future RL-style optimization.
- New environments can be plugged in via `create_environment(name)` without rewriting the planner loop.

Memory layers:

- Working memory: recent turns by `session_id` (`MEMORY_MAX_TURNS`).
- Experience memory: task lessons persisted under `data/memory/experiences.json` and retrieved for future planning.
- World knowledge: RAG knowledge base under `data/knowledge_base`.
- Tune experience retrieval with `EXPERIENCE_MEMORY_TOP_K`, `EXPERIENCE_MEMORY_MAX_ITEMS`, and `EXPERIENCE_MEMORY_MIN_SCORE`.
- Eval sessions (`route-eval*`) and `x-persist-experience: false` do not write experience memory.

Try the `POST /tool-chat` endpoint with:

```json
{
  "message": "My rent is 420 AUD per week, can you estimate my weekly budget?",
  "session_id": "student-001"
}
```

Tool chat response includes:

- `answer`: final LLM response after tool execution.
- `used_tools`: list of executed tools, such as `estimate_weekly_budget`.

Memory behavior:

- Use the same `session_id` to keep conversation continuity across `/chat`, `/rag-chat`, `/tool-chat`, and `/agent-chat`.
- Working memory keeps recent turns in process memory (`MEMORY_MAX_TURNS`).
- Experience memory persists task lessons to `data/memory/experiences.json` and injects relevant lessons into planning.

Retry behavior:

- The backend retries transient Gemini errors (e.g. 429/503) with exponential backoff.
- Control retries via `RETRY_MAX_ATTEMPTS`, `RETRY_INITIAL_SECONDS`, and `RETRY_MAX_SECONDS`.

Trace-level logging:

- The backend logs `trace_id`, selected route, tool usage, retrieval counts, and request latency.
- Set `LOG_LEVEL=TRACE` for verbose routing traces.
- Pass `x-trace-id` header to correlate client and server logs (backend auto-generates one if missing).

## Route Evaluation

You can evaluate agent route accuracy (`chat` / `rag` / `tool`) with a built-in script.
The dataset now includes:

- single-turn
- multi-turn
- ambiguous intent
- adversarial queries
- edge cases

1) Start the backend:

```powershell
cd backend
uvicorn app.main:app --reload
```

2) In another terminal, run:

```powershell
cd backend
python eval/route_eval.py --base-url http://127.0.0.1:8000
```

Optional: customize output location/prefix:

```powershell
python eval/route_eval.py --base-url http://127.0.0.1:8000 --output-dir eval/reports --output-prefix route-eval
```

The script prints:

- total test cases
- total evaluated turns (for multi-turn threads)
- route accuracy
- per-category metrics
- confusion matrix (`expected -> predicted`)
- mismatch examples with `router_reason`
- JSON report files:
  - `eval/reports/route-eval-<timestamp>.json`
  - `eval/reports/latest.json`

## Task Evaluation

Evaluate hierarchical planning quality (success rate, steps, tools, reflection):

```powershell
cd backend
python eval/task_eval.py --base-url http://127.0.0.1:8000
```

The script prints:

- task success rate
- average steps / tool calls
- replan rate
- reflection finish rate
- memory hit rate
- per-category success
- JSON report files:
  - `eval/reports/task-eval-<timestamp>.json`
  - `eval/reports/task-latest.json`
