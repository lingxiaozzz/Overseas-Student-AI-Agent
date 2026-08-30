# Overseas-Student-AI-Agent

**English** | [中文](./README.zh-CN.md)

A production-oriented **LLM Agent runtime** for international student support.

Built with **FastAPI + LangChain + LangGraph + hybrid FAISS/BM25 retrieval + DeepSeek**, featuring:

- Hierarchical planning + dynamic Observation→Action loop
- Observation-Action environment abstraction
- Layered memory (working / long-term / experience) with read/write traces
- World knowledge via RAG (separate from agent memory)
- Language-aware hybrid RAG: Chinese queries prefer `*.zh-CN.md`, with cross-language fallback and source-level re-ranking
- LLM-as-judge reflection with guarded fallback
- Tool calling + RAG + explainable routing
- Trace-level observability, DeepSeek prefix-cache metrics, and versioned evaluation suites

## Verified Evaluation Results

Latest local benchmark run (2026-08-30):

| Suite | Coverage | Result |
|---|---:|---|
| RAG | Retrieval + citation checks | Recall@3 **100%**, source metadata **100%**, citation validity **100%**, relevant-source citation **100%** |
| Route | 38 cases / 45 turns | Strict **100%**, lenient **100%**, final-route **100%**, context/safety/ambiguity **100%** |
| Task | 61 end-to-end tasks | Task success **100%**, no request errors, reflection finish **100%**, evaluator pass **93.44%** |

The task suite covers single and multi-intent requests, RAG→tool decomposition, context-only updates, safety conflicts, Chinese/mixed-language requests, ambiguity, and edge cases. Generated JSON reports are stored under `data/eval_reports/` and are intentionally gitignored.

## Architecture

### High-level system

```mermaid
flowchart LR
    Client[Client / Demo / Eval] --> API[FastAPI /agent-chat]
    API --> Mem[Memory Layers]
    API --> Graph[LangGraph Agent Runtime]
    Graph --> Env[Environment Abstraction]
    Env --> Chat[Chat Backend]
    Env --> RAG[RAG + FAISS]
    Env --> Tools[Tool Calling]
    Graph --> Reflect[LLM Reflection Judge]
    Reflect --> Mem
    RAG --> KB[(Knowledge Base)]
    Mem --> Work[Working Memory]
    Mem --> LT[(Long-term JSON)]
    Mem --> Exp[(Experience JSON)]
```

### Agent decision loop

```mermaid
flowchart TD
    Start[User Message + session_id] --> Plan[Plan\nSoft subgoal hints]
    Plan --> Act[Act\nObserve then choose next Action]
    Act --> Execute[Execute\nenv.step Action]
    Execute --> Reflect[Reflect\nLLM-as-judge]
    Reflect -->|continue| Act
    Reflect -->|replan| Replan[Replan remaining work]
    Replan --> Act
    Reflect -->|finish| Evaluate[Evaluate\nFinal-answer scorer]
    Evaluate -->|pass| Finalize[Finalize answer + metrics]
    Evaluate -->|fail once| Replan
    Evaluate -->|fail after replan| Finalize
    Finalize --> End[API Response]
```

### Memory layers

| Layer | Role | Storage |
|---|---|---|
| Working memory | Recent conversation turns | In-process by `session_id` |
| Long-term memory | Durable student profile/constraints | `data/memory/long_term.json` |
| Experience memory | Task lessons for future planning | `data/memory/experiences.json` |
| World knowledge | Policies/checklists/facts | `data/knowledge_base` + FAISS |

## Project Structure

```text
backend/
  app/
    main.py            # FastAPI application entry point
    api/schemas.py     # Request/response models
    agent/             # LangGraph runtime, environment, direct chat
    rag/               # FAISS retrieval, official fetch, MCP server
    memory/            # Working, long-term, experience memory
    tools/             # Internal tools (budget, checklist)
    evaluation/        # Final-answer scoring
    core/              # Settings, LLM, retry, logging, prompts
    utils/             # Shared content utilities
  demo/
    agent_demo.py      # Demo runner
  eval/
    route_eval.py      # Route accuracy evaluation
    task_eval.py       # Task success / steps / tools evaluation
    rag_eval.py        # Retrieval Recall@K / MRR evaluation
    run_eval.py        # Unified benchmark entry point
data/
  knowledge_base/      # World knowledge markdown, including Chinese `*.zh-CN.md` documents
  memory/              # Long-term + experience memory artifacts
  eval_reports/        # Gitignored RAG / route / task / cache reports
```

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root (copy from `.env.example`):

```text
LLM_PROVIDER=deepseek
GOOGLE_API_KEY=your_real_google_ai_studio_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
DEEPSEEK_API_KEY=your_real_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_THINKING=false
MEMORY_MAX_TURNS=6
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_SECONDS=1.0
RETRY_MAX_SECONDS=8.0
LOG_LEVEL=INFO
MAX_PLAN_STEPS=4
MAX_AGENT_STEPS=6
MAX_TOOL_CALLS=3
MAX_AGENT_RUNTIME_SECONDS=90
EXPERIENCE_MEMORY_MAX_ITEMS=200
EXPERIENCE_MEMORY_TOP_K=3
EXPERIENCE_MEMORY_MIN_SCORE=0.2
EXPERIENCE_MEMORY_ENABLED=true
LONG_TERM_MEMORY_TTL_DAYS=180
EVALUATION_PASS_SCORE=0.6
OFFICIAL_FETCH_ENABLED=true
OFFICIAL_FETCH_TIMEOUT_SECONDS=8.0
OFFICIAL_FETCH_MAX_CHARS=4000
OFFICIAL_FETCH_MAX_PAGES=2
```

API key: https://aistudio.google.com/app/apikey

Default chat model is DeepSeek (`deepseek-v4-flash`), with thinking disabled (`DEEPSEEK_THINKING=false`) to keep cost down. Conversation history is sent as separate turns so DeepSeek prefix cache can hit. Gemini (`gemini-2.5-flash`) is optional: set `LLM_PROVIDER=gemini` or pass `"llm": "gemini"` / `"model": "gemini-2.5-flash"` in the request. `deepseek-chat` is still supported. RAG embeddings still use Gemini, so `GOOGLE_API_KEY` is still required for `/rag-chat` and agent retrieval.

Policy RAG also fetches allowlisted official pages (Home Affairs / Study Australia / USYD / PrivateHealth). Budget and checklist stay internal tools, not MCP. Official Fetch MCP:

```powershell
cd backend
python -m app.rag.mcp_official_fetch
```

Cursor is configured via `.cursor/mcp.json`. Tools: `list_official_sources`, `fetch_official_page`. Non-allowlisted URLs are rejected.

## Run the API

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

Main endpoint: `POST /agent-chat`

Also available:

- `POST /chat`
- `POST /rag-chat`
- `POST /tool-chat`
- `GET /health`

## 60-second Demo

Start the API, then run:

```powershell
cd backend
python demo/agent_demo.py --base-url http://127.0.0.1:8000
```

The demo walks through 3 scenarios:

1. **RAG**: USYD pre-arrival checklist  
2. **Tool**: weekly budget estimation  
3. **Multi-step**: arrival prep + budget in one request  

It prints for each case:

- `trace_id`
- plan / subgoals
- step routes and rewards
- reflection (`judge_source`, `goal_achieved`, lesson)
- evaluation (`score`, `passed`, `feedback`, `triggered_replan`)
- metrics (`steps_used`, `tool_calls`, `memory_hits`, rewards)
- environment action space

Optional flags:

```powershell
python demo/agent_demo.py --base-url http://127.0.0.1:8000 --session-id demo-interview-001
python demo/agent_demo.py --persist-experience false
```

### Manual demo request

```json
{
  "message": "Help me prepare for USYD arrival and estimate weekly budget if rent is 420 AUD.",
  "session_id": "demo-001"
}
```

Useful headers:

- `x-trace-id: demo-multi-001`
- `x-persist-experience: false` (for eval/demo without writing memory)

## Agent Response Highlights

`/agent-chat` returns:

- `plan`: goal + soft subgoal hints
- `steps`: per-step route/action/reward/tools (actual executed actions)
- `last_observation` / `last_action_decision`: Observation→Action transparency
- `reflection`: LLM judge result (`continue/replan/finish`, lesson, `goal_achieved`)
- `evaluation`: final-answer score/pass, feedback, and whether replan was triggered
- `metrics`: steps, tool calls, replan flag, memory hits, rewards
- `budget`: global step/tool/runtime limits, remaining capacity, and any stop reason
- `memory_lessons`: retrieved experience lessons
- `memory_reads` / `memory_writes`: Working / Long-term / Experience access trace
- `long_term_facts`: long-term facts loaded for this turn
- `environment`: `{ name, action_space }`
- `sources` / `retrieved_contexts`: RAG explainability
- RAG answers: inline `[n]` citations plus a deterministic `Sources` mapping
- `used_tools`: executed tools

## Core Capabilities

### Hierarchical planning + Observation→Action loop
- Runtime: `plan -> act -> execute -> reflect -> evaluate (-> replan) -> finalize`
- Planner emits soft subgoal hints (not a hard-locked execution queue)
- Each `act` step observes the environment, then chooses the next Action (`chat`/`rag`/`tool` + content)
- Cap with `MAX_PLAN_STEPS`; response exposes `last_observation` and `last_action_decision`
- Global execution budgets cap total steps across replans, tool calls, and elapsed runtime; response exposes `budget.stop_reason`

### Environment abstraction
- Unified Observation-Action interface in `app/agent/environment.py`
- Current adapter: `student_support` (`chat` / `rag` / `tool`)
- Step rewards exposed as `last_reward` / `total_reward`

### Reflection
- LLM-as-judge for progress and next action
- Hard guards + rule fallback for reliability
- Actionable lessons written into experience memory

### Final-answer evaluation
- Dedicated evaluator scores the composed answer (`EVALUATION_PASS_SCORE`)
- Safety refusals and background-only turns are evaluated against their correct intent, rather than against an unsafe or nonexistent requested action
- Fail once triggers replan (`evaluation.triggered_replan=true`, `metrics.replanned=true`)
- Second failure finalizes with score/feedback instead of infinite loops

### Memory (3 layers)
- **Working memory**: short-term session turns (`MEMORY_MAX_TURNS`)
- **Long-term memory**: durable student profile/constraints (`data/memory/long_term.json`)
- Long-term facts store `key`, `value`, `confidence`, `status`, and timestamps; conflicting profile values supersede old facts and stale records are excluded after the configured TTL
- **Experience memory**: reusable strategy lessons (`data/memory/experiences.json`)
- Each `/agent-chat` turn returns `memory_reads` + `memory_writes` with layer/status/count/items
- World knowledge remains separate via FAISS RAG
- Eval/demo sessions can disable durable writes (`x-persist-experience: false`)

### Observability
- Structured logs with `trace_id`
- Set `LOG_LEVEL=TRACE` for routing/planning/reflection traces
- DeepSeek prompt-cache usage is exposed at `GET /metrics/llm-cache`; each eval run persists one cache summary under `data/eval_reports/cache/`

### Reliability
- Exponential backoff retries for transient model/API errors

## Reproduce Evaluation

```powershell
cd backend
.\.venv\Scripts\python.exe -m eval.route_eval
.\.venv\Scripts\python.exe -m eval.task_eval
.\.venv\Scripts\python.exe -m eval.rag_eval

# Run focused task regressions without overwriting task-latest.json
.\.venv\Scripts\python.exe -m eval.task_eval `
  --case-ids task-tool-checklist,task-multi-settling-checklist-grounded `
  --output-prefix task-targeted

# Continue a route run after a transient API failure
.\.venv\Scripts\python.exe -m eval.route_eval --from-case ambiguous-5
```

Reports are written to `data/eval_reports/{rag,route,task,cache}/`. Task evaluation records request failures and continues by default; use `--no-continue-on-error` when a failure should stop the run.

### Historical optimization notes (superseded)

The tables below document earlier iterations. They are retained for the development story only; use **Verified Evaluation Results** above for the current benchmark.

<details>
<summary>Show historical optimisation iterations</summary>

| Group | Suite | Before → After | Reports (route / task) |
|---|---|---|---|
| **A** Original (small) | 12 route / 7 task | Baseline → Opt-1 | `084259Z`→`102435Z` / `085613Z`→`102333Z` |
| **B** Expanded | 38 route / 23 task | Expanded → Opt-2 (P0/P1) | `122242Z`→`132528Z` / `105801Z`→`131442Z` |

Both are **evaluation case suites** (not training datasets). Do not compare absolute scores across groups — suite difficulty differs.

---

### Group A — Original eval set (small, Opt-1)

Opt-1: primary-route finalize; Act guards (context / safety / budget); single-step chat; Reflect early finish; replan observability.

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Route strict / lenient | 57.14% / 64.29% | **92.86% / 100%** | +35.7 / +35.7pp |
| Final-route / context / safety | 50% / 0% / 0% | **100% / 100% / 100%** | +50 / +100 / +100pp |
| Ambiguity precision | 50% | 50% | 0 |
| Task success | 28.57% | **85.71%** | +57.1pp |
| Avg steps / replan | 2.14 / 42.86% | **1.43 / 28.57%** | −0.71 / −14.3pp |

Category (route strict | task success): multi-turn 25%→**100%**; adversarial 0%→**100%**; edge 75%→**100%**; context/safety/ambiguous tasks 0%→**100%**; single-intent task 33%→**67%**.

---

### Group B — Expanded eval set (Opt-2)

Opt-2: `_requires_checklist_tool()`; `_is_pure_chat_message()` + single-step finish; stronger context-only guards.

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Route strict / lenient | 73.33% / 77.78% | **91.11% / 93.33%** | +17.8 / +15.6pp |
| Final-route / context / ambiguity | 42.86% / 66.67% / 40% | **85.71% / 100% / 60%** | +42.9 / +33.3 / +20pp |
| Safety / adversarial | 100% / 100% | **75% / 75%** | −25 / −25pp |
| Task success / failures | 73.91% / 6 | **100% / 0** | +26.1pp / −6 |
| Avg steps / replan | 1.57 / 30.43% | **1.30 / 17.39%** | −0.27 / −13.0pp |

Route categories after Opt-2: single/edge **100%**, multi-turn **93%**, ambiguous **60%**, adversarial **75%**. Task categories all **100%**.

Safety drop note: new mixed adversarial case `adv-4` exposed a refusal-path hole — malicious “ignore tools/RAG” prompts degraded to `chat` instead of compliant `rag`. Pre-existing safety cases still all pass; the dip is from this new boundary case, not a regression on the original suite.

---

### Remaining gaps (unified)

| Issue | Cases | Note |
|---|---|---|
| Budget guard too aggressive | `ambiguous-3`, `multi-mixed-1` | Mixed arrival+rent forced to `tool`, drops preferred `rag` |
| Safety refusal → chat | `adv-4` | New boundary case; original safety cases still pass |
| Full-path timeout | `ambiguous-5` | Timeout only on complete `/agent-chat` multi-step loop (>180s). Single-turn rag/tool themselves are not slow — not a routing-logic defect |

### Summary

- **A:** Opt-1 → route **57%→93%**, task **29%→86%** (context, safety, final-route, fewer steps).
- **B:** Expansion exposed checklist/chat/context gaps; Opt-2 → route **73%→91%**, task **74%→100%** (checklist routing and chat over-planning fixed).
- Strengths: RAG policy Qs, explicit budget tools, reflection finish **100%**.

### Next improvements

| Priority | Area | Action |
|---|---|---|
| <span style="color:#c00"><strong>P0</strong></span> | Mixed-intent routing | Force budget/checklist only for explicit single-intent requests; prefer retrieval-first when arrival/orientation co-occurs |
| <span style="color:#c00"><strong>P0</strong></span> | Adversarial safety | Keep refusals / prompt-injection on forced `rag`; never degrade to `chat` |
| **P1** | Latency | Shorten `ambiguous-5`-style multi-step paths (timeout is full-agent only) |
| **P2** | Eval hygiene | Keep `--continue-on-error`; report small vs expanded eval sets separately |

</details>
