# Overseas-Student-AI-Agent
A production-ready AI agent system for international student support, integrating LLM-based reasoning, retrieval-augmented generation (RAG), and tool calling.  Features include multi-step planning, long/short-term memory, vector search, and external API integration.  Built with LangChain, FAISS, FastAPI, and modern full-stack technologies.

## Step 1: Minimal LangChain Chat API

This first version is intentionally small:

- `FastAPI` provides the web API.
- `LangChain` connects the app to a Gemini chat model.
- `/chat` accepts a student question and returns an AI answer.

Later steps will add RAG, tools, memory, and LangGraph.

## Project Structure

```text
backend/
  app/
    main.py          # FastAPI app and API routes
    chat_service.py  # LangChain chat chain
    config.py        # Environment variable loading
    schemas.py       # Request and response models
  requirements.txt
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
