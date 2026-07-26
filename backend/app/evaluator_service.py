from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.chat_service import MissingApiKeyError
from app.config import settings
from app.retry_service import with_retry

EVALUATOR_PROMPT = """You are a strict final-answer evaluator for an international student assistant.
Score whether the candidate answer fully satisfies the user goal.

Scoring guide:
- 1.0: complete, actionable, and aligned with the goal
- 0.7-0.9: mostly good with minor gaps
- 0.4-0.6: partial answer, missing key parts
- 0.0-0.3: irrelevant, empty, hallucinated, or fails the goal

Rules:
1) If the goal requires budget calculation and no numeric estimate is present, score <= 0.4.
2) If the goal requires USYD/policy facts and answer is too generic, score <= 0.5.
3) Prefer lower scores when critical constraints in the user message are ignored.

Return strict JSON with:
- score: float between 0 and 1
- passed: boolean
- feedback: one short sentence explaining the score
"""


class EvaluationDecision(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: str = ""


def compose_candidate_answer(step_results: list[dict]) -> str:
    if not step_results:
        return ""
    if len(step_results) == 1:
        return str(step_results[0].get("answer", "")).strip()
    parts = [
        f"### Step {item.get('step_index')}: {item.get('subgoal')}\n{item.get('answer', '')}"
        for item in step_results
    ]
    return "\n\n".join(parts).strip()


def rule_evaluate(
    goal: str,
    answer: str,
    *,
    used_tools: list[str] | None = None,
    sources: list[str] | None = None,
) -> EvaluationDecision:
    used_tools = used_tools or []
    sources = sources or []
    text = answer.strip()
    goal_lower = goal.lower()

    if not text:
        return EvaluationDecision(score=0.0, passed=False, feedback="Final answer is empty.")

    score = 0.7
    feedback = "Answer is non-empty and usable."

    needs_budget = any(token in goal_lower for token in ("budget", "rent", "cost", "estimate"))
    needs_policy = any(token in goal_lower for token in ("usyd", "arrival", "visa", "oshc", "checklist"))

    if needs_budget:
        has_number = any(char.isdigit() for char in text)
        if "estimate_weekly_budget" in used_tools and has_number:
            score = 0.9
            feedback = "Budget goal covered with tool-backed numeric estimate."
        elif has_number:
            score = 0.65
            feedback = "Budget numbers present but tool usage is missing."
        else:
            score = 0.3
            feedback = "Budget goal lacks numeric estimate."

    if needs_policy:
        if sources:
            score = max(score, 0.85)
            feedback = "Policy/onboarding goal grounded with retrieved sources."
        elif score >= 0.7:
            score = 0.6
            feedback = "Policy/onboarding answer is generic without sources."

    passed = score >= settings.evaluation_pass_score
    return EvaluationDecision(score=score, passed=passed, feedback=feedback)


async def llm_evaluate(goal: str, answer: str, plan_summary: str = "") -> EvaluationDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVALUATOR_PROMPT),
            (
                "human",
                "User goal:\n{goal}\n\nPlan summary:\n{plan_summary}\n\nCandidate answer:\n{answer}",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(EvaluationDecision)
    chain = prompt | model
    decision = await with_retry(
        lambda: chain.ainvoke(
            {
                "goal": goal,
                "plan_summary": plan_summary or "n/a",
                "answer": answer,
            }
        )
    )
    # Keep threshold ownership in app settings.
    passed = decision.score >= settings.evaluation_pass_score
    return EvaluationDecision(
        score=float(decision.score),
        passed=passed,
        feedback=decision.feedback.strip() or "No feedback provided.",
    )
