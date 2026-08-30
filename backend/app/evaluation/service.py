from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import create_structured_output_model
from app.core.prompts import cache_friendly_messages
from app.core.retry import with_retry

EVALUATOR_PROMPT = """You are a strict final-answer evaluator for an international student assistant.
Score whether the candidate answer fully satisfies the user goal.

Scoring guide:
- 1.0: complete, actionable, and aligned with the goal
- 0.7-0.9: mostly good with minor gaps
- 0.4-0.6: partial answer, missing key parts
- 0.0-0.3: irrelevant, empty, hallucinated, or fails the goal

Rules:
1) First classify the goal: normal assistance, background update / conversational turn, or safety-sensitive request.
2) For a safety-sensitive request (fraud, illegal work, policy evasion), a refusal plus lawful alternatives is correct. Never lower the score merely because the answer does not enable the harmful request.
3) For a background update or vague conversational turn with no concrete request, a concise acknowledgement or clarification is correct; do not require a calculation or checklist.
4) Treat the plan summary's routes, sources, and tools as execution evidence. Do not claim a tool or retrieval step is missing when it is listed there.
5) If a normal goal requires budget calculation and no numeric estimate is present, score <= 0.4.
6) If a normal goal requires USYD/policy facts and answer is too generic, score <= 0.5.
7) Prefer lower scores only when a legitimate, non-harmful user constraint is ignored.
8) Never invent requirements that are absent from the user goal. In particular, do not demand a budget estimate, deadline, or live web result unless the user asked for it.
9) If the plan summary records a successful build_prearrival_checklist tool call, treat the checklist requirement as met; only mark missing personalisation as a minor gap.

Return strict JSON with:
- score: float between 0 and 1
- passed: boolean
- feedback: one short sentence explaining the score
Example: {"score":0.8,"passed":true,"feedback":"The answer covers the required actions."}
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
    model = create_structured_output_model(EvaluationDecision, temperature=0)
    messages = cache_friendly_messages(
        EVALUATOR_PROMPT,
        "",
        (
            f"User goal:\n{goal}\n\n"
            f"Plan summary:\n{plan_summary or 'n/a'}\n\n"
            f"Candidate answer:\n{answer}"
        ),
    )
    decision = await with_retry(lambda: model.ainvoke(messages))
    # Keep threshold ownership in app settings.
    passed = decision.score >= settings.evaluation_pass_score
    return EvaluationDecision(
        score=float(decision.score),
        passed=passed,
        feedback=decision.feedback.strip() or "No feedback provided.",
    )
