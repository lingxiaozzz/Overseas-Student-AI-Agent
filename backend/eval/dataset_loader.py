"""Validation and loading for versioned evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


DatasetSuite = Literal["route", "task", "rag"]


def load_cases(path_value: str, *, suite: DatasetSuite) -> list[dict[str, Any]]:
    """Load a dataset without silently accepting malformed evaluation cases."""
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"Dataset file does not exist: {path}")

    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset is not valid JSON: {path}: {exc.msg}") from exc

    if not isinstance(dataset, dict):
        raise ValueError("Dataset root must be a JSON object.")
    if dataset.get("schema_version") != 1:
        raise ValueError("Dataset schema_version must be 1.")
    if dataset.get("suite") != suite:
        raise ValueError(f"Dataset suite must be '{suite}'.")

    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain a non-empty 'cases' list.")

    case_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case {index} needs a non-empty string id.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        case_ids.add(case_id)

        if suite == "route":
            has_message = isinstance(case.get("message"), str) and bool(case["message"].strip())
            has_turns = isinstance(case.get("turns"), list) and bool(case["turns"])
            if not has_message and not has_turns:
                raise ValueError(f"Route case '{case_id}' needs message or turns.")
            if not any(key in case for key in ("expected_route", "expected_routes", "final_expected_route")):
                raise ValueError(f"Route case '{case_id}' needs an expected route.")
        elif suite == "task":
            if not isinstance(case.get("message"), str) or not case["message"].strip():
                raise ValueError(f"Task case '{case_id}' needs a non-empty message.")
            if "expected_final_route" not in case and "expected_routes_any" not in case:
                raise ValueError(f"Task case '{case_id}' needs an expected route.")
        else:
            if not isinstance(case.get("message"), str) or not case["message"].strip():
                raise ValueError(f"RAG case '{case_id}' needs a non-empty message.")
            sources = case.get("relevant_sources")
            if not isinstance(sources, list) or not all(isinstance(item, str) and item for item in sources):
                raise ValueError(f"RAG case '{case_id}' needs non-empty relevant_sources.")

    return cases
