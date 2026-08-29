"""Evaluate retrieval quality from the public /rag-chat response contract."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from cache_metrics import fetch_cache_metrics, persist_cache_run
from dataset_loader import load_cases


DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "rag_cases.json"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_reports" / "rag"


def post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc


def _ranked_sources(response: dict[str, Any]) -> list[str]:
    contexts = response.get("retrieved_contexts") or []
    ranked = sorted(contexts, key=lambda item: int(item.get("rank", 0)))
    return [str(item.get("source", "")) for item in ranked if item.get("source")]


def evaluate_case(case: dict[str, Any], response: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    expected = set(case["relevant_sources"])
    ranked_sources = _ranked_sources(response)
    retrieved_at_k = ranked_sources[:top_k]
    first_rank = next((index for index, source in enumerate(ranked_sources, start=1) if source in expected), None)
    source_metadata = set(str(source) for source in response.get("sources", []))
    answer = str(response.get("answer", "")).strip()
    citation_indices = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    response_sources = [str(source) for source in response.get("sources", [])]
    cited_sources = {
        response_sources[index - 1]
        for index in citation_indices
        if 1 <= index <= len(response_sources)
    }

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "message": case["message"],
        "expected_sources": sorted(expected),
        "retrieved_sources": ranked_sources,
        "retrieved_sources_at_k": retrieved_at_k,
        "recall_at_k": bool(expected.intersection(retrieved_at_k)),
        "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
        "source_metadata_coverage": bool(expected.intersection(source_metadata)),
        "answer_has_citations": bool(citation_indices),
        "citation_mapping_valid": bool(citation_indices)
        and all(1 <= index <= len(response_sources) for index in citation_indices),
        "relevant_source_cited": bool(expected.intersection(cited_sources)),
        "non_empty_answer": bool(answer),
        "answer_preview": " ".join(answer.split())[:180],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /rag-chat retrieval Recall@K and MRR.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases-file", default=str(DEFAULT_DATASET))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory for RAG JSON reports. Defaults to data/eval_reports/rag.",
    )
    parser.add_argument("--output-prefix", default="rag-eval")
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated RAG case ids for a targeted run. Does not overwrite rag-latest.json.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    cases = load_cases(args.cases_file, suite="rag")
    selected_case_ids = [case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()]
    if selected_case_ids:
        available_cases = {str(case["id"]): case for case in cases}
        missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in available_cases]
        if missing_case_ids:
            raise SystemExit(f"Unknown RAG case id(s): {', '.join(missing_case_ids)}")
        cases = [available_cases[case_id] for case_id in selected_case_ids]
    cache_metrics_before = fetch_cache_metrics(args.base_url)
    endpoint = f"{args.base_url.rstrip('/')}/rag-chat"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Running {case['id']}...")
        try:
            response = post_json(endpoint, {"message": case["message"], "session_id": f"rag-eval-{case['id']}"}, timeout=args.timeout)
            results.append(evaluate_case(case, response, top_k=args.top_k))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append({"id": case["id"], "error": str(exc)})

    recall_at_k = mean([float(item["recall_at_k"]) for item in results]) if results else 0.0
    mrr = mean([item["reciprocal_rank"] for item in results]) if results else 0.0
    source_coverage = mean([float(item["source_metadata_coverage"]) for item in results]) if results else 0.0
    citation_rate = mean([float(item["answer_has_citations"]) for item in results]) if results else 0.0
    citation_mapping_validity = mean([float(item["citation_mapping_valid"]) for item in results]) if results else 0.0
    relevant_source_cited_rate = mean([float(item["relevant_source_cited"]) for item in results]) if results else 0.0
    answer_rate = mean([float(item["non_empty_answer"]) for item in results]) if results else 0.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    report = {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "base_url": args.base_url,
        "dataset_file": args.cases_file,
        "selected_case_ids": selected_case_ids or None,
        "top_k": args.top_k,
        "total_cases": len(cases),
        "completed_cases": len(results),
        "request_errors": errors,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "source_metadata_coverage": source_coverage,
        "answer_citation_rate": citation_rate,
        "citation_mapping_validity": citation_mapping_validity,
        "relevant_source_cited_rate": relevant_source_cited_rate,
        "non_empty_answer_rate": answer_rate,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.output_prefix}-{timestamp}.json"
    latest_path = output_dir / ("rag-targeted-latest.json" if selected_case_ids else "rag-latest.json")
    payload = json.dumps(report, indent=2)
    report_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    cache_report_path = persist_cache_run(
        suite="rag-targeted" if selected_case_ids else "rag",
        base_url=args.base_url,
        before=cache_metrics_before,
        after=fetch_cache_metrics(args.base_url),
    )

    print("\nRAG Evaluation Summary")
    print(f"- Recall@{args.top_k}: {recall_at_k:.2%}")
    print(f"- MRR: {mrr:.3f}")
    print(f"- Source metadata coverage: {source_coverage:.2%}")
    print(f"- Answer citation rate: {citation_rate:.2%}")
    print(f"- Citation mapping validity: {citation_mapping_validity:.2%}")
    print(f"- Relevant source cited: {relevant_source_cited_rate:.2%}")
    print(f"- Non-empty answer rate: {answer_rate:.2%}")
    print(f"- Request errors: {len(errors)}")
    print(f"Report saved: {report_path}")
    if cache_report_path:
        print(f"Cache summary saved: {cache_report_path}")


if __name__ == "__main__":
    main()
