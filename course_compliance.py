from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_CATALOG_FIELDS = ("source_id", "title", "summary", "url", "source_type", "year", "organization")


def build_compliance_report() -> dict[str, object]:
    import benchmark_engine
    import hypertension_benchmark
    import nutrition_benchmark

    corpus_path = ROOT / "data" / "pubmed_corpus.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8")) if corpus_path.exists() else {"documents": []}
    documents = payload.get("documents", [])
    missing_by_field = {
        field: sum(not item.get(field) for item in documents)
        for field in REQUIRED_CATALOG_FIELDS
    }
    source_ids = [str(item.get("source_id") or "") for item in documents]
    checks = {
        "catalog_has_at_least_500_documents": len(documents) >= 500,
        "catalog_metadata_is_complete": not any(missing_by_field.values()),
        "catalog_source_ids_are_unique": len(source_ids) == len(set(source_ids)) and all(source_ids),
        "catalog_is_in_nutrition_retrieval": len(nutrition_benchmark.CATALOG_EVIDENCE) >= 500,
        "catalog_is_in_hypertension_retrieval": len(hypertension_benchmark.CATALOG_EVIDENCE) >= 500,
        "each_domain_has_at_least_8_frozen_questions": all(
            len(questions) >= 8
            for questions in (nutrition_benchmark.QUESTIONS, hypertension_benchmark.QUESTIONS)
        ),
        "question_sets_include_abstention": all(
            any(bool(case["should_abstain"]) for case in questions)
            for questions in (nutrition_benchmark.QUESTIONS, hypertension_benchmark.QUESTIONS)
        ),
        "retrieval_uses_hybrid_rrf_and_mmr": all(
            term in benchmark_engine.RETRIEVAL_VERSION for term in ("bm25", "tfidf", "rrf", "mmr")
        ),
        "six_dimension_rubric_is_frozen": len(benchmark_engine.RUBRIC) == 6,
        "llm_judge_pipeline_is_available": callable(benchmark_engine.judge_answer),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "catalog": {
            "document_count": len(documents),
            "missing_by_field": missing_by_field,
        },
        "external_acceptance_items": [
            "Run the frozen benchmark with a real model and preregistered repeats.",
            "Collect independent blind scores from at least two qualified reviewers.",
            "Report inter-rater agreement and do not claim superiority from pipeline_demo output.",
        ],
    }


def main() -> None:
    report = build_compliance_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
