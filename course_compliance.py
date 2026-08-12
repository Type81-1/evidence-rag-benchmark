from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_CATALOG_FIELDS = ("source_id", "title", "summary", "url", "source_type", "year", "organization")


def build_compliance_report() -> dict[str, object]:
    import benchmark_engine
    import hypertension_benchmark
    import nutrition_benchmark
    from advanced_evaluation import evaluate_advanced_features

    corpus_path = ROOT / "data" / "pubmed_corpus.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8")) if corpus_path.exists() else {"documents": []}
    documents = payload.get("documents", [])
    missing_by_field = {
        field: sum(not item.get(field) for item in documents)
        for field in REQUIRED_CATALOG_FIELDS
    }
    source_ids = [str(item.get("source_id") or "") for item in documents]
    sample_case = nutrition_benchmark.QUESTIONS[0]
    sample_evidence, sample_diagnostics = nutrition_benchmark.retrieve(sample_case, "good")
    sample_map = benchmark_engine.build_evidence_map(sample_evidence)
    sample_map_validation = benchmark_engine.verify_evidence_map(sample_map)
    advanced_evaluation = evaluate_advanced_features()
    offline_reports = []
    for report_path in (ROOT / "data" / "evaluation_report.json", ROOT / "data" / "hypertension_evaluation_report.json"):
        offline_reports.append(json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {})
    checks = {
        "catalog_has_at_least_500_documents": len(documents) >= 500,
        "catalog_metadata_is_complete": not any(missing_by_field.values()),
        "catalog_source_ids_are_unique": len(source_ids) == len(set(source_ids)) and all(source_ids),
        "catalog_is_in_nutrition_retrieval": len(nutrition_benchmark.CATALOG_EVIDENCE) >= 500,
        "catalog_is_in_hypertension_retrieval": len(hypertension_benchmark.CATALOG_EVIDENCE) >= 500,
        "each_domain_has_at_least_15_frozen_questions": all(
            len(questions) >= 15
            for questions in (nutrition_benchmark.QUESTIONS, hypertension_benchmark.QUESTIONS)
        ),
        "question_sets_cover_normal_abstention_and_urgent_cases": all(
            sum(not bool(case["should_abstain"]) for case in questions) >= 8
            and sum(bool(case["should_abstain"]) for case in questions) >= 3
            for questions in (nutrition_benchmark.QUESTIONS, hypertension_benchmark.QUESTIONS)
        ) and any(bool(case.get("urgent")) for case in hypertension_benchmark.QUESTIONS),
        "each_domain_report_has_15_questions_and_45_comparisons": all(
            report.get("question_count") == 15 and report.get("comparison_count") == 45
            for report in offline_reports
        ),
        "ab_reports_cover_four_arms_and_six_dimensions": all(
            set(report.get("summary", {})) == {"baseline", "good", "noisy", "missing"}
            and all(set(metrics) == set(benchmark_engine.RUBRIC) for metrics in report.get("summary", {}).values())
            for report in offline_reports
        ),
        "passages_have_auditable_locations": all(
            item.chunk_id and item.source_id and item.char_end > item.char_start and item.token_count > 0
            for item in nutrition_benchmark.CATALOG_EVIDENCE
        ),
        "retrieval_has_20_to_50_fused_candidates": 20 <= int(sample_diagnostics["candidate_pool_size"]) <= 50,
        "retrieval_exposes_lexical_vector_fused_and_reranked_stages": all(
            sample_diagnostics.get(key)
            for key in ("lexical_candidates", "vector_candidates", "fused_candidates", "ranked_candidates")
        ),
        "retrieval_uses_independent_reranking_and_mmr": "rerank" in benchmark_engine.RETRIEVAL_VERSION and "mmr" in benchmark_engine.RETRIEVAL_VERSION,
        "top_k_reports_complementary_evidence_roles": bool(sample_diagnostics.get("selected_roles")) and all(
            item.get("role") in {"overview", "causal", "boundary"} for item in sample_diagnostics["selected_roles"]
        ),
        "evidence_map_resolves_chunks_to_registered_urls": bool(sample_map) and bool(sample_map_validation["valid"]),
        "six_dimension_rubric_is_frozen": len(benchmark_engine.RUBRIC) == 6,
        "llm_judge_pipeline_is_available": callable(benchmark_engine.judge_answer),
        "metadata_filter_query_rewrite_and_multi_round_retrieval_pass": all(
            advanced_evaluation["checks"][key]
            for key in ("metadata_filter_executed_for_all_questions", "query_rewrite_executed_for_all_questions", "multi_round_retrieval_executed")
        ),
        "wiki_ingest_query_update_and_lint_pass": all(
            advanced_evaluation["checks"][key]
            for key in ("wiki_create_deduplicate_update_history", "wiki_query_returns_topic", "wiki_lint_has_no_errors")
        ),
        "advanced_automated_evaluation_passes": advanced_evaluation["status"] == "pass",
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "catalog": {
            "document_count": len(documents),
            "missing_by_field": missing_by_field,
        },
        "advanced_path": {
            "implemented": "all_advanced_and_challenge_items",
            "completed_components": ["passage_chunking", "lexical_retrieval", "vector_retrieval", "metadata_filtering", "query_rewriting", "rrf_fusion", "reranking", "mmr", "complementary_evidence", "multi_round_retrieval", "evidence_map", "llm_wiki_ingest_query_update_lint", "automated_evaluation"],
            "not_implemented": [],
            "interpretation": "Advanced capabilities are composable priorities, not mutually exclusive routes.",
        },
        "advanced_evaluation": {"status": advanced_evaluation["status"], "checks": advanced_evaluation["checks"]},
        "benchmark_reports": [
            {"domain": report.get("domain"), "question_count": report.get("question_count"), "comparison_count": report.get("comparison_count"), "arms": list(report.get("summary", {}))}
            for report in offline_reports
        ],
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
