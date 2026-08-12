from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hypertension_benchmark
import nutrition_benchmark
from wiki_engine import build_topic_page, ingest_topic, lint_wiki, query_wiki


ROOT = Path(__file__).resolve().parent


def evaluate_advanced_features() -> dict[str, object]:
    retrieval_rows: list[dict[str, object]] = []
    top3_hits = 0
    total = 0
    for module in (nutrition_benchmark, hypertension_benchmark):
        for case in module.QUESTIONS:
            evidence, diagnostics = module.retrieve(case, "good")
            sources = {item.source_id or item.id for item in evidence}
            hit = bool(sources & set(case["relevant_evidence_ids"]))
            top3_hits += int(hit)
            total += 1
            retrieval_rows.append(
                {
                    "question_id": case["id"],
                    "metadata_filter": diagnostics["metadata_filter"],
                    "query_rewrite": diagnostics["query_rewrite"],
                    "retrieval_rounds": diagnostics["retrieval_rounds"],
                    "top3_source_ids": sorted(sources),
                    "gold_source_hit": hit,
                }
            )

    with tempfile.TemporaryDirectory() as directory:
        wiki_path = Path(directory) / "wiki.json"
        case = nutrition_benchmark.QUESTIONS[0]
        result = nutrition_benchmark.compare_question(str(case["id"]), "good")
        selected_ids = {item["chunk_id"] for item in result["rag"]["evidence"]}
        evidence = []
        for item in nutrition_benchmark.EVIDENCE:
            chunks = [item] if item.chunk_id else __import__("benchmark_engine").passage_evidence(item)
            evidence.extend(chunk for chunk in chunks if chunk.chunk_id in selected_ids)
        page = build_topic_page(str(case["topic"]), str(case["question"]), str(result["rag"]["answer"]), evidence, "nutrition")
        first_ingest = ingest_topic(page, wiki_path)
        unchanged_ingest = ingest_topic(page, wiki_path)
        updated_page = {**page, "content": f"{page['content']}\n\n更新说明：自动化评估版本。", "content_hash": f"{page['content_hash']}-updated"}
        updated_ingest = ingest_topic(updated_page, wiki_path)
        wiki_results = query_wiki("限钠 高血压", wiki_path)
        wiki_lint = lint_wiki(wiki_path)

    checks = {
        "metadata_filter_executed_for_all_questions": all(row["metadata_filter"]["output_count"] > 0 for row in retrieval_rows),
        "query_rewrite_executed_for_all_questions": all(row["query_rewrite"]["rewritten"] != row["query_rewrite"]["original"] for row in retrieval_rows),
        "multi_round_retrieval_executed": any(len(row["retrieval_rounds"]) > 1 for row in retrieval_rows),
        "top3_gold_source_hit_rate_is_100_percent": top3_hits == total,
        "wiki_create_deduplicate_update_history": [first_ingest["action"], unchanged_ingest["action"], updated_ingest["action"]] == ["created", "unchanged", "updated"] and updated_ingest["history_count"] == 1,
        "wiki_query_returns_topic": bool(wiki_results) and wiki_results[0]["slug"] == page["slug"],
        "wiki_lint_has_no_errors": bool(wiki_lint["valid"]),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "retrieval": {"question_count": total, "top3_hit_count": top3_hits, "rows": retrieval_rows},
        "wiki": {"first_ingest": first_ingest, "unchanged_ingest": unchanged_ingest, "updated_ingest": updated_ingest, "query_result_count": len(wiki_results), "lint": wiki_lint},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行全部进阶与挑战能力验收")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "advanced_evaluation_report.json")
    args = parser.parse_args()
    report = evaluate_advanced_features()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
