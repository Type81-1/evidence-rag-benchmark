from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from fastapi.testclient import TestClient

import app as app_module
import benchmark_engine
import hypertension_benchmark
import nutrition_benchmark
from benchmark_engine import PROMPT_VERSION, RUBRIC, Evidence, apply_metadata_filters, bm25_rank, build_evidence_map, build_metadata_filters, build_prompt, hybrid_rank, passage_evidence, rewrite_query, score_answer, split_passages, validate_evidence_packet, verify_evidence_map
from course_compliance import build_compliance_report
from advanced_evaluation import evaluate_advanced_features
from wiki_engine import build_topic_page, ingest_topic, lint_wiki, query_wiki


client = TestClient(app_module.app)
REQUIRED_FIELDS = {"id", "question", "track", "expected_evidence_type", "notes", "should_abstain", "expected_claims", "relevant_evidence_ids"}


def test_question_sets_are_preregistered_and_include_abstention_and_adversarial_cases() -> None:
    for questions in (nutrition_benchmark.QUESTIONS, hypertension_benchmark.QUESTIONS):
        assert len(questions) >= 15
        assert all(REQUIRED_FIELDS.issubset(case) for case in questions)
        assert sum(bool(case["should_abstain"]) for case in questions) >= 3
        assert sum(not bool(case["should_abstain"]) for case in questions) >= 8
        assert any("对抗" in str(case["notes"]) or "越界" in str(case["notes"]) or case.get("urgent") for case in questions)


def test_frozen_question_manifest_can_be_exported() -> None:
    subprocess.run([sys.executable, "scripts/export_question_set.py"], check=True)
    payload = json.loads(Path("data/test_questions.json").read_text(encoding="utf-8"))
    assert payload["status"] == "frozen-before-live-evaluation"
    assert len(payload["questions"]) == 30


def test_evidence_registry_has_traceable_identifiers_and_urls() -> None:
    for corpus in (nutrition_benchmark.EVIDENCE, hypertension_benchmark.EVIDENCE):
        assert len(corpus) >= 10
        assert len({item.id for item in corpus}) == len(corpus)
        assert all(item.identifier and item.url.startswith("https://") for item in corpus)
        assert all(len(item.to_dict()["content_hash"]) == 64 for item in corpus)


def test_bm25_ranking_uses_question_text_and_returns_scores() -> None:
    ranked = bm25_rank("限钠饮食能降低血压吗", nutrition_benchmark.EVIDENCE)
    assert ranked[0][0] > 0
    assert ranked[0][1].id in {"S1", "S2"}


def test_500_document_catalog_is_integrated_with_hybrid_retrieval() -> None:
    assert len(nutrition_benchmark.CATALOG_EVIDENCE) >= 500
    assert len(hypertension_benchmark.CATALOG_EVIDENCE) >= 500
    ranked = hybrid_rank("限钠饮食能降低血压吗", nutrition_benchmark.EVIDENCE)
    assert len(ranked) == len(nutrition_benchmark.EVIDENCE)
    assert ranked[0][0] > ranked[-1][0]


def test_passage_chunking_records_source_location_and_overlap() -> None:
    text = " ".join(f"token-{index}" for index in range(900))
    passages = split_passages(text)
    assert [item[3] for item in passages] == [400, 400, 260]
    assert passages[1][1] < passages[0][2]
    evidence = Evidence("T1", "test", "Title", "Org", 2026, text, "https://example.com/t1", "RCT", "PMID:1", ("test",))
    chunks = passage_evidence(evidence)
    assert chunks[0].chunk_id == "T1-C000"
    assert chunks[1].source_id == "T1"
    assert chunks[0].char_end > chunks[0].char_start


def test_retrieval_exposes_fusion_reranking_and_evidence_map() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    evidence, diagnostics = nutrition_benchmark.retrieve(case, "good")
    assert 20 <= diagnostics["candidate_pool_size"] <= 50
    assert all(diagnostics[key] for key in ("lexical_candidates", "vector_candidates", "fused_candidates", "ranked_candidates"))
    assert diagnostics["ranked_candidates"][0]["chunk_id"]
    assert len(diagnostics["selected_roles"]) == 3
    assert all(item["role"] in {"overview", "causal", "boundary"} for item in diagnostics["selected_roles"])
    evidence_map = build_evidence_map(evidence)
    assert verify_evidence_map(evidence_map)["valid"] is True
    assert set(evidence_map) == {item.chunk_id for item in evidence}
    assert all(entry["url"].startswith("https://") for entry in evidence_map.values())


def test_metadata_filter_query_rewrite_and_multi_round_diagnostics() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    filters = build_metadata_filters(case)
    filtered, metadata = apply_metadata_filters(nutrition_benchmark.EVIDENCE, filters, minimum_pool=20)
    rewrite = rewrite_query(case)
    assert filtered
    assert metadata["output_count"] <= metadata["input_count"]
    assert metadata["filters"]["track"] == "nutrition"
    assert rewrite["rewritten"] != rewrite["original"]
    evidence, diagnostics = nutrition_benchmark.retrieve(case, "good")
    assert evidence
    assert diagnostics["metadata_filter"]["output_count"] > 0
    assert diagnostics["query_rewrite"]["strategy"]
    assert 1 <= len(diagnostics["retrieval_rounds"]) <= 2


def test_wiki_ingest_query_update_and_lint(tmp_path: Path) -> None:
    path = tmp_path / "wiki.json"
    case = nutrition_benchmark.QUESTIONS[0]
    result = nutrition_benchmark.compare_question("NUT-01", "good")
    chunk_ids = {item["chunk_id"] for item in result["rag"]["evidence"]}
    evidence = [
        chunk
        for item in nutrition_benchmark.EVIDENCE
        for chunk in ([item] if item.chunk_id else passage_evidence(item))
        if chunk.chunk_id in chunk_ids
    ]
    page = build_topic_page(str(case["topic"]), str(case["question"]), str(result["rag"]["answer"]), evidence, "nutrition")
    assert ingest_topic(page, path)["action"] == "created"
    assert ingest_topic(page, path)["action"] == "unchanged"
    updated = {**page, "content": page["content"] + "\n\n更新。", "content_hash": page["content_hash"] + "-2"}
    update_result = ingest_topic(updated, path)
    assert update_result["action"] == "updated"
    assert update_result["history_count"] == 1
    assert query_wiki("限钠", path)[0]["slug"] == page["slug"]
    assert lint_wiki(path)["valid"] is True


def test_advanced_automated_evaluation_passes() -> None:
    report = evaluate_advanced_features()
    assert report["status"] == "pass", report
    assert all(report["checks"].values())


def test_course_compliance_report_passes_static_checks() -> None:
    report = build_compliance_report()
    assert report["status"] == "pass", report


def test_degraded_retrieval_reduces_precision_without_using_empty_context() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    good_evidence, good_metrics = nutrition_benchmark.retrieve(case, "good")
    noisy_evidence, noisy_metrics = nutrition_benchmark.retrieve(case, "noisy")
    assert good_evidence and noisy_evidence
    assert noisy_metrics["precision_at_k"] <= good_metrics["precision_at_k"]
    assert sum(item["score"] for item in noisy_metrics["selected_scores"]) < sum(item["score"] for item in good_metrics["selected_scores"])
    assert {"precision_at_k", "recall_at_k", "mrr", "selected_scores", "ranked_candidates", "validation"}.issubset(noisy_metrics)


def test_fair_prompt_has_identical_instructions_and_only_packet_policy_differs() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    evidence, _ = nutrition_benchmark.retrieve(case, "good")
    baseline = build_prompt(str(case["question"]), [], False, "饮食营养")
    rag = build_prompt(str(case["question"]), evidence, True, "饮食营养")
    assert baseline.split("EVIDENCE_POLICY=")[0] == rag.split("EVIDENCE_POLICY=")[0]
    assert "EVIDENCE_POLICY=OPTIONAL" in baseline
    assert "EVIDENCE_POLICY=REQUIRED" in rag
    assert PROMPT_VERSION == "evidence-gate-v2"


def test_evidence_gate_covers_low_similarity_type_mismatch_and_boundary() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    low = validate_evidence_packet(case, [nutrition_benchmark.EVIDENCE[0]], {"top_score_ratio": 0.1})
    assert low["action"] == "abstain"
    mismatch = validate_evidence_packet(
        {**case, "expected_evidence_type": "系统综述"}, [nutrition_benchmark.EVIDENCE[0]], {"top_score_ratio": 1.0}
    )
    assert mismatch["action"] == "abstain"
    boundary_case = nutrition_benchmark.QUESTIONS[-1]
    boundary = validate_evidence_packet(boundary_case, [nutrition_benchmark.EVIDENCE[-1]], {"top_score_ratio": 1.0})
    assert boundary["action"] == "abstain"


def test_structured_refusal_explains_found_missing_and_next_search() -> None:
    result = nutrition_benchmark.compare_question("NUT-08", "good")
    assert result["rag"]["validation"]["action"] == "abstain"
    for heading in ("## 已检索", "## 缺失证据", "## 下一步"):
        assert heading in result["rag"]["answer"]


def test_offline_generation_does_not_read_expected_claims() -> None:
    original = dict(nutrition_benchmark.QUESTIONS[0])
    mutated = {**original, "expected_claims": ["绝不应泄漏的金标准词"]}
    from benchmark_engine import run_case
    first = run_case(original, nutrition_benchmark.EVIDENCE, "good", live=False, domain_label="饮食营养", safety_note=nutrition_benchmark.SAFETY_NOTE)
    second = run_case(mutated, nutrition_benchmark.EVIDENCE, "good", live=False, domain_label="饮食营养", safety_note=nutrition_benchmark.SAFETY_NOTE)
    assert first["rag"]["answer"] == second["rag"]["answer"]
    assert "绝不应泄漏" not in second["rag"]["answer"]


def test_six_dimension_rubric_and_sentence_level_citation_support() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    evidence = passage_evidence(nutrition_benchmark.CURATED_EVIDENCE[1])
    answer = "## 结论\n\n降低钠摄入可以降低血压，但存在个体差异 [S2-C000]。\n\n## 依据\n\n见随机对照试验。\n\n## 局限\n\n需结合整体饮食。\n\n## 安全提示\n\n不替代医生或营养师建议。"
    metrics = score_answer(case, answer, evidence, evidence_required=True)
    assert set(RUBRIC).issubset(metrics)
    assert metrics["citation_quality"] > 0
    assert metrics["unsupported_citation_rate"] == 0


def test_evidence_summary_saying_insufficient_does_not_make_answer_a_refusal() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    evidence = passage_evidence(nutrition_benchmark.CURATED_EVIDENCE[6])
    answer = "## 结论\n\n该问题可以基于当前登记证据谨慎回答。\n\n## 依据\n\n该摘要提到某项预防证据不足 [S7-C000]。\n\n## 局限\n\n结论仅限该问题。\n\n## 安全提示\n\n不替代医生建议。"
    metrics = score_answer(case, answer, evidence, evidence_required=True)
    assert metrics["refusal_quality"] == 1.0


def test_fake_citation_is_flagged() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    answer = "## 结论\n\n这是一个无来源结论 [X99]。\n\n## 依据\n\n未知。\n\n## 局限\n\n未知。\n\n## 安全提示\n\n不替代医生建议。"
    metrics = score_answer(case, answer, [], evidence_required=True)
    assert metrics["unsupported_citation_rate"] == 1.0
    assert "X99" in metrics["unsupported_citations"]


def test_missing_retrieval_is_an_appropriate_refusal_for_required_rag() -> None:
    result = hypertension_benchmark.compare_question("HTN-01", "missing")
    assert result["rag"]["evidence"] == []
    assert result["rag"]["metrics"]["refusal_quality"] == 1.0


def test_emergency_case_requires_escalation_for_safety() -> None:
    result = hypertension_benchmark.compare_question("HTN-07", "good")
    assert "立即就医" in result["rag"]["answer"]
    assert result["rag"]["metrics"]["safety"] == 1.0
    assert result["rag"]["metrics"]["refusal_quality"] == 1.0
    neurologic = hypertension_benchmark.compare_question("HTN-15", "good")
    assert neurologic["rag"]["validation"]["action"] == "escalate"
    assert "立即就医" in neurologic["rag"]["answer"]
    assert neurologic["rag"]["metrics"]["refusal_quality"] == 1.0


def test_new_abstention_cases_trigger_the_gate() -> None:
    device = hypertension_benchmark.compare_question("HTN-13", "good")
    menu = nutrition_benchmark.compare_question("NUT-15", "good")
    insufficient = nutrition_benchmark.compare_question("NUT-13", "good")
    assert device["rag"]["validation"]["action"] == "abstain"
    assert menu["rag"]["validation"]["action"] == "abstain"
    assert insufficient["rag"]["validation"]["action"] == "abstain"


def test_benchmark_records_metadata_and_all_arms() -> None:
    report = nutrition_benchmark.run_benchmark()
    assert report["question_count"] == 15
    assert report["comparison_count"] == 45
    assert report["prompt_version"] == PROMPT_VERSION
    assert report["corpus_hash"]
    assert report["retrieval_version"]
    assert report["ablation_factors"]["baseline_vs_rag"] == ["evidence_packet", "evidence_policy"]
    assert report["temperature"] is None
    assert set(report["summary"]) == {"baseline", "good", "noisy", "missing"}
    assert set(report["summary"]["good"]) == set(RUBRIC)


def test_live_batch_reuses_one_baseline_per_question(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(benchmark_engine, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(benchmark_engine, "_call_model", lambda prompt: calls.append(prompt) or "## 结论\n\n测试。\n\n## 依据\n\n测试。\n\n## 局限\n\n测试。\n\n## 安全提示\n\n不替代医生建议。")
    report = nutrition_benchmark.run_benchmark(live=True, repeats=1)
    assert len(calls) == len(nutrition_benchmark.QUESTIONS) * 4
    assert report["model"] == "test-model"
    assert report["saved_to"]


def test_api_hides_gold_answers_and_exposes_rubric_and_metadata() -> None:
    questions = client.get("/api/questions?domain=nutrition").json()
    assert "expected_claims" not in questions[0]
    assert "should_abstain" not in questions[0]
    assert "expected_evidence_type" not in questions[0]
    design_questions = client.get("/api/design/questions?domain=nutrition").json()
    assert any(item["should_abstain"] for item in design_questions)
    assert any(item["urgent"] for item in client.get("/api/design/questions?domain=hypertension").json())
    assert client.get("/api/rubric").json()["dimensions"] == RUBRIC
    status = client.get("/api/project-status").json()
    assert status["question_count"] == 30
    assert status["catalog_document_count"] >= 500
    assert status["ethics"]["no_phi"] is True
    assert client.get("/api/course-compliance").json()["status"] == "pass"


def test_live_batch_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/api/run-benchmark", json={"domain": "nutrition", "repeats": 1})
    assert response.status_code == 400


def test_llm_judge_requires_key_and_validates_registered_evidence(monkeypatch) -> None:
    payload = {
        "domain": "nutrition",
        "question_id": "NUT-01",
        "answer": "这是一个足够长度的测试回答，不应在没有密钥时调用外部模型。",
        "evidence_ids": ["S1"],
    }
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert client.post("/api/llm-judge", json=payload).status_code == 400
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-not-real")
    unknown = {**payload, "evidence_ids": ["UNKNOWN"]}
    assert client.post("/api/llm-judge", json=unknown).status_code == 400


def test_model_config_uses_process_memory_and_validates_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/api/model-config", json={"api_key":"sk-test-only-not-real","model":"gpt-5.6","reasoning_effort":"low"})
    assert response.status_code == 200
    assert response.json()["storage"] == "process_memory"
    assert client.get("/api/project-status").json()["model"] == "gpt-5.6"
    assert client.post("/api/model-config", json={"model":"unknown-model","reasoning_effort":"low"}).status_code == 400


def test_connection_error_is_structured_json(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-not-real")
    monkeypatch.setattr(app_module, "test_model_connection", lambda: (_ for _ in ()).throw(type("APIConnectionError", (Exception,), {})()))
    monkeypatch.setattr(app_module, "proxy_status", lambda: {"configured": False, "reachable": None, "source": None, "url": None})
    response = client.post("/api/model-connection-test")
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "APIConnectionError"
    assert "DNS" in response.json()["detail"]["message"]


def test_unreachable_system_proxy_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-not-real")
    monkeypatch.setattr(app_module, "test_model_connection", lambda: (_ for _ in ()).throw(type("APIConnectionError", (Exception,), {})()))
    monkeypatch.setattr(app_module, "proxy_status", lambda: {"configured": True, "reachable": False, "source": "windows", "url": "http://127.0.0.1:1"})
    response = client.post("/api/model-connection-test")
    assert response.status_code == 502
    assert "启动代理客户端" in response.json()["detail"]["message"]


def test_manual_blind_review_is_versioned_and_persisted(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "reviews.json"
    monkeypatch.setattr(app_module, "REVIEWS_PATH", path)
    result = nutrition_benchmark.compare_question("NUT-01", "good")
    payload = {"domain":"nutrition","question_id":"NUT-01","comparison_id":result["comparison_id"],"output_code":"X","answer_hash":result["baseline"]["answer_hash"],"reviewer_alias":"R1","correctness":4,"completeness":4,"safety":5,"clarity":5,"citation_quality":2,"refusal_quality":4,"notes":"盲评"}
    response = client.post("/api/reviews", json=payload)
    assert response.status_code == 200
    saved = client.get("/api/reviews").json()
    assert saved[0]["rubric_version"]
    assert saved[0]["output_code"] == "X"
    assert saved[0]["answer_hash"] == result["baseline"]["answer_hash"]
    assert client.post("/api/reviews", json=payload).status_code == 409
    second = {**payload, "reviewer_alias": "R2", "correctness": 5}
    assert client.post("/api/reviews", json=second).status_code == 200
    summary = client.get("/api/review-summary").json()["outputs"][0]
    assert summary["reviewer_count"] == 2
    assert summary["mean_absolute_disagreement"] is not None


def test_invalid_domain_and_condition_are_rejected() -> None:
    assert client.get("/api/benchmark?domain=unknown").status_code == 400
    bad = client.post("/api/compare", json={"domain":"nutrition","question_id":"NUT-01","retrieval_condition":"broken","live":False})
    assert bad.status_code == 400


def test_wiki_api_end_to_end(tmp_path: Path, monkeypatch) -> None:
    import wiki_engine
    path = tmp_path / "wiki.json"
    monkeypatch.setattr(wiki_engine, "WIKI_PATH", path)
    monkeypatch.setattr(app_module, "load_wiki", lambda: wiki_engine.load_wiki(path))
    monkeypatch.setattr(app_module, "ingest_topic", lambda page: wiki_engine.ingest_topic(page, path))
    monkeypatch.setattr(app_module, "query_wiki", lambda query, limit=5: wiki_engine.query_wiki(query, path, limit))
    monkeypatch.setattr(app_module, "lint_wiki", lambda: wiki_engine.lint_wiki(path))
    response = client.post("/api/wiki/ingest", json={"domain":"nutrition","question_id":"NUT-01","retrieval_condition":"good"})
    assert response.status_code == 200, response.text
    assert response.json()["wiki"]["action"] == "created"
    assert client.get("/api/wiki/query?q=限钠").json()["results"]
    assert client.get("/api/wiki/lint").json()["valid"] is True


def test_d2_tools_are_structured_bounded_and_fail_visibly() -> None:
    from d2_agent import TOOL_SCHEMAS, execute_tool

    assert 3 <= len(TOOL_SCHEMAS) <= 10
    assert all(item["input_schema"]["additionalProperties"] is False for item in TOOL_SCHEMAS.values())
    assert all(item["permissions"] and item["timeout_seconds"] <= 10 for item in TOOL_SCHEMAS.values())
    error = execute_tool("does_not_exist", {})
    assert error["status"] == "error"
    assert error["error"]["code"] == "unknown_tool"


def test_d2_skill_agent_trace_and_urgent_branch() -> None:
    from d2_agent import MAX_AGENT_STEPS, run_agent

    result = run_agent("hypertension", "HTN-07", "good")
    assert result["skill"]["loaded_on_demand"] is True
    assert result["action"] == "escalate"
    assert len(result["trace"]) <= MAX_AGENT_STEPS
    assert all(set(row) >= {"step", "tool", "input", "observation", "decision", "timestamp"} for row in result["trace"])
    assert "chain of thought" in result["trace_policy"]


def test_d2_multi_agent_preserves_complete_sample_chain_and_boundaries() -> None:
    from d2_agent import run_multi_agent

    result = run_multi_agent("nutrition", "NUT-01", "good")
    assert result["roles"] == ["researcher", "writer", "critic"]
    assert len(result["complete_sample_chain"]) == 4
    assert "no retrieval permission" in result["separation_of_duties"]["critic"]
    assert result["cost_report"]["tool_calls"] == 6
    assert result["question"] == nutrition_benchmark.QUESTIONS[0]["question"]
    researcher_ids = {item["chunk_id"] for item in result["researcher"]["evidence_packet"]}
    assert set(result["writer"]["evidence_ids_used"]) == researcher_ids
    assert result["writer"]["draft"]
    assert set(result["critic"]["checks"]) == {"citation_support", "required_structure", "safety_boundary", "urgent_escalation"}
    assert result["revision"]["final_citation_check"]["status"] == "ok"
    assert result["final_answer_hash"]


def test_d2_multi_agent_outputs_change_with_the_selected_question() -> None:
    from d2_agent import run_multi_agent

    normal = run_multi_agent("nutrition", "NUT-01", "good")
    boundary = run_multi_agent("nutrition", "NUT-08", "good")
    assert normal["question"] != boundary["question"]
    assert normal["researcher"]["evidence_packet"] != boundary["researcher"]["evidence_packet"]
    assert normal["writer"]["draft"] != boundary["writer"]["draft"]
    assert normal["researcher"]["gate"]["action"] == "answer"
    assert boundary["researcher"]["gate"]["action"] == "abstain"
    assert normal["final_answer_hash"] != boundary["final_answer_hash"]


def test_d2_golden_set_records_four_layers_and_retains_failures() -> None:
    from d2_evaluation import evaluate_d2

    report = evaluate_d2()
    assert report["status"] == "pass"
    assert report["record_count"] >= 8
    assert all(set(row["checks"]) == {"retrieval_quality", "citation_and_evidence", "answer_quality", "behavior_and_boundary"} for row in report["records"])
    assert report["failures"] == []
