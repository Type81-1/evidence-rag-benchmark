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
from benchmark_engine import PROMPT_VERSION, RUBRIC, Evidence, bm25_rank, build_prompt, score_answer


client = TestClient(app_module.app)
REQUIRED_FIELDS = {"id", "question", "track", "expected_evidence_type", "notes", "should_abstain", "expected_claims", "relevant_evidence_ids"}


def test_question_sets_are_preregistered_and_include_abstention_and_adversarial_cases() -> None:
    for questions in (nutrition_benchmark.QUESTIONS, hypertension_benchmark.QUESTIONS):
        assert len(questions) >= 8
        assert all(REQUIRED_FIELDS.issubset(case) for case in questions)
        assert 1 <= sum(bool(case["should_abstain"]) for case in questions) <= 3
        assert any("对抗" in str(case["notes"]) or "越界" in str(case["notes"]) or case.get("urgent") for case in questions)


def test_frozen_question_manifest_can_be_exported() -> None:
    subprocess.run([sys.executable, "scripts/export_question_set.py"], check=True)
    payload = json.loads(Path("data/test_questions.json").read_text(encoding="utf-8"))
    assert payload["status"] == "frozen-before-live-evaluation"
    assert len(payload["questions"]) == 16


def test_evidence_registry_has_traceable_identifiers_and_urls() -> None:
    for corpus in (nutrition_benchmark.EVIDENCE, hypertension_benchmark.EVIDENCE):
        assert len(corpus) >= 10
        assert len({item.id for item in corpus}) == len(corpus)
        assert all(item.identifier and item.url.startswith("https://") for item in corpus)


def test_bm25_ranking_uses_question_text_and_returns_scores() -> None:
    ranked = bm25_rank("限钠饮食能降低血压吗", nutrition_benchmark.EVIDENCE)
    assert ranked[0][0] > 0
    assert ranked[0][1].id in {"S1", "S2"}


def test_degraded_retrieval_reduces_precision_without_using_empty_context() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    good_evidence, good_metrics = nutrition_benchmark.retrieve(case, "good")
    noisy_evidence, noisy_metrics = nutrition_benchmark.retrieve(case, "noisy")
    assert good_evidence and noisy_evidence
    assert noisy_metrics["precision_at_k"] < good_metrics["precision_at_k"]
    assert set(noisy_metrics) == {"precision_at_k", "recall_at_k", "mrr"}


def test_fair_prompt_has_identical_instructions_and_only_packet_policy_differs() -> None:
    case = nutrition_benchmark.QUESTIONS[0]
    evidence, _ = nutrition_benchmark.retrieve(case, "good")
    baseline = build_prompt(str(case["question"]), [], False, "饮食营养")
    rag = build_prompt(str(case["question"]), evidence, True, "饮食营养")
    assert baseline.split("EVIDENCE_POLICY=")[0] == rag.split("EVIDENCE_POLICY=")[0]
    assert "EVIDENCE_POLICY=OPTIONAL" in baseline
    assert "EVIDENCE_POLICY=REQUIRED" in rag
    assert PROMPT_VERSION == "fair-ab-v1"


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
    evidence = [nutrition_benchmark.EVIDENCE[1]]
    answer = "## 结论\n\n降低钠摄入可以降低血压，但存在个体差异 [S2]。\n\n## 依据\n\n见随机对照试验。\n\n## 局限\n\n需结合整体饮食。\n\n## 安全提示\n\n不替代医生或营养师建议。"
    metrics = score_answer(case, answer, evidence, evidence_required=True)
    assert set(RUBRIC).issubset(metrics)
    assert metrics["citation_quality"] > 0
    assert metrics["unsupported_citation_rate"] == 0


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


def test_benchmark_records_metadata_and_all_arms() -> None:
    report = nutrition_benchmark.run_benchmark()
    assert report["question_count"] == 8
    assert report["comparison_count"] == 24
    assert report["prompt_version"] == PROMPT_VERSION
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
    assert client.get("/api/rubric").json()["dimensions"] == RUBRIC
    status = client.get("/api/project-status").json()
    assert status["question_count"] == 16
    assert status["ethics"]["no_phi"] is True


def test_live_batch_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/api/run-benchmark", json={"domain": "nutrition", "repeats": 1})
    assert response.status_code == 400


def test_manual_blind_review_is_versioned_and_persisted(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "reviews.json"
    monkeypatch.setattr(app_module, "REVIEWS_PATH", path)
    payload = {"domain":"nutrition","question_id":"NUT-01","arm_code":"A","reviewer_alias":"R1","correctness":4,"completeness":4,"safety":5,"clarity":5,"citation_quality":2,"refusal_quality":4,"notes":"盲评"}
    response = client.post("/api/reviews", json=payload)
    assert response.status_code == 200
    saved = client.get("/api/reviews").json()
    assert saved[0]["rubric_version"]
    assert saved[0]["arm_code"] == "A"


def test_invalid_domain_and_condition_are_rejected() -> None:
    assert client.get("/api/benchmark?domain=unknown").status_code == 400
    bad = client.post("/api/compare", json={"domain":"nutrition","question_id":"NUT-01","retrieval_condition":"broken","live":False})
    assert bad.status_code == 400
