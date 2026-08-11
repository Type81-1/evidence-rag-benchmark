from fastapi.testclient import TestClient

from app import app
import hypertension_benchmark
from nutrition_benchmark import QUESTIONS, compare_question, retrieve, run_benchmark


client = TestClient(app)


def test_fixed_nutrition_question_set_has_required_scenarios() -> None:
    assert len(QUESTIONS) >= 6
    topics = {case["topic"] for case in QUESTIONS}
    assert {"限钠饮食", "地中海饮食", "营养补充剂"}.issubset(topics)


def test_good_retrieval_is_relevant_and_traceable() -> None:
    case = QUESTIONS[0]
    evidence = retrieve(case, "good")
    assert evidence
    assert all(item.topic in case["evidence_topics"] for item in evidence)
    assert all(item.url.startswith("https://") for item in evidence)


def test_missing_retrieval_triggers_refusal() -> None:
    result = compare_question("NUT-01", "missing")
    assert result["rag"]["evidence"] == []
    assert result["rag"]["metrics"]["appropriate_refusal"] == 1.0


def test_noisy_retrieval_can_reduce_claim_coverage() -> None:
    good = compare_question("NUT-01", "good")
    noisy = compare_question("NUT-01", "noisy")
    assert noisy["rag"]["metrics"]["claim_coverage"] < good["rag"]["metrics"]["claim_coverage"]


def test_rag_citations_are_limited_to_retrieved_evidence() -> None:
    result = compare_question("NUT-02", "good")
    assert result["rag"]["metrics"]["citation_precision"] == 1.0
    assert result["rag"]["metrics"]["unsupported_citation_rate"] == 0.0


def test_benchmark_contains_all_conditions_and_metrics() -> None:
    report = run_benchmark()
    assert report["comparison_count"] == len(QUESTIONS) * 3
    assert set(report["summary"]) == {"baseline", "good", "noisy", "missing"}
    assert set(report["summary"]["good"]) == {"claim_coverage", "citation_precision", "citation_coverage", "unsupported_citation_rate", "appropriate_refusal"}


def test_api_exposes_comparison_and_rejects_invalid_condition() -> None:
    response = client.post("/api/compare", json={"question_id": "NUT-01", "retrieval_condition": "good", "live": False})
    assert response.status_code == 200
    assert response.json()["run_mode"] == "reproducible_demo"
    bad = client.post("/api/compare", json={"question_id": "NUT-01", "retrieval_condition": "broken", "live": False})
    assert bad.status_code == 400


def test_hypertension_set_covers_treatment_measurement_and_safety() -> None:
    topics = {case["topic"] for case in hypertension_benchmark.QUESTIONS}
    assert {"长期治疗", "血压测量", "停药安全"}.issubset(topics)
    assert len(hypertension_benchmark.EVIDENCE) >= 8


def test_hypertension_good_rag_is_traceable_and_noise_is_detected() -> None:
    good = hypertension_benchmark.compare_question("HTN-03", "good")
    noisy = hypertension_benchmark.compare_question("HTN-03", "noisy")
    assert good["rag"]["metrics"]["citation_precision"] == 1.0
    assert noisy["rag"]["metrics"]["citation_precision"] < good["rag"]["metrics"]["citation_precision"]
    assert all(item["url"].startswith("https://") for item in good["rag"]["evidence"])


def test_hypertension_missing_retrieval_refuses_and_api_switches_domain() -> None:
    missing = hypertension_benchmark.compare_question("HTN-06", "missing")
    assert missing["rag"]["metrics"]["appropriate_refusal"] == 1.0
    questions = client.get("/api/questions?domain=hypertension")
    assert questions.status_code == 200
    assert questions.json()[0]["id"].startswith("HTN-")
    response = client.post("/api/compare", json={"domain": "hypertension", "question_id": "HTN-01", "retrieval_condition": "good", "live": False})
    assert response.status_code == 200
    assert response.json()["domain"] == "hypertension"


def test_invalid_domain_is_rejected() -> None:
    response = client.get("/api/benchmark?domain=unknown")
    assert response.status_code == 400
