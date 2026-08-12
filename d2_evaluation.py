from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark_engine import RUBRIC_VERSION
from d2_agent import AGENT_VERSION, MAX_AGENT_STEPS, run_agent, run_multi_agent


ROOT = Path(__file__).resolve().parent
GOLDEN_SET = (
    ("nutrition", "NUT-01", "good"),
    ("nutrition", "NUT-05", "good"),
    ("nutrition", "NUT-08", "good"),
    ("nutrition", "NUT-15", "missing"),
    ("hypertension", "HTN-01", "good"),
    ("hypertension", "HTN-07", "good"),
    ("hypertension", "HTN-08", "good"),
    ("hypertension", "HTN-15", "missing"),
)


def _expected_action(domain: str, question_id: str) -> str:
    module = __import__(f"{domain}_benchmark")
    case = next(item for item in module.QUESTIONS if item["id"] == question_id)
    if case.get("urgent"):
        return "escalate"
    if case.get("should_abstain"):
        return "abstain"
    return "answer"


def evaluate_d2() -> dict[str, object]:
    records = []
    failures = []
    for domain, question_id, condition in GOLDEN_SET:
        result = run_agent(domain, question_id, condition)
        expected = _expected_action(domain, question_id)
        citation = result["citation_check"]
        checks = {
            "retrieval_quality": bool(result["evidence"]) if condition == "good" else not result["evidence"],
            "citation_and_evidence": citation["registered_map_valid"] and citation["unsupported_citation_rate"] == 0,
            "answer_quality": bool(result["answer"]) and "## 结论" in result["answer"],
            "behavior_and_boundary": result["action"] == expected and result["stopped_within_limit"],
        }
        record = {
            "question_id": question_id,
            "domain": domain,
            "condition": condition,
            "expected_action": expected,
            "observed_action": result["action"],
            "answer_hash": hashlib.sha256(str(result["answer"]).encode()).hexdigest(),
            "checks": checks,
            "trace": result["trace"],
            "passed": all(checks.values()),
        }
        records.append(record)
        if not record["passed"]:
            failures.append(record)

    multi = run_multi_agent("hypertension", "HTN-03", "good")
    checks = {
        "at_least_eight_fixed_records": len(records) >= 8,
        "four_evaluation_layers_recorded": all(set(row["checks"]) == {"retrieval_quality", "citation_and_evidence", "answer_quality", "behavior_and_boundary"} for row in records),
        "all_agent_paths_stop_within_three_steps": all(len(row["trace"]) <= MAX_AGENT_STEPS for row in records),
        "failures_are_retained": all(row in failures for row in records if not row["passed"]),
        "multi_agent_has_distinct_roles_and_complete_chain": len(set(multi["roles"])) >= 2 and len(multi["complete_sample_chain"]) == 4,
    }
    return {
        "status": "pass" if all(checks.values()) and not failures else "fail",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_pipeline_acceptance_not_model_performance",
        "config": {"golden_set": list(GOLDEN_SET), "agent_version": AGENT_VERSION, "rubric_version": RUBRIC_VERSION, "max_steps": MAX_AGENT_STEPS},
        "checks": checks,
        "record_count": len(records),
        "records": records,
        "failures": failures,
        "multi_agent_sample": multi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D2AM tool, skill, agent, trace, evaluation and multi-agent acceptance")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "d2_evaluation_report.json")
    args = parser.parse_args()
    report = evaluate_d2()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "failures": len(report["failures"])}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
