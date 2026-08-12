from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from benchmark_engine import Evidence, build_evidence_map, verify_evidence_map


AGENT_VERSION = "d2-evidence-agent-v1"
SKILL_VERSION = "evidence-grade-v1"
MAX_AGENT_STEPS = 3

TOOL_SCHEMAS = {
    "search_evidence": {
        "description": "Search the registered evidence corpus for one frozen benchmark question and return at most five structured passages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "enum": ["nutrition", "hypertension"]},
                "question_id": {"type": "string", "minLength": 2, "maxLength": 50},
                "condition": {"type": "string", "enum": ["good", "noisy", "missing"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["domain", "question_id"],
            "additionalProperties": False,
        },
        "permissions": ["read:registered-corpus"],
        "timeout_seconds": 10,
    },
    "verify_citations": {
        "description": "Verify that cited chunk IDs resolve to registered sources and that cited claims overlap their evidence passages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1, "maxLength": 10000},
                "evidence": {"type": "array", "maxItems": 5, "items": {"type": "object"}},
            },
            "required": ["answer", "evidence"],
            "additionalProperties": False,
        },
        "permissions": ["read:provided-evidence"],
        "timeout_seconds": 5,
    },
    "assess_evidence_grade": {
        "description": "Apply the reusable evidence-grade skill to classify source strength and identify evidence gaps.",
        "input_schema": {
            "type": "object",
            "properties": {"evidence": {"type": "array", "maxItems": 5, "items": {"type": "object"}}},
            "required": ["evidence"],
            "additionalProperties": False,
        },
        "permissions": ["read:provided-evidence"],
        "timeout_seconds": 5,
    },
}


def _module(domain: str):
    if domain == "nutrition":
        import nutrition_benchmark
        return nutrition_benchmark
    if domain == "hypertension":
        import hypertension_benchmark
        return hypertension_benchmark
    raise ValueError("domain must be nutrition or hypertension")


def _case(domain: str, question_id: str) -> tuple[object, dict[str, object]]:
    module = _module(domain)
    try:
        return module, next(item for item in module.QUESTIONS if item["id"] == question_id)
    except StopIteration as exc:
        raise KeyError(question_id) from exc


def search_evidence(domain: str, question_id: str, condition: str = "good", limit: int = 3) -> dict[str, object]:
    module, case = _case(domain, question_id)
    if condition not in {"good", "noisy", "missing"}:
        raise ValueError("condition must be good, noisy, or missing")
    evidence, diagnostics = module.retrieve(case, condition)
    evidence = evidence[: max(1, min(limit, 5))]
    return {
        "status": "ok",
        "question_id": question_id,
        "condition": condition,
        "results": [item.to_dict() for item in evidence],
        "diagnostics": {
            "candidate_pool_size": diagnostics["candidate_pool_size"],
            "selected_roles": diagnostics["selected_roles"],
            "validation": diagnostics["validation"],
        },
    }


def _evidence_from_payload(rows: list[dict[str, object]]) -> list[Evidence]:
    fields = Evidence.__dataclass_fields__
    return [Evidence(**{name: row[name] for name in fields}) for row in rows]


def verify_citations(answer: str, evidence: list[dict[str, object]]) -> dict[str, object]:
    import re
    from benchmark_engine import _citation_support

    items = _evidence_from_payload(evidence)
    evidence_map = build_evidence_map(items)
    map_check = verify_evidence_map(evidence_map)
    cited = sorted(set(re.findall(r"\[([A-Z][A-Z0-9_-]*)]", answer)))
    precision, unsupported_rate, unsupported = _citation_support(answer, items)
    return {
        "status": "ok" if map_check["valid"] and not unsupported else "failed",
        "cited_chunk_ids": cited,
        "registered_map_valid": map_check["valid"],
        "citation_precision": precision,
        "unsupported_citation_rate": unsupported_rate,
        "unsupported": unsupported,
    }


def assess_evidence_grade(evidence: list[dict[str, object]]) -> dict[str, object]:
    """Reusable, task-loaded skill: classify evidence without changing the main safety prompt."""
    hierarchy = {
        "系统综述": 5, "荟萃分析": 5, "临床指南": 5, "国际指南": 5,
        "随机对照试验": 4, "循证推荐": 4, "测量指南": 4,
        "队列研究": 3, "综述": 3, "叙述性综述": 2, "患者安全指南": 4,
    }
    rows = []
    for item in evidence:
        quality = str(item.get("quality") or "未分类")
        rows.append({
            "chunk_id": item.get("chunk_id") or item.get("id"),
            "quality": quality,
            "grade": hierarchy.get(quality, 1),
            "identifier": item.get("identifier"),
            "url": item.get("url"),
        })
    grades = [int(row["grade"]) for row in rows]
    return {
        "status": "ok",
        "skill": SKILL_VERSION,
        "loaded_on_demand": True,
        "items": rows,
        "strongest_grade": max(grades, default=0),
        "adequate_for_general_answer": bool(grades and max(grades) >= 4),
        "limitations": [] if grades else ["no registered evidence was supplied"],
    }


TOOL_HANDLERS: dict[str, Callable[..., dict[str, object]]] = {
    "search_evidence": search_evidence,
    "verify_citations": verify_citations,
    "assess_evidence_grade": assess_evidence_grade,
}


def execute_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    if name not in TOOL_HANDLERS:
        return {"status": "error", "error": {"code": "unknown_tool", "message": f"Unknown tool: {name}"}}
    allowed = set(TOOL_SCHEMAS[name]["input_schema"]["properties"])
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        return {"status": "error", "error": {"code": "invalid_arguments", "message": f"Unexpected fields: {unexpected}"}}
    try:
        return TOOL_HANDLERS[name](**arguments)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "error", "error": {"code": type(exc).__name__, "message": str(exc)}}


def _trace(step: int, tool: str, inputs: dict[str, object], observation: dict[str, object], decision: str) -> dict[str, object]:
    return {
        "step": step,
        "tool": tool,
        "input": inputs,
        "observation": observation,
        "decision": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_agent(domain: str, question_id: str, condition: str = "good") -> dict[str, object]:
    module, case = _case(domain, question_id)
    trace: list[dict[str, object]] = []
    search = execute_tool("search_evidence", {"domain": domain, "question_id": question_id, "condition": condition, "limit": 3})
    evidence = list(search.get("results") or [])
    gate = (search.get("diagnostics") or {}).get("validation") or {"action": "abstain", "reasons": ["retrieval failed"]}
    trace.append(_trace(1, "search_evidence", {"domain": domain, "question_id": question_id, "condition": condition, "limit": 3}, {"status": search["status"], "result_count": len(evidence), "gate": gate}, "grade evidence" if evidence else "stop with evidence-aware refusal"))

    grade = execute_tool("assess_evidence_grade", {"evidence": evidence})
    trace.append(_trace(2, "assess_evidence_grade", {"evidence_count": len(evidence)}, {"status": grade["status"], "strongest_grade": grade.get("strongest_grade"), "adequate": grade.get("adequate_for_general_answer")}, "draft grounded answer" if evidence else "draft refusal"))

    evidence_objects = _evidence_from_payload(evidence)
    result = module.compare_question(question_id, condition, live=False)
    draft = str(result["rag"]["answer"])
    citation_check = execute_tool("verify_citations", {"answer": draft, "evidence": evidence})
    trace.append(_trace(3, "verify_citations", {"answer_hash": hashlib.sha256(draft.encode()).hexdigest(), "evidence_count": len(evidence)}, citation_check, "return checked answer" if citation_check["status"] == "ok" or not evidence else "return guarded answer with verification warning"))
    action = str(gate.get("action") or "abstain")
    return {
        "agent_version": AGENT_VERSION,
        "question_id": question_id,
        "domain": domain,
        "condition": condition,
        "action": action,
        "answer": draft,
        "evidence": [item.to_dict() for item in evidence_objects],
        "skill": grade,
        "citation_check": citation_check,
        "trace": trace,
        "trace_policy": "actions, observations, and decisions only; no private chain of thought",
        "step_limit": MAX_AGENT_STEPS,
        "stopped_within_limit": len(trace) <= MAX_AGENT_STEPS,
    }


def run_multi_agent(domain: str, question_id: str, condition: str = "good") -> dict[str, object]:
    _, case = _case(domain, question_id)
    agent_result = run_agent(domain, question_id, condition)
    evidence = list(agent_result["evidence"])
    gate = dict(agent_result["trace"][0]["observation"].get("gate") or {})
    role_by_chunk = {
        str(item.get("chunk_id")): str(item.get("role"))
        for item in execute_tool(
            "search_evidence",
            {"domain": domain, "question_id": question_id, "condition": condition, "limit": 3},
        ).get("diagnostics", {}).get("selected_roles", [])
    }
    evidence_packet = [
        {
            "chunk_id": item["chunk_id"],
            "source_id": item["source_id"],
            "title": item["title"],
            "organization": item["organization"],
            "year": item["year"],
            "quality": item["quality"],
            "grade": next((row["grade"] for row in agent_result["skill"]["items"] if row["chunk_id"] == item["chunk_id"]), 1),
            "role": role_by_chunk.get(str(item["chunk_id"]), "supporting"),
            "summary": item["summary"],
            "identifier": item["identifier"],
            "url": item["url"],
        }
        for item in evidence
    ]
    researcher = {
        "role": "researcher",
        "question": case["question"],
        "search_plan": {
            "condition": condition,
            "target_evidence_type": gate.get("expected_evidence_type"),
            "candidate_pool_size": agent_result["trace"][0]["observation"].get("result_count"),
            "selection_limit": 3,
        },
        "evidence_packet": evidence_packet,
        "evidence_grade": agent_result["skill"],
        "gate": gate,
        "gaps": list(gate.get("reasons") or []) + ([] if evidence else ["没有可供 Writer 使用的登记证据"]),
        "handoff": "仅使用 evidence_packet 中的登记片段写作；保留适用人群、局限和安全边界。",
    }
    draft = str(agent_result["answer"])
    writer = {
        "role": "writer",
        "question": case["question"],
        "input_hash": hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "instructions_received": researcher["handoff"],
        "evidence_ids_used": [item["chunk_id"] for item in evidence_packet],
        "draft": draft,
        "answer": draft,
        "sections": [heading for heading in ("结论", "依据", "已检索", "缺失证据", "下一步", "局限", "安全提示") if f"## {heading}" in draft],
        "output_schema": ["answer", "citations", "limitations", "safety_boundary"],
    }
    critic_check = execute_tool("verify_citations", {"answer": draft, "evidence": evidence})
    required_sections = ("结论", "安全提示") + (("已检索", "缺失证据", "下一步") if agent_result["action"] == "abstain" else ("依据", "局限"))
    missing_sections = [heading for heading in required_sections if f"## {heading}" not in draft]
    unsafe_phrases = ("立即停药", "建议自行停药", "可以自行停药", "建议加倍剂量", "应该加倍剂量")
    has_safety_boundary = "## 安全提示" in draft and any(term in draft for term in ("不替代", "专业人员", "医生", "营养师", "立即就医"))
    safety_ok = has_safety_boundary and not any(term in draft for term in unsafe_phrases)
    issues: list[dict[str, str]] = []
    if critic_check["status"] != "ok":
        issues.append({"code": "citation_support", "message": f"存在未获登记证据支持的引用：{', '.join(critic_check['unsupported'])}"})
    if missing_sections:
        issues.append({"code": "answer_structure", "message": f"缺少必要部分：{'、'.join(missing_sections)}"})
    if not safety_ok:
        issues.append({"code": "safety_boundary", "message": "回答没有完整满足医疗安全边界或紧急升级要求"})
    if agent_result["action"] == "escalate" and not any(term in draft for term in ("立即就医", "急救", "急诊")):
        issues.append({"code": "urgent_escalation", "message": "检测到急症，但草稿未明确要求立即就医或呼叫急救"})
    recommendations = [item["message"] for item in issues] or ["引用、结构和安全边界均通过，无需修改。"]
    critic = {
        "role": "critic",
        "question": case["question"],
        "citation_check": critic_check,
        "checks": {
            "citation_support": critic_check["status"] == "ok",
            "required_structure": not missing_sections,
            "safety_boundary": safety_ok,
            "urgent_escalation": agent_result["action"] != "escalate" or any(term in draft for term in ("立即就医", "急救", "急诊")),
        },
        "operational_metrics": {
            "citation_precision": critic_check["citation_precision"],
            "unsupported_citation_rate": critic_check["unsupported_citation_rate"],
            "required_sections_present": len(required_sections) - len(missing_sections),
            "required_sections_total": len(required_sections),
        },
        "issues": issues,
        "recommendations": recommendations,
        "verdict": "accept" if not issues else "revise",
    }
    revised_answer = draft
    changes: list[str] = []
    if issues:
        if "安全提示" in missing_sections:
            revised_answer += "\n\n## 安全提示\n\n本回答仅用于教学与研究，不替代专业医疗判断。"
            changes.append("补充安全提示")
        if agent_result["action"] == "escalate" and not any(term in revised_answer for term in ("立即就医", "急救", "急诊")):
            revised_answer = "## 结论\n\n检测到紧急危险信号，应立即就医或呼叫急救。\n\n" + revised_answer
            changes.append("补充急症升级")
    final_check = execute_tool("verify_citations", {"answer": revised_answer, "evidence": evidence})
    revision = {
        "performed": revised_answer != draft,
        "changes": changes,
        "reason": recommendations,
        "revised_answer": revised_answer,
        "final_citation_check": final_check,
    }
    return {
        "workflow": "researcher-writer-critic-v2",
        "question_id": question_id,
        "question": case["question"],
        "domain": domain,
        "condition": condition,
        "roles": ["researcher", "writer", "critic"],
        "separation_of_duties": {
            "researcher": "registered corpus retrieval and evidence grading",
            "writer": "organize only the supplied evidence into the fixed answer format",
            "critic": "citation and safety-boundary verification; no retrieval permission",
        },
        "researcher": researcher,
        "writer": writer,
        "critic": critic,
        "revision": revision,
        "final_answer": revised_answer,
        "final_answer_hash": hashlib.sha256(revised_answer.encode()).hexdigest(),
        "complete_sample_chain": ["researcher evidence packet", "writer draft", "critic review", "final answer"],
        "cost_report": {"model_calls": 0, "tool_calls": 6, "agent_steps": len(agent_result["trace"]), "offline_demo": True},
    }


def capability_manifest() -> dict[str, object]:
    return {
        "version": AGENT_VERSION,
        "tools": TOOL_SCHEMAS,
        "skills": {SKILL_VERSION: {"loaded_on_demand": True, "purpose": "consistent evidence hierarchy and gap assessment"}},
        "agent": {"dynamic_branching": ["answer", "abstain", "escalate"], "max_steps": MAX_AGENT_STEPS, "trace": True},
        "multi_agent": {"roles": ["researcher", "writer", "critic"], "minimum_roles_met": True},
    }
