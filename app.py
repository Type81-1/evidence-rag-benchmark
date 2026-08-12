from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv

import hypertension_benchmark
import nutrition_benchmark
from benchmark_engine import LLM_JUDGE_VERSION, PROMPT_VERSION, RETRIEVAL_VERSION, RUBRIC, RUBRIC_VERSION, RUNS_DIR, judge_answer, proxy_status, test_model_connection
from course_compliance import build_compliance_report
from d2_agent import TOOL_SCHEMAS, capability_manifest, execute_tool, run_agent, run_multi_agent
from wiki_engine import WIKI_VERSION, build_topic_page, ingest_topic, lint_wiki, load_wiki, query_wiki


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REVIEWS_PATH = BASE_DIR / "data" / "manual_reviews.json"
load_dotenv(BASE_DIR / ".env", override=False)
SUPPORTED_MODELS = ("gpt-5.6", "gpt-5.6-terra", "gpt-5.5", "gpt-5.4")

app = FastAPI(title="EvidenceLab RAG 评测台", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class CompareRequest(BaseModel):
    domain: str = Field(default="nutrition")
    question_id: str = Field(min_length=2, max_length=50)
    retrieval_condition: str = Field(default="good")
    live: bool = False


class BatchRequest(BaseModel):
    domain: str = Field(default="nutrition")
    repeats: int = Field(default=1, ge=1, le=3)


class ModelConfigRequest(BaseModel):
    api_key: SecretStr | None = None
    model: str = Field(default="gpt-5.6")
    reasoning_effort: str = Field(default="low", pattern="^(none|low|medium|high)$")


class ReviewRequest(BaseModel):
    domain: str
    question_id: str
    comparison_id: str = Field(min_length=16, max_length=64)
    output_code: str = Field(pattern="^[XY]$")
    answer_hash: str = Field(pattern="^[a-f0-9]{64}$")
    reviewer_alias: str = Field(min_length=1, max_length=50)
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    citation_quality: int = Field(ge=1, le=5)
    refusal_quality: int = Field(ge=1, le=5)
    notes: str = Field(default="", max_length=1000)


class JudgeRequest(BaseModel):
    domain: str = Field(default="nutrition")
    question_id: str = Field(min_length=2, max_length=50)
    answer: str = Field(min_length=20, max_length=10000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)


class WikiIngestRequest(BaseModel):
    domain: str = Field(default="nutrition")
    question_id: str = Field(min_length=2, max_length=50)
    retrieval_condition: str = Field(default="good", pattern="^(good|noisy|missing)$")


class AdvancedCompareRequest(BaseModel):
    domain: str = Field(default="nutrition")
    question_id: str = Field(min_length=2, max_length=50)
    retrieval_condition: str = Field(default="good", pattern="^(good|noisy|missing)$")
    update_wiki: bool = True


class ToolCallRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    arguments: dict[str, object] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    domain: str = Field(default="nutrition", pattern="^(nutrition|hypertension)$")
    question_id: str = Field(min_length=2, max_length=50)
    retrieval_condition: str = Field(default="good", pattern="^(good|noisy|missing)$")


def _benchmark_module(domain: str):
    if domain == "nutrition":
        return nutrition_benchmark
    if domain == "hypertension":
        return hypertension_benchmark
    raise HTTPException(status_code=400, detail="实验域必须是 nutrition 或 hypertension")


def _public_question(case: dict[str, object]) -> dict[str, object]:
    return {key: case[key] for key in ("id", "question", "track", "topic")}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "evidence-rag-benchmark"}


@app.get("/api/project-status")
def project_status() -> dict[str, object]:
    return {
        "track": "赛道三：专用 AI vs 通用大模型对比评估",
        "domains": ["nutrition", "hypertension"],
        "question_count": len(nutrition_benchmark.QUESTIONS) + len(hypertension_benchmark.QUESTIONS),
        "evidence_count": len(nutrition_benchmark.CURATED_EVIDENCE) + len(hypertension_benchmark.CURATED_EVIDENCE) + len(nutrition_benchmark.CATALOG_EVIDENCE),
        "curated_evidence_count": len(nutrition_benchmark.CURATED_EVIDENCE) + len(hypertension_benchmark.CURATED_EVIDENCE),
        "catalog_document_count": len(nutrition_benchmark.CATALOG_EVIDENCE),
        "retrieval_candidates_by_domain": {
            "nutrition": len(nutrition_benchmark.EVIDENCE),
            "hypertension": len(hypertension_benchmark.EVIDENCE),
        },
        "conditions": ["baseline", "good", "noisy", "missing"],
        "retrieval_gate": True,
        "blind_review_hides_gold": True,
        "rubric_dimensions": list(RUBRIC),
        "live_model_available": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-5.6"),
        "reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", "low"),
        "supported_models": SUPPORTED_MODELS,
        "proxy": {key: value for key, value in proxy_status().items() if key != "url"},
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "llm_judge_version": LLM_JUDGE_VERSION,
        "wiki_version": WIKI_VERSION,
        "d2_capabilities": capability_manifest(),
        "ethics": {"no_phi": True, "ai_disclosure": True, "emergency_escalation": True},
    }


@app.get("/api/course-compliance")
def course_compliance() -> dict[str, object]:
    return build_compliance_report()


@app.get("/api/d2-compliance")
def d2_compliance() -> dict[str, object]:
    return build_compliance_report()


@app.get("/api/tools")
def tools() -> dict[str, object]:
    return {"count": len(TOOL_SCHEMAS), "tools": TOOL_SCHEMAS}


@app.post("/api/tools/execute")
def tool_execute(payload: ToolCallRequest) -> dict[str, object]:
    return execute_tool(payload.name, payload.arguments)


@app.post("/api/agent/run")
def agent_run(payload: AgentRunRequest) -> dict[str, object]:
    try:
        return run_agent(payload.domain, payload.question_id, payload.retrieval_condition)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="未找到该测试问题") from exc


@app.post("/api/multi-agent/run")
def multi_agent_run(payload: AgentRunRequest) -> dict[str, object]:
    try:
        return run_multi_agent(payload.domain, payload.question_id, payload.retrieval_condition)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="未找到该测试问题") from exc


def _model_error(exc: Exception) -> HTTPException:
    name = type(exc).__name__
    messages = {
        "APIConnectionError": "无法连接 api.openai.com。请检查 VPN/代理和 DNS；当前错误与 API Key 是否正确无关。",
        "AuthenticationError": "API Key 无效、已撤销或不属于可用项目。",
        "PermissionDeniedError": "当前 API 项目无权使用所选模型，请换用有权限的模型。",
        "NotFoundError": "所选模型不存在或当前 API 项目尚未获得访问权限。",
        "RateLimitError": "请求达到速率或额度限制，请检查项目余额与用量层级。",
        "BadRequestError": "OpenAI 拒绝了请求参数，请检查模型和推理强度设置。",
    }
    message = messages.get(name, "OpenAI 调用失败，请查看服务日志。")
    proxy = proxy_status()
    if name == "APIConnectionError" and proxy["configured"] and proxy["reachable"] is False:
        message = "Windows 已配置代理，但代理端口没有程序监听。请先启动代理客户端，再测试连接。"
    return HTTPException(status_code=502, detail={"code": name, "message": message, "proxy": {key: value for key, value in proxy.items() if key != "url"}})


@app.post("/api/model-config")
def configure_model(payload: ModelConfigRequest) -> dict[str, object]:
    if payload.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="不支持的模型选项")
    if payload.api_key and payload.api_key.get_secret_value().strip():
        os.environ["OPENAI_API_KEY"] = payload.api_key.get_secret_value().strip()
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="请输入 OpenAI API Key")
    os.environ["OPENAI_MODEL"] = payload.model
    os.environ["OPENAI_REASONING_EFFORT"] = payload.reasoning_effort
    return {"configured": True, "model": payload.model, "reasoning_effort": payload.reasoning_effort, "storage": "process_memory"}


@app.post("/api/model-connection-test")
def model_connection_test() -> dict[str, object]:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="请先配置 OpenAI API Key")
    try:
        return {"connected": True, **test_model_connection()}
    except Exception as exc:
        raise _model_error(exc) from exc


@app.post("/api/llm-judge")
def llm_judge(payload: JudgeRequest) -> dict[str, object]:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="模型评审需要配置 OPENAI_API_KEY")
    module = _benchmark_module(payload.domain)
    try:
        case = next(item for item in module.QUESTIONS if item["id"] == payload.question_id)
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail="未找到该测试问题") from exc
    registry = {item.id: item for item in module.EVIDENCE}
    unknown = sorted(set(payload.evidence_ids) - set(registry))
    if unknown:
        raise HTTPException(status_code=400, detail=f"证据 ID 不在登记表中：{', '.join(unknown)}")
    try:
        return judge_answer(str(case["question"]), payload.answer, [registry[item_id] for item_id in payload.evidence_ids])
    except Exception as exc:
        raise _model_error(exc) from exc


@app.get("/api/questions")
def questions(domain: str = "nutrition") -> list[dict[str, object]]:
    return [_public_question(case) for case in _benchmark_module(domain).QUESTIONS]


@app.get("/api/design/questions")
def design_questions(domain: str = "nutrition") -> list[dict[str, object]]:
    return [
        {
            **_public_question(case),
            "expected_evidence_type": case["expected_evidence_type"],
            "should_abstain": case["should_abstain"],
            "urgent": case.get("urgent", False),
            "notes": case["notes"],
        }
        for case in _benchmark_module(domain).QUESTIONS
    ]


@app.get("/api/rubric")
def rubric() -> dict[str, object]:
    return {"version": RUBRIC_VERSION, "scale": "1-5", "dimensions": RUBRIC}


@app.get("/api/benchmark")
def benchmark(domain: str = "nutrition") -> dict[str, object]:
    result = _benchmark_module(domain).run_benchmark(live=False)
    result["domain"] = domain
    return result


@app.post("/api/run-benchmark")
def run_live_benchmark(payload: BatchRequest) -> dict[str, object]:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="批量真实实验需要配置 OPENAI_API_KEY")
    try:
        return _benchmark_module(payload.domain).run_benchmark(live=True, repeats=payload.repeats)
    except Exception as exc:
        raise _model_error(exc) from exc


@app.post("/api/compare")
def compare(payload: CompareRequest) -> dict[str, object]:
    if payload.retrieval_condition not in {"good", "noisy", "missing"}:
        raise HTTPException(status_code=400, detail="检索条件必须是 good、noisy 或 missing")
    if payload.live and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="实时模型模式需要配置 OPENAI_API_KEY")
    try:
        result = _benchmark_module(payload.domain).compare_question(payload.question_id, payload.retrieval_condition, live=payload.live)
        result["domain"] = payload.domain
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="未找到该测试问题") from exc
    except Exception as exc:
        raise _model_error(exc) from exc


@app.post("/api/advanced-compare")
def advanced_compare(payload: AdvancedCompareRequest) -> dict[str, object]:
    module = _benchmark_module(payload.domain)
    try:
        result = module.compare_question(payload.question_id, payload.retrieval_condition, live=False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="未找到该测试问题") from exc
    wiki = None
    if payload.update_wiki and result["rag"]["evidence"]:
        evidence_registry = {item.id: item for item in module.EVIDENCE}
        evidence = [evidence_registry[item["chunk_id"]] for item in result["rag"]["evidence"] if item["chunk_id"] in evidence_registry]
        if not evidence:
            from benchmark_engine import Evidence
            evidence = [Evidence(**{key: item[key] for key in Evidence.__dataclass_fields__}) for item in result["rag"]["evidence"]]
        case = result["question"]
        page = build_topic_page(str(case["topic"]), str(case["question"]), str(result["rag"]["answer"]), evidence, payload.domain)
        wiki = ingest_topic(page)
    result["domain"] = payload.domain
    result["wiki"] = wiki
    return result


@app.post("/api/wiki/ingest")
def wiki_ingest(payload: WikiIngestRequest) -> dict[str, object]:
    result = advanced_compare(AdvancedCompareRequest(**payload.model_dump(), update_wiki=True))
    return {"wiki": result["wiki"], "question_id": payload.question_id}


@app.get("/api/wiki/pages")
def wiki_pages() -> dict[str, object]:
    store = load_wiki()
    return {"version": store["version"], "pages": list(store["pages"].values()), "history_count": len(store["history"])}


@app.get("/api/wiki/query")
def wiki_query(q: str, limit: int = 5) -> dict[str, object]:
    return {"query": q, "results": query_wiki(q, limit=max(1, min(limit, 20)))}


@app.get("/api/wiki/lint")
def wiki_lint() -> dict[str, object]:
    return lint_wiki()


@app.get("/api/runs")
def runs() -> list[dict[str, object]]:
    if not RUNS_DIR.exists():
        return []
    items = []
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.append({key: payload.get(key) for key in ("run_id", "domain", "model", "temperature", "executed_at", "repeats", "saved_to")})
    return items


@app.get("/api/reviews")
def reviews() -> list[dict[str, object]]:
    if not REVIEWS_PATH.exists():
        return []
    return json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))


@app.post("/api/reviews")
def save_review(payload: ReviewRequest) -> dict[str, object]:
    _benchmark_module(payload.domain)
    existing = reviews()
    record = payload.model_dump()
    record["rubric_version"] = RUBRIC_VERSION
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    record["review_id"] = f"{payload.comparison_id}:{payload.output_code}:{payload.reviewer_alias}"
    if any(item.get("review_id") == record["review_id"] for item in existing):
        raise HTTPException(status_code=409, detail="该评审已提交过这一匿名输出")
    existing.append(record)
    REVIEWS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "review_count": len(existing)}


@app.get("/api/review-summary")
def review_summary() -> dict[str, object]:
    dimensions = list(RUBRIC)
    groups: dict[str, list[dict[str, object]]] = {}
    for record in reviews():
        groups.setdefault(str(record["answer_hash"]), []).append(record)
    outputs = []
    for answer_hash, records in groups.items():
        means = {name: round(sum(int(row[name]) for row in records) / len(records), 2) for name in dimensions}
        pair_differences = [
            abs(int(left[name]) - int(right[name]))
            for index, left in enumerate(records)
            for right in records[index + 1 :]
            for name in dimensions
        ]
        outputs.append(
            {
                "answer_hash": answer_hash,
                "reviewer_count": len({str(row["reviewer_alias"]) for row in records}),
                "mean_scores": means,
                "mean_absolute_disagreement": round(sum(pair_differences) / len(pair_differences), 3) if pair_differences else None,
            }
        )
    return {"rubric_version": RUBRIC_VERSION, "outputs": outputs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
