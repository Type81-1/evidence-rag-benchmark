from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import hypertension_benchmark
import nutrition_benchmark
from benchmark_engine import PROMPT_VERSION, RETRIEVAL_VERSION, RUBRIC, RUBRIC_VERSION, RUNS_DIR


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REVIEWS_PATH = BASE_DIR / "data" / "manual_reviews.json"

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
        "evidence_count": len(nutrition_benchmark.EVIDENCE) + len(hypertension_benchmark.EVIDENCE),
        "conditions": ["baseline", "good", "noisy", "missing"],
        "retrieval_gate": True,
        "blind_review_hides_gold": True,
        "rubric_dimensions": list(RUBRIC),
        "live_model_available": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "ethics": {"no_phi": True, "ai_disclosure": True, "emergency_escalation": True},
    }


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
        raise HTTPException(status_code=502, detail=f"模型实验失败：{type(exc).__name__}") from exc


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
