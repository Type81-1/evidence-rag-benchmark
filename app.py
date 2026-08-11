from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import hypertension_benchmark
import nutrition_benchmark


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="EvidenceLab RAG 评测台", version="2.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class CompareRequest(BaseModel):
    domain: str = Field(default="nutrition")
    question_id: str = Field(min_length=2, max_length=50)
    retrieval_condition: str = Field(default="good")
    live: bool = False


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
        "conditions": ["baseline", "good", "noisy", "missing"],
        "live_model_available": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    }


def _benchmark_module(domain: str):
    if domain == "nutrition":
        return nutrition_benchmark
    if domain == "hypertension":
        return hypertension_benchmark
    raise HTTPException(status_code=400, detail="实验域必须是 nutrition 或 hypertension")


@app.get("/api/questions")
def questions(domain: str = "nutrition") -> list[dict[str, object]]:
    return _benchmark_module(domain).QUESTIONS


@app.get("/api/benchmark")
def benchmark(domain: str = "nutrition") -> dict[str, object]:
    result = _benchmark_module(domain).run_benchmark()
    result["domain"] = domain
    return result


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
