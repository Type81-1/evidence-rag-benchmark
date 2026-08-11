from __future__ import annotations

import argparse
import json
from pathlib import Path

import hypertension_benchmark
import nutrition_benchmark


ROOT = Path(__file__).resolve().parent


def run_evaluation(domain: str = "nutrition") -> dict[str, object]:
    return (hypertension_benchmark if domain == "hypertension" else nutrition_benchmark).run_benchmark()


def main() -> None:
    parser = argparse.ArgumentParser(description="运行营养 RAG 对照评测")
    parser.add_argument("--domain", choices=("nutrition", "hypertension"), default="nutrition")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation_report.json")
    args = parser.parse_args()
    report = run_evaluation(args.domain)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"问题数：{report['question_count']}；对照数：{report['comparison_count']}；报告：{args.output}")


if __name__ == "__main__":
    main()
