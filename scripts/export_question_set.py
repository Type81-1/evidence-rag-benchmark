from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hypertension_benchmark  # noqa: E402
import nutrition_benchmark  # noqa: E402


OUTPUT = ROOT / "data" / "test_questions.json"


def main() -> None:
    payload = {
        "schema_version": "2.0",
        "status": "frozen-before-live-evaluation",
        "required_fields": ["id", "question", "track", "expected_evidence_type", "notes", "should_abstain"],
        "questions": nutrition_benchmark.QUESTIONS + hypertension_benchmark.QUESTIONS,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(payload['questions'])} 题：{OUTPUT}")


if __name__ == "__main__":
    main()
