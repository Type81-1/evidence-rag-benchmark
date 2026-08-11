from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_service import search_pubmed


DEFAULT_QUERIES = [
    "hypertension treatment cardiovascular outcomes",
    "dyslipidemia statin cardiovascular prevention",
    "diabetes cardiovascular risk management",
    "stroke prevention guideline",
    "heart failure guideline therapy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建心血管临床证据语料清单")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/pubmed_corpus.json"))
    args = parser.parse_args()
    per_query = max(140, ((args.target * 7 // 5) + len(DEFAULT_QUERIES) - 1) // len(DEFAULT_QUERIES))
    records: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for query in DEFAULT_QUERIES:
        try:
            for item in search_pubmed(query, limit=per_query):
                records[item.identifier] = item.to_dict()
        except Exception as exc:
            errors.append(f"{query}: {type(exc).__name__}: {exc}")
        time.sleep(0.4)
    documents = list(records.values())[: args.target]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "valid_document_count": len(documents),
        "candidate_document_count": len(records),
        "queries": DEFAULT_QUERIES,
        "errors": errors,
        "documents": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"有效文献：{payload['valid_document_count']}；输出：{args.output}")
    if len(documents) < args.target:
        raise SystemExit(f"未达到目标 {args.target}，请检查网络或扩大检索式。")


if __name__ == "__main__":
    main()
