from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
from pathlib import Path

import requests


def batches(items: list[dict[str, object]], size: int = 10):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_metadata(group: list[dict[str, object]], headers: dict[str, str]) -> dict[str, dict[str, object]]:
    pmids = ["".join(re.findall(r"\d+", str(item.get("identifier") or ""))) for item in group]
    params = {
        "query": f"({' OR '.join(f'EXT_ID:{pmid}' for pmid in pmids)}) AND SRC:MED",
        "format": "json",
        "resultType": "core",
        "pageSize": len(pmids),
    }
    for attempt in range(1, 5):
        try:
            response = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params,
                headers=headers,
                timeout=45,
            )
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(attempt * 1.5)
    found: dict[str, dict[str, object]] = {}
    for record in response.json().get("resultList", {}).get("result", []):
        pmid = str(record.get("pmid") or record.get("id") or "")
        if pmid:
            found[pmid] = record
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="补齐 PubMed 目录的年份、期刊和规范 source_id")
    parser.add_argument("--path", type=Path, default=Path("data/pubmed_corpus.json"))
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    headers = {"User-Agent": "EvidenceLab-course-audit/1.0"}
    metadata: dict[str, dict[str, object]] = {}
    groups = list(batches(documents))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_metadata, group, headers) for group in groups]
        for future in as_completed(futures):
            metadata.update(future.result())

    unresolved: list[str] = []
    for item in documents:
        identifier = str(item.get("identifier") or item.get("source_id") or "").strip()
        pmid = "".join(re.findall(r"\d+", identifier))
        record = metadata.get(pmid, {})
        year_match = re.search(r"(?:19|20)\d{2}", str(record.get("pubYear") or record.get("firstIndexDate") or ""))
        item["source_id"] = identifier
        item["year"] = int(year_match.group()) if year_match else item.get("year")
        journal = record.get("journalInfo", {}).get("journal", {}) if isinstance(record.get("journalInfo"), dict) else {}
        item["organization"] = str(journal.get("title") or journal.get("medlineAbbreviation") or item.get("organization") or "").strip()
        if not item.get("year") or not item.get("organization"):
            unresolved.append(identifier)
    if unresolved:
        raise SystemExit(f"仍有 {len(unresolved)} 条记录缺少年份或期刊，未覆盖原文件。")
    payload["metadata_schema"] = "course-d1-v1"
    args.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已补齐 {len(documents)} 条记录：{args.path}")


if __name__ == "__main__":
    main()
