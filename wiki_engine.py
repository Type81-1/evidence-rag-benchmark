from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from benchmark_engine import Evidence, build_evidence_map, verify_evidence_map


ROOT = Path(__file__).resolve().parent
WIKI_PATH = ROOT / "data" / "wiki_store.json"
WIKI_VERSION = "evidence-wiki-v1"


def _empty_store() -> dict[str, object]:
    return {"version": WIKI_VERSION, "pages": {}, "history": []}


def load_wiki(path: Path = WIKI_PATH) -> dict[str, object]:
    if not path.exists():
        return _empty_store()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("version", WIKI_VERSION)
    payload.setdefault("pages", {})
    payload.setdefault("history", [])
    return payload


def save_wiki(store: dict[str, object], path: Path = WIKI_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def topic_slug(topic: str) -> str:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", topic.lower()).strip("-")
    return normalized or hashlib.sha256(topic.encode("utf-8")).hexdigest()[:12]


def build_topic_page(topic: str, question: str, answer: str, evidence: list[Evidence], domain: str) -> dict[str, object]:
    evidence_map = build_evidence_map(evidence)
    validation = verify_evidence_map(evidence_map)
    if not validation["valid"]:
        raise ValueError(f"Wiki evidence map invalid: {validation['invalid_chunks']}")
    citations = sorted(set(re.findall(r"\[([A-Z][A-Z0-9_-]*)]", answer)))
    unknown = sorted(set(citations) - set(evidence_map))
    if unknown:
        raise ValueError(f"Wiki answer cites unknown chunks: {unknown}")
    now = datetime.now(timezone.utc).isoformat()
    return {
        "slug": topic_slug(topic),
        "topic": topic,
        "domain": domain,
        "question": question,
        "content": answer,
        "citations": citations,
        "evidence_map": evidence_map,
        "updated_at": now,
        "content_hash": hashlib.sha256(f"{topic}|{answer}|{json.dumps(evidence_map, sort_keys=True)}".encode("utf-8")).hexdigest(),
    }


def ingest_topic(page: dict[str, object], path: Path = WIKI_PATH) -> dict[str, object]:
    store = load_wiki(path)
    pages = dict(store["pages"])
    slug = str(page["slug"])
    previous = pages.get(slug)
    action = "unchanged" if previous and previous.get("content_hash") == page.get("content_hash") else ("updated" if previous else "created")
    if action != "unchanged":
        if previous:
            store["history"].append({"slug": slug, "archived_at": datetime.now(timezone.utc).isoformat(), "page": previous})
        pages[slug] = page
        store["pages"] = pages
        save_wiki(store, path)
    return {"action": action, "slug": slug, "page_count": len(pages), "history_count": len(store["history"])}


def query_wiki(query: str, path: Path = WIKI_PATH, limit: int = 5) -> list[dict[str, object]]:
    store = load_wiki(path)
    query_terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query.lower()))
    ranked: list[tuple[int, dict[str, object]]] = []
    for page in store["pages"].values():
        haystack = f"{page.get('topic', '')} {page.get('question', '')} {page.get('content', '')}".lower()
        score = sum(3 if term in str(page.get("topic", "")).lower() else 1 for term in query_terms if term in haystack)
        if score:
            ranked.append((score, page))
    ranked.sort(key=lambda row: (row[0], str(row[1].get("updated_at", ""))), reverse=True)
    return [page for _, page in ranked[:limit]]


def lint_wiki(path: Path = WIKI_PATH, *, current_year: int | None = None) -> dict[str, object]:
    store = load_wiki(path)
    year = current_year or datetime.now(timezone.utc).year
    issues: list[dict[str, str]] = []
    pages = store["pages"]
    referenced_topics: set[str] = set()
    for slug, page in pages.items():
        content = str(page.get("content") or "")
        evidence_map = page.get("evidence_map") or {}
        citations = set(re.findall(r"\[([A-Z][A-Z0-9_-]*)]", content))
        for target in re.findall(r"\[\[([^]]+)]]", content):
            referenced_topics.add(topic_slug(target))
        if not citations:
            issues.append({"severity": "error", "code": "missing-citation", "slug": slug})
        for citation in citations - set(evidence_map):
            issues.append({"severity": "error", "code": "unknown-citation", "slug": slug, "detail": citation})
        for chunk_id, entry in evidence_map.items():
            if not str(entry.get("url") or "").startswith("https://"):
                issues.append({"severity": "error", "code": "invalid-url", "slug": slug, "detail": chunk_id})
        updated = str(page.get("updated_at") or "")
        updated_year = int(updated[:4]) if re.match(r"\d{4}", updated) else 0
        if updated_year and year - updated_year > 2:
            issues.append({"severity": "warning", "code": "stale-page", "slug": slug})
    for slug in pages:
        if len(pages) > 1 and slug not in referenced_topics and not any(f"[[{pages[slug].get('topic')}]]" in str(page.get("content")) for page in pages.values() if page is not pages[slug]):
            issues.append({"severity": "warning", "code": "orphan-page", "slug": slug})
    return {
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "page_count": len(pages),
        "issues": issues,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
