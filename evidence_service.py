from __future__ import annotations

import os
import re
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "pubmed_corpus.json"


@dataclass(frozen=True)
class Evidence:
    source_type: str
    title: str
    summary: str
    url: str
    identifier: str
    year: int | None = None
    organization: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["source_id"] = self.identifier
        return payload


LOCAL_EVIDENCE = [
    Evidence(
        source_type="指南",
        title="2024 ESC Guidelines for the management of elevated blood pressure and hypertension",
        summary=(
            "高血压通常需要持续管理。生活方式干预适用于所有患者；是否开始药物治疗以及治疗强度，"
            "需要结合血压水平、心血管风险、靶器官损害和耐受性判断。持续控制血压旨在降低卒中、"
            "心肌梗死、心力衰竭和肾脏结局风险。"
        ),
        url="https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/Elevated-Blood-Pressure-and-Hypertension",
        identifier="ESC-HTN-2024",
    ),
    Evidence(
        source_type="指南",
        title="2021 ESC Guidelines on cardiovascular disease prevention in clinical practice",
        summary=(
            "心血管预防应同时处理血压、血脂、吸烟、体重、饮食和运动等风险因素。降脂药物的使用"
            "应依据 LDL-C 水平和总体心血管风险，而不是只看一次化验结果。"
        ),
        url="https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/CVD-Prevention-Guidelines",
        identifier="ESC-CVD-PREVENTION-2021",
    ),
    Evidence(
        source_type="指南",
        title="2019 ESC/EAS Guidelines for the management of dyslipidaemias",
        summary=(
            "生活方式管理是血脂异常治疗的基础，包括健康饮食、规律运动、体重管理和戒烟。"
            "对于达到相应风险阈值的人群，他汀等降脂药可降低 LDL-C，并减少动脉粥样硬化性心血管事件。"
        ),
        url="https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/Dyslipidaemias-Management-of",
        identifier="ESC-EAS-LIPIDS-2019",
    ),
    Evidence(
        source_type="研究",
        title="SPRINT: A Randomized Trial of Intensive versus Standard Blood-Pressure Control",
        summary=(
            "在部分心血管风险较高且无糖尿病的成人中，强化收缩压控制降低了主要心血管事件和全因死亡，"
            "但部分不良事件增加，说明治疗目标需要个体化并接受监测。"
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/26551272/",
        identifier="PMID:26551272",
    ),
]


QUERY_TERMS = {
    "高血压": "hypertension long-term treatment guideline cardiovascular outcomes",
    "血压": "hypertension treatment guideline cardiovascular outcomes",
    "血脂": "dyslipidemia lifestyle statin cardiovascular risk",
    "胆固醇": "hypercholesterolemia lifestyle statin cardiovascular risk",
    "糖尿病": "diabetes cardiovascular prevention guideline",
    "卒中": "stroke secondary prevention guideline",
}


def build_english_query(question: str) -> str:
    for keyword, query in QUERY_TERMS.items():
        if keyword in question:
            return query
    return "cardiovascular disease clinical guideline treatment"


def _request(url: str, *, params: dict[str, object]) -> requests.Response:
    headers = {"User-Agent": "OpenEvidence-MVP/1.0 (educational project)"}
    response = requests.get(url, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    return response


def search_pubmed(query: str, limit: int = 3) -> list[Evidence]:
    common: dict[str, object] = {"db": "pubmed", "retmode": "json"}
    if os.getenv("NCBI_API_KEY"):
        common["api_key"] = os.environ["NCBI_API_KEY"]
    if os.getenv("NCBI_EMAIL"):
        common["email"] = os.environ["NCBI_EMAIL"]

    search_response = _request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={**common, "term": query, "retmax": limit, "sort": "relevance"},
    )
    pmids = search_response.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    fetch_response = _request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={**common, "id": ",".join(pmids), "retmode": "xml"},
    )
    root = ET.fromstring(fetch_response.text)
    results: list[Evidence] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="").strip()
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else "PubMed article"
        abstracts = ["".join(node.itertext()).strip() for node in article.findall(".//AbstractText")]
        summary = re.sub(r"\s+", " ", " ".join(abstracts)).strip()
        if pmid and summary:
            pub_date = article.findtext(".//JournalIssue/PubDate/Year", default="") or article.findtext(
                ".//JournalIssue/PubDate/MedlineDate", default=""
            )
            year_match = re.search(r"(?:19|20)\d{2}", pub_date)
            journal = article.findtext(".//Journal/Title", default="").strip()
            results.append(
                Evidence(
                    "PubMed",
                    title,
                    summary,
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    f"PMID:{pmid}",
                    int(year_match.group()) if year_match else None,
                    journal,
                )
            )
    return results


def search_trials(query: str, limit: int = 2) -> list[Evidence]:
    response = _request(
        "https://clinicaltrials.gov/api/v2/studies",
        params={"query.term": query, "pageSize": limit, "sort": "LastUpdatePostDate:desc"},
    )
    results: list[Evidence] = []
    for item in response.json().get("studies", []):
        protocol = item.get("protocolSection", {})
        identity = protocol.get("identificationModule", {})
        description = protocol.get("descriptionModule", {})
        status = protocol.get("statusModule", {})
        nct_id = identity.get("nctId", "")
        title = identity.get("briefTitle", "Clinical trial")
        summary = description.get("briefSummary", "")
        overall_status = status.get("overallStatus", "Unknown")
        if nct_id and summary:
            results.append(Evidence("临床试验", title, f"状态：{overall_status}。{summary[:900]}", f"https://clinicaltrials.gov/study/{nct_id}", nct_id))
    return results


def select_local_evidence(question: str, limit: int = 4) -> list[Evidence]:
    topic_terms = {
        "高血压": {"高血压", "血压", "降压", "hypertension"},
        "血脂": {"血脂", "胆固醇", "ldl", "他汀", "dyslip"},
        "糖尿病": {"糖尿病", "血糖", "diabetes"},
        "心脑血管": {"心血管", "脑血管", "卒中", "心梗", "stroke"},
    }
    question_lower = question.lower()
    terms = {term for values in topic_terms.values() for term in values if term in question_lower}
    terms.update(re.findall(r"[A-Za-z]+", question_lower))
    scored: list[tuple[int, Evidence]] = []
    for evidence in LOCAL_EVIDENCE:
        haystack = f"{evidence.title} {evidence.summary}".lower()
        score = sum(2 if term in evidence.summary.lower() else 1 for term in terms if term in haystack)
        scored.append((score, evidence))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored if item[0] > 0]
    return (selected or LOCAL_EVIDENCE[:2])[:limit]


def search_local_corpus(question: str, limit: int = 3) -> list[Evidence]:
    if not CORPUS_PATH.exists():
        return []
    try:
        payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z]{3,}", build_english_query(question))
        if term.lower() not in {"and", "the", "for", "with"}
    }
    ranked: list[tuple[int, dict[str, str]]] = []
    for item in payload.get("documents", []):
        haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        score = sum(3 if term in item.get("title", "").lower() else 1 for term in query_terms if term in haystack)
        if score:
            ranked.append((score, item))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        Evidence(
            source_type=str(item.get("source_type") or "PubMed"),
            title=str(item.get("title") or ""),
            summary=str(item.get("summary") or ""),
            url=str(item.get("url") or ""),
            identifier=str(item.get("identifier") or item.get("source_id") or ""),
            year=int(item["year"]) if item.get("year") else None,
            organization=str(item.get("organization") or ""),
        )
        for _, item in ranked[:limit]
    ]


def retrieve_evidence(question: str, online: bool = True) -> tuple[list[Evidence], list[str]]:
    evidence = select_local_evidence(question)
    evidence.extend(search_local_corpus(question))
    warnings: list[str] = []
    if online:
        query = build_english_query(question)
        for label, searcher in (("PubMed", search_pubmed), ("ClinicalTrials.gov", search_trials)):
            try:
                evidence.extend(searcher(query))
            except (requests.RequestException, ValueError, ET.ParseError) as exc:
                warnings.append(f"{label} 暂时不可用，已使用本地证据：{type(exc).__name__}")

    unique: dict[str, Evidence] = {}
    for item in evidence:
        unique[item.identifier] = item
    return list(unique.values())[:8], warnings
