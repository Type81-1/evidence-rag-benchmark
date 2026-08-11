from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from statistics import mean


SAFETY_NOTE = "仅用于营养证据教学与方法评测，不替代医生或注册营养师的个体化建议。"


@dataclass(frozen=True)
class Evidence:
    id: str
    topic: str
    title: str
    organization: str
    year: int
    summary: str
    url: str
    quality: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EVIDENCE = [
    Evidence("S1", "sodium", "Guideline for the pharmacological treatment of hypertension in adults", "WHO", 2021, "减少膳食钠摄入是高血压和心血管风险管理中推荐的生活方式措施之一；它应与整体饮食和必要的医疗管理结合。", "https://www.who.int/publications/i/item/9789240033986", "国际指南"),
    Evidence("S2", "sodium", "Effects on blood pressure of reduced dietary sodium and the DASH diet", "DASH-Sodium Collaborative Research Group", 2001, "随机对照试验显示，降低钠摄入和 DASH 饮食均可降低血压，两者结合时效果更明显；不同人群反应存在差异。", "https://pubmed.ncbi.nlm.nih.gov/11136953/", "随机对照试验"),
    Evidence("S3", "mediterranean", "Primary Prevention of Cardiovascular Disease with a Mediterranean Diet", "PREDIMED Investigators", 2018, "在心血管高风险成人中，添加特级初榨橄榄油或坚果的地中海饮食组主要心血管事件发生率低于对照饮食组。", "https://pubmed.ncbi.nlm.nih.gov/29897866/", "随机对照试验"),
    Evidence("S4", "mediterranean", "Healthy diet", "WHO", 2020, "健康饮食强调水果、蔬菜、豆类、坚果和全谷物，并限制游离糖、盐和不健康脂肪；饮食模式应结合当地食物文化。", "https://www.who.int/news-room/fact-sheets/detail/healthy-diet", "公共卫生指南"),
    Evidence("S5", "fiber", "Carbohydrate quality and human health", "The Lancet", 2019, "系统综述与荟萃分析发现，较高膳食纤维和全谷物摄入与较低的全因死亡及多种非传染性疾病风险相关，但个体耐受和总体饮食结构仍需考虑。", "https://pubmed.ncbi.nlm.nih.gov/30638909/", "系统综述"),
    Evidence("S6", "sugar", "WHO guideline: sugars intake for adults and children", "WHO", 2015, "WHO 建议成人和儿童将游离糖摄入降至总能量摄入的 10% 以下，进一步降至 5% 以下可带来额外益处；重点是游离糖而非完整水果中的内源糖。", "https://www.who.int/publications/i/item/9789241549028", "国际指南"),
    Evidence("S7", "supplement", "Vitamin and Mineral Supplementation to Prevent CVD and Cancer", "USPSTF", 2022, "对于一般社区成人，现有证据不足以判断多数单一或复合维生素补充剂预防心血管病或癌症的获益与风险；不应将补充剂等同于均衡饮食。", "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/vitamin-supplementation-to-prevent-cvd-and-cancer-preventive-medication", "循证推荐"),
    Evidence("S8", "protein", "Dietary protein intake and human health", "Food & Function", 2016, "蛋白质需求受年龄、活动水平和健康状况影响。食物来源、总能量和整体饮食模式比单独追求高蛋白数字更重要。", "https://pubmed.ncbi.nlm.nih.gov/26797090/", "叙述性综述"),
]


QUESTIONS: list[dict[str, object]] = [
    {"id": "NUT-01", "topic": "限钠饮食", "question": "限钠饮食对高血压是否真的有帮助？", "expected_claims": ["降低血压", "个体差异", "整体饮食"], "evidence_topics": ["sodium"]},
    {"id": "NUT-02", "topic": "地中海饮食", "question": "地中海饮食能降低心血管风险吗？证据有多可靠？", "expected_claims": ["心血管事件", "随机对照试验", "高风险成人"], "evidence_topics": ["mediterranean"]},
    {"id": "NUT-03", "topic": "膳食纤维", "question": "多吃膳食纤维对心血管健康有什么证据？", "expected_claims": ["全谷物", "风险相关", "总体饮食"], "evidence_topics": ["fiber", "mediterranean"]},
    {"id": "NUT-04", "topic": "游离糖", "question": "为了健康，水果和添加糖都应该一概戒掉吗？", "expected_claims": ["游离糖", "完整水果", "10%"], "evidence_topics": ["sugar"]},
    {"id": "NUT-05", "topic": "营养补充剂", "question": "健康成年人每天吃复合维生素能预防心血管病吗？", "expected_claims": ["证据不足", "不能替代", "均衡饮食"], "evidence_topics": ["supplement"]},
    {"id": "NUT-06", "topic": "蛋白质", "question": "所有成年人都应该采用高蛋白饮食吗？", "expected_claims": ["个体需求", "食物来源", "整体饮食"], "evidence_topics": ["protein"]},
]


def get_question(question_id: str, questions: list[dict[str, object]] = QUESTIONS) -> dict[str, object]:
    return next(item for item in questions if item["id"] == question_id)


def retrieve(case: dict[str, object], condition: str, corpus: list[Evidence] = EVIDENCE) -> list[Evidence]:
    topics = set(case["evidence_topics"])
    relevant = [item for item in corpus if item.topic in topics]
    if condition == "good":
        return relevant[:3]
    if condition == "missing":
        return []
    unrelated = [item for item in corpus if item.topic not in topics]
    # 保留一条弱相关证据并加入两条高表面可信度噪声，模拟召回或重排失败。
    return (unrelated[:2] + relevant[-1:])[:3]


def _offline_baseline(case: dict[str, object], safety_note: str = SAFETY_NOTE) -> str:
    topic = case["topic"]
    claims = "、".join(case["expected_claims"][:2])
    return (
        f"## 结论\n\n关于{topic}，通用知识通常会讨论{claims}，但具体结论取决于适用人群和个人情况。\n\n"
        "## 依据与局限\n\n这是基于模型已有知识的概括，没有为本次回答检索或核验原始来源，因此不能确认具体数字、适用人群或资料版本。\n\n"
        f"## 安全边界\n\n{safety_note}"
    )


def _offline_rag(case: dict[str, object], evidence: list[Evidence], condition: str, safety_note: str = SAFETY_NOTE) -> str:
    if not evidence:
        return f"## 结论\n\n当前知识库没有检索到足以支持该问题的证据，因此拒绝给出确定答案。\n\n## 下一步\n\n建议扩大检索范围或由人工核查指南。\n\n## 安全边界\n\n{safety_note}"
    bullets = "\n".join(f"- {item.summary} [{item.id}]" for item in evidence)
    caveat = "检索结果与问题高度相关。" if condition == "good" else "检索结果包含主题不匹配的材料，结论可能被噪声带偏。"
    return (
        f"## 证据回答\n\n{bullets}\n\n## 检索诊断\n\n{caveat}\n\n"
        f"## 安全边界\n\n{safety_note}"
    )


def _call_model(case: dict[str, object], evidence: list[Evidence] | None, domain_label: str = "营养健康") -> str:
    from openai import OpenAI

    question = case["question"]
    if evidence is None:
        prompt = f"""你是通用大模型。请基于已有知识回答以下{domain_label}问题。给出结论、依据、局限和安全边界。若写引用，使用 [G1] 编号。不要假装已经检索数据库。\n\n问题：{question}"""
    else:
        context = "\n\n".join(f"[{item.id}] {item.title}\n{item.summary}\n{item.url}" for item in evidence) or "（没有检索结果）"
        prompt = f"""你是{domain_label}证据 RAG 系统。只能使用检索结果回答；关键结论使用证据编号引用。证据不足时明确拒答，不得用模型记忆补齐。输出结论、证据、局限和安全边界。\n\n问题：{question}\n\n检索结果：\n{context}"""
    response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), input=prompt
    )
    return response.output_text.strip()


def score(case: dict[str, object], answer: str, evidence: list[Evidence] | None) -> dict[str, float]:
    expected = [str(item).lower() for item in case["expected_claims"]]
    coverage = mean(float(term in answer.lower()) for term in expected)
    cited_ids = set(re.findall(r"\[([A-Z]\d+)]", answer))
    allowed = {item.id for item in (evidence or [])}
    relevant_topics = set(case["evidence_topics"])
    relevant = {item.id for item in (evidence or []) if item.topic in relevant_topics}
    supported = cited_ids & allowed & relevant
    citation_precision = len(supported) / len(cited_ids) if cited_ids else 0.0
    citation_coverage = min(1.0, len(supported) / max(1, len(expected)))
    unsupported_rate = (len(cited_ids - allowed) / len(cited_ids)) if cited_ids else 0.0
    refusal = float("拒绝" in answer or "不足以支持" in answer or "没有检索到" in answer)
    return {
        "claim_coverage": round(coverage, 2),
        "citation_precision": round(citation_precision, 2),
        "citation_coverage": round(citation_coverage, 2),
        "unsupported_citation_rate": round(unsupported_rate, 2),
        "appropriate_refusal": refusal,
    }


def compare_question(question_id: str, condition: str = "good", live: bool = False) -> dict[str, object]:
    case = get_question(question_id)
    return compare_case(case, EVIDENCE, condition, live=live)


def compare_case(
    case: dict[str, object],
    corpus: list[Evidence],
    condition: str,
    *,
    live: bool = False,
    domain_label: str = "营养健康",
    safety_note: str = SAFETY_NOTE,
) -> dict[str, object]:
    evidence = retrieve(case, condition, corpus)
    baseline_answer = _call_model(case, None, domain_label) if live else _offline_baseline(case, safety_note)
    rag_answer = _call_model(case, evidence, domain_label) if live else _offline_rag(case, evidence, condition, safety_note)
    return {
        "question": case,
        "condition": condition,
        "run_mode": "live_model" if live else "reproducible_demo",
        "baseline": {"answer": baseline_answer, "metrics": score(case, baseline_answer, None)},
        "rag": {"answer": rag_answer, "metrics": score(case, rag_answer, evidence), "evidence": [item.to_dict() for item in evidence]},
    }


def run_benchmark() -> dict[str, object]:
    return run_cases_benchmark(QUESTIONS, EVIDENCE)


def run_cases_benchmark(
    questions: list[dict[str, object]],
    corpus: list[Evidence],
    *,
    domain_label: str = "营养健康",
    safety_note: str = SAFETY_NOTE,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for case in questions:
        for condition in ("good", "noisy", "missing"):
            result = compare_case(case, corpus, condition, domain_label=domain_label, safety_note=safety_note)
            rows.append({"question_id": case["id"], "topic": case["topic"], "condition": condition, "baseline": result["baseline"]["metrics"], "rag": result["rag"]["metrics"]})
    summary: dict[str, dict[str, float]] = {}
    for condition in ("baseline", "good", "noisy", "missing"):
        metric_rows = [row["baseline"] for row in rows if row["condition"] == "good"] if condition == "baseline" else [row["rag"] for row in rows if row["condition"] == condition]
        summary[condition] = {name: round(mean(float(row[name]) for row in metric_rows), 2) for name in metric_rows[0]}
    return {"run_mode": "reproducible_demo", "question_count": len(questions), "comparison_count": len(rows), "summary": summary, "cases": rows}
