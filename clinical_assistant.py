from __future__ import annotations

import os
import re

from evidence_service import Evidence, retrieve_evidence


SAFETY_NOTE = "本回答用于临床证据检索与教学，不替代诊断、处方或停药建议；具体决策应由临床医生结合患者情况作出。"


def _extractive_answer(question: str, evidence: list[Evidence], warnings: list[str]) -> str:
    if not evidence:
        return f"现有来源中没有检索到足够证据，暂时无法回答。\n\n{SAFETY_NOTE}"

    lines = ["## 证据总结", ""]
    for index, item in enumerate(evidence[:5], start=1):
        summary = item.summary if len(item.summary) <= 600 else f"{item.summary[:600].rstrip()}……"
        lines.append(f"- {summary} [{index}]")
    lines.extend(["", "## 临床解读", ""])
    if "高血压" in question or "血压" in question:
        lines.append(
            "高血压往往是长期存在的风险状态。药物控制的是血压及其相关风险，并不等同于消除病因；"
            "如果停药后血压再次升高，卒中、心肌梗死、心衰和肾损害风险会随之增加，因此不少患者需要长期治疗 [1]。"
        )
    elif "血脂" in question or "胆固醇" in question:
        lines.append(
            "生活方式干预是基础，但是否需要他汀等药物取决于 LDL-C 和总体心血管风险。"
            "二者通常不是互相替代，而是按风险分层组合使用 [1][2]。"
        )
    else:
        lines.append("现有证据支持按疾病风险、治疗获益和不良反应进行个体化决策，不能仅依据单一指标下结论 [1]。")

    lines.extend(["", "## 证据局限", ""])
    lines.append("当前回答包含指南摘要和检索结果，但未完成针对具体患者的系统综述、偏倚评价或证据分级。")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## 参考证据", ""])
    for index, item in enumerate(evidence[:5], start=1):
        lines.append(f"[{index}] {item.source_type}：[{item.title}]({item.url})（{item.identifier}）")
    lines.extend(["", "## 安全提示", "", SAFETY_NOTE])
    return "\n".join(lines)


def _llm_answer(question: str, evidence: list[Evidence]) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[{index}] {item.title}\n{item.summary}\nURL: {item.url}"
        for index, item in enumerate(evidence, start=1)
    )
    prompt = f"""你是谨慎的临床证据助手。只使用给出的证据回答，不得补造事实、数字或引用。
每个关键结论必须引用 [1] 形式的编号。证据不足时明确说明。不提供个人处方或停药建议。

问题：{question}

证据：
{context}

用中文输出：证据总结、临床解读、证据局限、安全提示。"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), input=prompt)
    answer = response.output_text.strip()
    allowed = {str(index) for index in range(1, len(evidence) + 1)}
    cited = set(re.findall(r"\[(\d+)]", answer))
    if not answer or not cited or not cited.issubset(allowed):
        raise ValueError("大模型回答未通过引用校验")
    references = ["", "## 参考证据", ""]
    for index in sorted(cited, key=int):
        item = evidence[int(index) - 1]
        references.append(f"[{index}] {item.source_type}：[{item.title}]({item.url})（{item.identifier}）")
    return answer + "\n" + "\n".join(references) + f"\n\n## 安全提示\n\n{SAFETY_NOTE}"


def answer_question(question: str, online: bool = True, use_llm: bool = True) -> dict[str, object]:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")
    evidence, warnings = retrieve_evidence(question, online=online)
    mode = "extractive"
    answer = _extractive_answer(question, evidence, warnings)
    if use_llm and os.getenv("OPENAI_API_KEY") and os.getenv("LLM_MODE", "auto").lower() != "extractive":
        try:
            answer = _llm_answer(question, evidence)
            mode = "openai"
        except Exception as exc:
            warnings.append(f"大模型不可用，已切换到证据摘录模式：{type(exc).__name__}")
            answer = _extractive_answer(question, evidence, warnings)
    return {"question": question, "answer": answer, "evidence": [item.to_dict() for item in evidence], "warnings": warnings, "mode": mode}
