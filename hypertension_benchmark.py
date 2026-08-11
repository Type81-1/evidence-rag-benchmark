from __future__ import annotations

from nutrition_benchmark import Evidence, compare_case, get_question, run_cases_benchmark


SAFETY_NOTE = "仅用于临床证据教学与方法评测，不替代诊断、处方或停药建议；具体决策应由医生结合患者情况作出。"


EVIDENCE = [
    Evidence("H1", "long_term", "2024 ESC Guidelines for the management of elevated blood pressure and hypertension", "European Society of Cardiology", 2024, "高血压通常需要长期管理。治疗目标是持续降低卒中、心肌梗死、心力衰竭和肾脏事件风险；生活方式、药物选择和目标值应结合总体风险与耐受性。", "https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/Elevated-Blood-Pressure-and-Hypertension", "临床指南"),
    Evidence("H2", "long_term", "Guideline for the pharmacological treatment of hypertension in adults", "WHO", 2021, "WHO 建议达到治疗指征的成人接受降压药物治疗，并根据血压反应、合并症和风险持续随访；药物方案需要规范管理而非自行调整。", "https://www.who.int/publications/i/item/9789240033986", "国际指南"),
    Evidence("H3", "intensive", "A Randomized Trial of Intensive versus Standard Blood-Pressure Control", "SPRINT Research Group", 2015, "在部分心血管高风险且无糖尿病的成人中，强化收缩压控制降低主要心血管事件和全因死亡，但低血压、电解质异常和急性肾损伤等不良事件更常见。", "https://pubmed.ncbi.nlm.nih.gov/26551272/", "随机对照试验"),
    Evidence("H4", "older", "Trial of Intensive Blood-Pressure Control in Older Patients with Hypertension", "STEP Study Group", 2021, "在中国 60 至 80 岁高血压患者中，强化收缩压目标降低复合心血管结局，但低血压发生更多，说明老年人目标仍需结合耐受性。", "https://pubmed.ncbi.nlm.nih.gov/34491661/", "随机对照试验"),
    Evidence("H5", "measurement", "2017 ACC/AHA Guideline for High Blood Pressure in Adults", "ACC/AHA", 2018, "高血压判断应基于规范测量和多次读数，并推荐在适当情况下使用家庭或动态血压监测确认诊断、识别白大衣或隐匿性高血压。", "https://pubmed.ncbi.nlm.nih.gov/29133356/", "临床指南"),
    Evidence("H6", "measurement", "2021 European Society of Hypertension practice guidelines for office and out-of-office blood pressure measurement", "European Society of Hypertension", 2021, "诊室、家庭和动态血压测量都需要标准化流程。单次读数容易受休息、姿势、袖带和环境影响，不足以单独决定长期治疗。", "https://pubmed.ncbi.nlm.nih.gov/33710173/", "测量指南"),
    Evidence("H7", "lifestyle", "Effects of comprehensive lifestyle modification on blood pressure control", "PREMIER Collaborative Research Group", 2003, "综合生活方式干预可以降低血压，包括体重管理、限钠、运动和 DASH 型饮食；是否能替代药物取决于基线血压和总体心血管风险。", "https://pubmed.ncbi.nlm.nih.gov/12709466/", "随机对照试验"),
    Evidence("H8", "safety", "2018 ESC/ESH Guidelines for the management of arterial hypertension", "ESC/ESH", 2018, "降压治疗需要随访疗效和不良反应。血压控制正常通常代表治疗有效，不能据此推断高血压已经消失或自行停药是安全的。", "https://pubmed.ncbi.nlm.nih.gov/30165516/", "临床指南"),
]


QUESTIONS: list[dict[str, object]] = [
    {"id": "HTN-01", "topic": "长期治疗", "question": "高血压患者为什么常常需要长期服药？", "expected_claims": ["长期管理", "心血管风险", "持续随访"], "evidence_topics": ["long_term"]},
    {"id": "HTN-02", "topic": "生活方式", "question": "生活方式干预能完全替代降压药吗？", "expected_claims": ["降低血压", "总体心血管风险", "不能一概替代"], "evidence_topics": ["lifestyle", "long_term"]},
    {"id": "HTN-03", "topic": "强化降压", "question": "强化降压治疗有哪些潜在获益和风险？", "expected_claims": ["心血管事件", "全因死亡", "不良事件"], "evidence_topics": ["intensive"]},
    {"id": "HTN-04", "topic": "血压测量", "question": "为什么不能根据一次诊室血压决定长期治疗？", "expected_claims": ["多次读数", "家庭血压", "标准化测量"], "evidence_topics": ["measurement"]},
    {"id": "HTN-05", "topic": "老年高血压", "question": "老年高血压患者是否都应该追求更低的血压目标？", "expected_claims": ["心血管结局", "低血压", "耐受性"], "evidence_topics": ["older", "intensive"]},
    {"id": "HTN-06", "topic": "停药安全", "question": "血压已经正常了，可以马上自行停用降压药吗？", "expected_claims": ["不能自行停药", "治疗有效", "医生随访"], "evidence_topics": ["safety", "long_term"]},
]


def compare_question(question_id: str, condition: str = "good", live: bool = False) -> dict[str, object]:
    case = get_question(question_id, QUESTIONS)
    return compare_case(case, EVIDENCE, condition, live=live, domain_label="高血压临床", safety_note=SAFETY_NOTE)


def run_benchmark() -> dict[str, object]:
    return run_cases_benchmark(QUESTIONS, EVIDENCE, domain_label="高血压临床", safety_note=SAFETY_NOTE)
