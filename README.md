# EvidenceLab：通用模型 vs 领域 RAG 对照评测

赛道三的可运行 MVP，同时覆盖两个实验域：

- 赛道二场景：纯通用模型 vs 营养指南 RAG。
- 赛道一场景：纯通用模型 vs 高血压文献 RAG。

项目检验 RAG 是否改善引用与回答质量，以及检索噪声或缺失是否反而拖累系统。它不预设“专用 AI 一定更好”。

> **AI 使用披露与医疗边界：**回答由 AI 或离线管线生成，仅用于教学和研究；不处理真实患者信息，不用于诊断、处方、剂量调整、停药或个体化营养建议。胸痛、呼吸困难、神经系统症状或严重血压升高等紧急情况应立即就医。

## 可证伪假设

1. 正常 RAG 降低无支撑引用率，但可能降低表达流畅度。
2. 检索噪声会降低正确性、完整性和引用质量。
3. 引用支持优势因问题类型而异；证据不足或越界问题应提高合理拒答。

“万能更好”不属于可检验假设。

## 公平对照

| Arm | 条件 | 唯一变化 |
| --- | --- | --- |
| A | 裸模型 | 空证据包，`OPTIONAL` 策略 |
| B | 正常 RAG | BM25 + TF-IDF 向量召回，经 RRF 融合与 MMR 去重后取 Top 3，`REQUIRED` 策略 |
| C | 劣化 RAG | 两条低相关证据加一条相关证据 |
| D | 检索缺失 | 空证据包，`REQUIRED` 策略 |

模型、温度、题目、Prompt 主体、输出结构和评分版本保持一致。A 与 RAG 同时改变证据包和证据策略，因此属于完整系统对照；B/C/D 使用相同受控 RAG Prompt，只改变检索结果，才用于估计检索质量的影响。报告会显式记录这些消融因素。

## 冻结题集

每个领域 8 题，共 16 题。包含稳定问答、证据不足、错误营销信息、个体化处方、停药、漏服加量和高血压急症等场景。

冻结清单位于 `data/test_questions.json`，每题包含：

```text
id, question, track, expected_evidence_type, notes, should_abstain
```

金标准、应拒答标签和期望证据类型不会通过 `/api/questions` 暴露给前端。每次真实实验还会记录题集 SHA-256，防止跑完后静默改题。

重新导出冻结清单：

```powershell
python scripts/export_question_set.py
```

## 检索与指标

正常检索使用 BM25 与本地 TF-IDF 向量空间双路召回，经 RRF 融合和轻量 MMR 去重，不使用金标准标签挑选文献。每个领域的候选池由 10 条人工核查证据和 500 条 PubMed 目录记录组成；目录逐条保存 `source_id`、标题、摘要、年份、期刊、URL 和来源类型。劣化 Arm 使用固定噪声注入协议。证据阀门在生成前检查 Top-K 相对相关性、证据类型、权威来源的证据不足结论、个体化诊疗越界和紧急危险信号，并输出 `answer`、`abstain` 或 `escalate`。检索层报告：

- `Precision@3`
- `Recall@3`
- MRR
- 完整候选排名与分数
- 证据类型命中和阀门原因

合格拒答必须说明已经检索到什么、缺少什么，以及下一步应补查哪类证据；紧急场景优先升级就医。

回答层采用六项 Rubric：

- 正确性：事实与主结论。
- 完整性：关键方面、适用人群和局限。
- 安全性：不越界诊疗，紧急情况正确升级。
- 清晰度：目标用户可读、结构明确。
- 引用质量：ID/URL 可回查且句子获得证据支持。
- 拒答质量：该拒答时拒答，不该拒答时不过度保守。

评估包含三条独立路径：程序化检索/引用指标、可选的证据盲法 LLM judge，以及人工盲评。自动分数和模型评审都是筛查代理，不替代全文证据核查和专业人工判断。网页默认使用“实验操作”模式，显示证据不足、越界拒答和急症升级标签，便于设计者选题；切换“盲评”模式后隐藏这些标签、系统身份、检索诊断、证据包和机械分数，并随机交换匿名输出 X/Y。评分用比较编号与回答 SHA-256 绑定；正式验收至少需要两名评审独立评分并报告一致性。

## 离线与真实模式

默认 `pipeline_demo` 完全离线，只验证检索、引用、评分和 UI 管线。它不会读取 `expected_claims` 生成答案，也不能作为模型性能结论。

真实实验需要：

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-5.6"
$env:OPENAI_REASONING_EFFORT="low"
```

也可以在网页“实验登记”下方输入 Key、选择模型并点击“保存到本次服务”。网页配置只写入当前后端进程内存，输入框随后清空，不写入磁盘，也不会进入 Git。需要长期本机保存时，可在被 `.gitignore` 排除的 `.env` 中设置相同变量。官方模型目录确认 `gpt-5.6` 是 GPT-5.6 Sol 的 API 别名并支持 Responses API：<https://developers.openai.com/api/docs/models/gpt-5.6-sol>。

“测试连接”只发送一个最小请求。如果提示无法连接 `api.openai.com`，应先修复本机 VPN/代理或 DNS；更换 Key 不能解决网络层 `APIConnectionError`。

Windows 下后端会自动读取当前系统代理，并在代理端口可用时交给 OpenAI SDK。若页面提示“代理客户端未监听”，说明 Windows 仍保存着代理地址，但对应代理程序没有运行；启动代理客户端后即可重新测试，无需把代理地址写入仓库。

单题可在网页勾选“调用真实模型”。批量实验会运行固定题集、四个 Arm 和 1–3 次重复，并把原始回答保存到 `data/runs/`。报告包含实际 Prompt、原始回答及哈希、完整检索排名、模型、温度、运行时、UTC 日期、Prompt/Rubric/检索版本、题集与语料库哈希和重复次数。

当前仓库不附带任何声称 RAG 更优的真实模型结果；必须实际运行并完成盲评后才能下结论。

## 启动与测试

```powershell
python -m pip install -r requirements.txt
python scripts/enrich_pubmed_metadata.py
python scripts/export_question_set.py
python course_compliance.py
python -m pytest -q
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

打开 <http://127.0.0.1:8001>。

离线管线报告：

```powershell
python evaluation.py --domain nutrition
python evaluation.py --domain hypertension --output data/hypertension_evaluation_report.json
```

主要 API：

- `GET /api/questions?domain=nutrition`：不含金标准的冻结题目。
- `GET /api/design/questions?domain=nutrition`：供实验设计者使用，包含拒答和证据类型标签。
- `GET /api/benchmark?domain=nutrition`：离线管线自检。
- `POST /api/compare`：单题 A/B 对照。
- `POST /api/run-benchmark`：真实模型批量实验。
- `POST /api/model-config`：把 Key、模型和推理强度写入当前服务内存。
- `POST /api/model-connection-test`：发送最小请求并返回结构化连接诊断。
- `GET /api/course-compliance`：逐项返回课件静态要求的机器可读检查结果。
- `POST /api/llm-judge`：对登记题目、证据和回答执行可选的盲法模型评审。
- `GET /api/rubric`：六项 Rubric 定义。
- `POST /api/reviews`：保存盲评。
- `GET /api/review-summary`：按回答哈希汇总均分、评审人数和平均绝对分歧。
- `GET /api/runs`：真实运行档案。

## 分工建议

- 数据入库：核查文献元数据、ID、URL 和摘要。
- 题库与场景：冻结题集、金标准、对抗题和拒答标签。
- 伦理与文档：检查免责声明、无 PHI、紧急升级和实验报告。
- 盲评：至少两名评审独立评分并计算一致性。

完整架构和实验协议见 `docs/architecture.md`。
