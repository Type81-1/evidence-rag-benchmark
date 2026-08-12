# D2AM 课件要求对照

本表依据 `D2AM.pdf` 全部 52 个图片页逐页视觉复核，覆盖基础、进阶和挑战最高档。机器验收入口为 `python d2_evaluation.py`、`python course_compliance.py` 和 `GET /api/d2-compliance`。

| D2AM 要求 | 实现与可复核证据 | 状态 |
| --- | --- | --- |
| 基础：至少 8 条评估记录 | `d2_evaluation.py` 固定 8 条 Golden Set，覆盖普通、证据不足、个体化越界、急症与检索缺失 | 已实现 |
| Benchmark：同底座 A/B，固定题集、配置、评分口径、保留失败 | 两域各 15 题；A/B/C/D；报告记录版本、哈希、原始输出、失败与 trace | 已实现 |
| Tool：一事一工具、结构化返回、描述清楚、最小权限 | `d2_agent.py` 提供 `search_evidence`、`verify_citations`、`assess_evidence_grade`；JSON Schema 禁止额外参数，限制 5 条结果、超时与读权限 | 已实现 |
| Skill：稳定判断标准、按任务加载、可复用 | `evidence-grade-v1` 仅在 Agent 证据分级阶段加载，不把任务规则塞入主 Prompt | 已实现 |
| 工作流：固定步骤优先、可复现可调试 | 原 A/B 检索生成质检流程保留；D2 Agent 只用于需要动态 answer/abstain/escalate 的局部路径 | 已实现 |
| Agent：行动→Observation→下一步，最多 3 轮 | `run_agent` 依次执行检索、证据分级、引用核验；trace 只记行动、观察和决策，不记录私有思维链 | 已实现 |
| Agent 护栏：边界、步数、工具白名单、失败可见 | 冻结问题 ID、登记语料只读、工具白名单、最多 3 步、结构化错误、急症升级 | 已实现 |
| 四层评估：检索、引用与证据、回答、行为与边界 | D2 报告逐题保存四个布尔检查、answer hash、完整 trace 与失败样本 | 已实现 |
| 规则、Judge、人工各管一段 | 程序规则与六项 Rubric 已实现；可选盲法 LLM judge 已实现；人工盲评录入、去重、版本和一致性摘要已实现 | 管线已实现 |
| 多 Agent：至少两角色、边界清晰、完整样例链 | Researcher 检索/分级，Writer 只组织证据，Critic 无检索权限且核验引用；保留证据包→草稿→评审→终稿链 | 已实现 |
| 多 Agent 成本与收益 | 样例报告 model/tool call、步数、是否修订；只在高价值完整链使用 | 已实现 |

## 不能由代码代替的正式验收

以下外部步骤仍需实际执行，项目不会用离线结果冒充：

1. 配置有效模型 API，按预注册重复次数完成真实 A/B/C/D 批跑。
2. 至少两名合格真人评审独立完成盲评。
3. 基于真实评分计算 ICC 或加权 kappa，并完成高风险医学内容人工校准。

在上述步骤完成前，只能声明工程管线和离线验收通过，不能声明 RAG 或 Agent 的真实效果优于基线。
