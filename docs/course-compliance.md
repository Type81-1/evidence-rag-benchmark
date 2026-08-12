# D1 课件要求对照

本表以 D1AM 与 D1PM 的可验收要求为准。机器可验证项由 `python course_compliance.py` 和 `GET /api/course-compliance` 复核。

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 选定赛道、提出可证伪假设 | 赛道三；README 预注册三条方向性假设 | 已完成 |
| 至少 500 条公开证据记录 | `data/pubmed_corpus.json` 含 500 条去重 PubMed 记录 | 已完成 |
| metadata 可追溯 | 每条含 source_id、title、summary、year、organization、url、source_type | 已完成 |
| 冻结测试题与对抗/拒答场景 | 两个领域各 8 题，包含证据不足、处方越界和急症升级 | 已完成 |
| passage 分块与定位 | 长材料约 400 token、80 token overlap；保存 chunk/source ID、token 数和字符区间 | 已完成 |
| 最小 RAG 与 Top-K | 500 条来源加人工核查证据进入 passage 候选池，输出 Top 3 | 已完成 |
| 混合召回、融合与重排 | BM25 + TF-IDF，RRF 融合 30 条候选，独立重排与 MMR | 已完成 |
| 互补证据 | Top-3 报告 overview、causal、boundary 角色并避免同源重复 | 已完成 |
| 引用回查与幻觉检测 | evidence map 验证 chunk ID、source ID、注册 ID/URL 和句子词项支持 | 已完成 |
| metadata 查询过滤 | 按领域、年份、证据类型生成过滤器；候选不足时审计式放宽 | 已完成 |
| 查询改写 | 保留原查询，增加领域和证据类型扩展词，并行融合而非覆盖原意图 | 已完成 |
| 多轮检索 | 首轮发现证据类型/角色缺口后生成 gap query，第二轮融合候选 | 已完成 |
| Wiki 闭环 | 主题页创建、幂等 ingest、更新历史、query、引用/URL/过期/孤立 lint | 已完成 |
| 自动化评估 | 16 题逐项执行进阶能力，并验证 Wiki 创建→去重→更新→查询→lint | 已完成 |
| 合格拒答 | 输出已检索、缺失证据、下一步；急症单独升级 | 已完成 |
| 公平 A/B 与劣化实验 | A 裸模型，B 正常 RAG，C 噪声 RAG，D 缺失检索 | 已完成 |
| 三路评估框架 | 程序化指标、可选 LLM judge、人工盲评录入与汇总 | 已完成 |
| AI 披露、无 PHI、非诊疗边界 | 页面、README、Prompt 和门控均显式声明 | 已完成 |

以下不是代码可以代替的验收证据，目前仍未完成：

1. 使用真实模型、冻结题集和预注册重复次数完成正式批跑。
2. 至少两名合格评审者独立完成盲评。
3. 基于真实盲评数据报告一致性，并在此之前不宣称 RAG 或专用系统更优。

课件中的进阶与挑战能力可以组合，图示表达实施优先级而非互斥选择。本项目现已完成图中全部进阶建议与挑战选做；`python advanced_evaluation.py` 会实际运行每项能力并在任一项失败时返回非零状态。
