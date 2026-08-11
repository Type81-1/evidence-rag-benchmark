# NutriEvidence：专用 AI vs 通用大模型评测台

赛道三的可运行 MVP。它支持饮食营养和高血压临床两个实验域，分别对比纯通用大模型与领域 RAG，并专门测试检索噪声和检索缺失是否拖累回答。

> 这是教学与研究原型，不用于诊断、治疗或个体化营养处方。

## 实验设计

控制变量是问题、模型与回答要求；自变量是知识上下文：

| 条件 | 给模型的上下文 | 用途 |
| --- | --- | --- |
| `baseline` | 不提供检索证据 | 通用模型基线 |
| `good` | 主题匹配的指南和研究 | 测试 RAG 的理想收益 |
| `noisy` | 混入主题不匹配的高可信资料 | 测试检索是否拖累回答 |
| `missing` | 不提供可用证据 | 测试系统能否合理拒答 |

营养固定测试集包含限钠、地中海饮食、膳食纤维、游离糖、补充剂和蛋白质 6 个主题。高血压固定测试集包含长期治疗、生活方式、强化降压、血压测量、老年患者和停药安全 6 个主题。内置证据来自 WHO、ESC、ACC/AHA、USPSTF 以及 PubMed 收录的 SPRINT、STEP 等研究。

核心指标：

- 主张覆盖率：预先定义的关键答案点覆盖程度。
- 引用精确率：回答中的引用有多少既来自本次检索结果，又与问题主题相关。
- 引用覆盖率：关键答案点获得可追溯证据的程度。
- 无支撑引用率：引用无法映射到检索上下文的比例。
- 合理拒答率：检索缺失时系统是否避免无依据作答。

## 两种运行模式

默认的 `reproducible_demo` 完全离线，用确定性回答验证实验流程、界面和指标。它不是模型性能结论。

配置 `OPENAI_API_KEY` 后，可在单题实验室勾选“调用真实模型”。两条路径使用同一个 `OPENAI_MODEL`，区别仅在于 RAG 路径获得检索上下文。

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-5-mini"
```

正式报告应重复运行真实模型、记录模型快照和参数，并由至少两名盲评者评价正确性与引用支持度。当前自动指标适合筛查，不替代营养专业人工审核。

## 启动

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

打开 <http://127.0.0.1:8001>。

## 运行评测与测试

```powershell
python evaluation.py --domain nutrition
python evaluation.py --domain hypertension --output data/hypertension_evaluation_report.json
python -m pytest -q
```

评测报告写入 `data/evaluation_report.json`。主要 API：

- `GET /api/benchmark?domain=hypertension`：指定领域的四条件离线总览。
- `GET /api/questions?domain=hypertension`：指定领域的固定问题及预期答案点。
- `POST /api/compare`：运行一组通用模型与 RAG 对照。
- `GET /api/project-status`：项目与实时模型配置状态。

`POST /api/compare` 示例：

```json
{"domain":"hypertension","question_id":"HTN-01","retrieval_condition":"noisy","live":false}
```

## 当前边界

- 内置知识库是用于三天课程的最小证据集，不是系统综述。
- 离线样例只证明评测机制可运行，不能证明 RAG 优于某个真实模型。
- 自动字符串指标无法判断复杂医学事实是否正确，正式实验必须增加盲法人工评分。
- 实时模式尚未固定随机种子；应进行多次重复并报告均值和置信区间。
