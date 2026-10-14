# CoRe 中文说明

[English README](README.md) 为主文档。

论文：**Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction**。

## 与论文对应的实现

| 模块 | 代码 | 功能 |
| --- | --- | --- |
| CEE | `core/cee.py` | 抽取 behavior、consequence、fact，评价质量，自动针对性修复，校验格式 |
| MPR | `core/mpr.py` | 三路独立 `IndexFlatL2` 检索，平方 L2 距离升序 |
| RPR | `core/rpr.py` | 三路结果交集，按 `1/(rank+alpha)` 累加并排序 |
| CRJ | `core/crj.py` | Judge A 初步预测，Judge B 重新选择先例并核验、修正 |

MultiLJP 保留每名被告**全部罪名和法条标签**。每个被告使用完整原始案件描述，加上目标被告标识，不用压缩摘要替换原文。
按 PDF 第 13 页，保留每名被告原有的一个最终刑期；不同被告的刑期分别保存在不同记录中。
罪名、法条为数组，两位法官均输出刑期月数，评估时转换成 11 个区间。
没有被告归属的多个刑期会报错，不自动取第一项、求和或删除。

Judge B 要求先例的罪名集合和法条集合同时与 Judge A 预测一致，集合顺序不影响匹配。
不足 K 条时，从第一轮先例中按排名补齐且不重复；少于 K 条候选时使用实际数量。
同一案件内两位法官顺序调用，不同案件可以并发。查询真实标签不进入推理或质量评估提示词。

默认参数：`M=5000`、`K=19`、`alpha=60`、抽取/两位法官温度 `0.1`；`text-embedding-ada-002`、1536 维。
论文配置采用 `rpr_scope=intersection`，对应“三个列表中均出现”的字面定义。
原始脚本中的并集仍可显式配置为 `union`，不作为论文默认值。

CEE 质量评估使用 `deepseek-v4-pro`、温度 0、最多 4096 个输出 token；整体分数至少 4 且无检测到的严重幻觉时保留。
否则自动路由到严格证据重抽取、反馈重抽取、仅后果修复或自动回退修复。
修复模型 `deepseek-v4-flash` 和温度 0 来自提供的修复代码。
这些步骤只根据原始事实评价，不读取查询的真实判决标签。

## 安装与环境

Python 3.10+；依赖与 `requirements.txt`、`pyproject.toml` 一致：
`numpy>=1.26,<3`、`faiss-cpu>=1.8,<2`、`openai>=1.30,<3`、`json-repair>=0.30,<1`、`scikit-learn>=1.4,<2`。
支持 CPU FAISS，不需要 CUDA 或本地模型权重。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m core --help
$env:CORE_LLM_API_KEY = "YOUR_LLM_KEY"
$env:CORE_LLM_BASE_URL = "https://api.deepseek.com"
$env:CORE_LLM_MODEL = "deepseek-v3"
$env:CORE_EMBEDDING_API_KEY = "YOUR_EMBEDDING_KEY"
$env:CORE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
$env:CORE_EMBEDDING_MODEL = "text-embedding-ada-002"
```

`.env.example` 不会自动加载。配置文件覆盖程序默认值，模型环境变量再覆盖对应配置。
质量评估与修复默认复用 LLM 密钥和地址，可通过 `CORE_CEE_EVALUATION_*`、`CORE_CEE_REPAIR_*` 独立设置。
实际服务必须支持指定模型；精确复现需确认历史模型版本，客户端不能自动验证服务别名对应的快照。

## 运行步骤

先使用原实验过滤后的训练/测试划分；原始数据、向量、索引和日志不随仓库分发。

```powershell
python -m core prepare --input data/multi-train.raw.jsonl --format multiljp --output data/train.jsonl
python -m core prepare --input data/multi-test.raw.jsonl --format multiljp --output data/test.jsonl
python -m core label-space --input data/train.jsonl --charges-out data/charges.txt --laws-out data/laws.txt
python -m core extract --input data/train.jsonl --output data/train.elements.jsonl --config configs/paper.json
python -m core extract --input data/test.jsonl --output data/test.elements.jsonl --config configs/paper.json
python -m core audit --input data/train.elements.jsonl --output data/train.audited.jsonl --config configs/paper.json
python -m core audit --input data/test.elements.jsonl --output data/test.audited.jsonl --config configs/paper.json
python -m core embed --input data/train.audited.jsonl --output data/train.embeddings.jsonl --config configs/paper.json
python -m core embed --input data/test.audited.jsonl --output data/test.embeddings.jsonl --config configs/paper.json
python -m core index --input data/train.embeddings.jsonl --index-dir outputs/train-index --config configs/paper.json
python -m core predict --input data/test.embeddings.jsonl --index-dir outputs/train-index --charges data/charges.txt --output outputs/predictions.jsonl --config configs/paper.json
python -m core evaluate --truth data/test.jsonl --predictions outputs/predictions.jsonl --output outputs/metrics.json
```

CAIL 使用对应的 `prepare --format cail`；数字标签需要原来的有序罪名/法条映射表。
CAIL 或 MultiLJP 已抽取文件如缺少原始案情，必须用 `--source-cases` 按案件 ID 补回原文。
`embed` 要求先完成 `audit`。API 阶段可以加 `--limit 5`；所有输出和索引目录拒绝覆盖。
各被告必须保持原案件的训练/测试归属，索引只能来自训练集。

罪名/法条 Accuracy 为完整标签集合匹配，Macro Precision/Recall/F1 按标签计算，标签空间为真实与预测标签的并集。
刑期 Accuracy 按区间计算，宏平均包含 11 个区间。
失败和缺失预测仍计入评估分母，按错误预测处理；覆盖率单独报告。
指标为百分比，覆盖率是 0–1 的比例。

## 结果与验证

| 数据集 | 罪名 Acc. / F1 | 法条 Acc. / F1 | 刑期 Acc. / F1 |
| --- | --- | --- | --- |
| CAIL-Small | 88.51 / 87.29 | 88.22 / 82.37 | 43.20 / 38.20 |
| MultiLJP | 90.70 / 85.45 | 94.09 / 77.86 | 38.23 / 36.30 |

以上为论文原文报告值，不是此次代码修改重跑得到的结果。
提供的旧 MultiLJP 罪名/法条表为 36/26，不作为论文报告的 23/22 标签空间默认输入。
`label-space` 从实际训练集导出标签表，不自行猜测历史过滤规则。
经过验证的完整过滤数据与标签清单未随材料提供，比较论文数值前仍需原实验处理后的数据与模型版本。

```powershell
python -m compileall -q core tests
python -m unittest discover -s tests -v
python -m pip check
```

测试使用真实 CPU FAISS 与模拟 API，无付费请求。
详细对应关系见 [docs/paper_alignment.md](docs/paper_alignment.md)，数据格式见 [docs/data.md](docs/data.md)。
