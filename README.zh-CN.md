# CoRe

**Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction**

Wei Cai、Zihan Mei、Haojun Huang、Bang Wang · EMNLP 2026

[English README](README.md) · [数据格式](docs/data.md)

CoRe 通过案件要素抽取、多视角先例检索、倒数排名重排序和两位 LLM 法官的
协同推理，预测罪名、适用法条和刑期。

![CoRe 框架](assets/overview.png)

## 方法

| 模块 | 功能 |
| --- | --- |
| CEE | 抽取行为、后果、事实；评估并修复抽取结果 |
| MPR | 使用三路独立的 `IndexFlatL2` 索引检索先例 |
| RPR | 对三个列表共有的候选按倒数排名分数重排序 |
| CRJ | Judge A 初步预测，Judge B 重新选择先例并修正判断 |

MultiLJP 按被告组织推理实例，输入完整案情和目标被告。
罪名、法条保留完整标签集合，每名被告分别保存一个最终刑期。
两位法官均输出刑期月数，评估时转换为 11 个区间。
Judge B 选择罪名和法条集合均匹配的先例；不足 K 条时，使用第一轮未选中的先例补齐。

[`configs/paper.json`](configs/paper.json) 的默认设置：

| 参数 | 值 |
| --- | --- |
| 抽取 / 判断模型 | `deepseek-v3` |
| 抽取 / Judge A / Judge B 温度 | 0.1 / 0.1 / 0.1 |
| 向量模型 / 维度 | `text-embedding-ada-002` / 1536 |
| 检索 M / 选择 K / 平滑 alpha | 5000 / 19 / 60 |
| 候选范围 / 距离 | 交集 / 平方 L2 |
| CEE 评估模型 / 温度 / 最大输出 token | `deepseek-v4-pro` / 0 / 4096 |
| CEE 保留条件 | 整体评分至少 4，且无严重幻觉 |
| CEE 修复模型 / 温度 | `deepseek-v4-flash` / 0 |

## 安装与配置

需要 Python 3.10+，检索使用 CPU FAISS。

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

也可使用 `core-ljp` 入口。[`.env.example`](.env.example) 列出了可配置变量，
不会自动加载。CEE 评估和修复默认复用 LLM 密钥与地址，也可分别设置
`CORE_CEE_EVALUATION_*` 和 `CORE_CEE_REPAIR_*`。
模型环境变量覆盖配置文件；服务端需支持所配置的模型。

## 运行

将过滤后的训练集和测试集放入 `data/`，同一案件的所有被告保持相同划分。
索引只使用训练集先例。

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

CAIL 使用 `prepare --format cail`；索引标签需指定
`--charges resources/cail_charges.txt --laws resources/cail_laws.txt`。
已抽取文件缺少原始案情时使用 `--source-cases`，详见[数据格式](docs/data.md)。
API 阶段可用 `--limit` 限制记录数；输出文件和索引目录使用新路径。

罪名、法条 Accuracy 为完整标签集合匹配，Macro Precision/Recall/F1 按标签平均。
刑期按 11 个区间评估。失败或缺失预测按错误计入评分，覆盖率单独报告。

## 示例

案件涉及拾得 VIVOX5 手机后，通过支付宝免密支付窃取 3,723 元；
被告家属全额赔偿并获得被害人谅解。

- [案情与抽取要素](examples/query.elements.jsonl)
- [训练先例](examples/train.jsonl)
- [预测输出示例](examples/predictions.jsonl)

| 法官 | 罪名 | 法条 | 刑期（月） |
| --- | --- | --- | --- |
| A | 盗窃 | 264 | 12 |
| B | 抢夺 | 267 | 6 |

JSON 中罪名、法条为数组，每名被告的刑期为单个数值。
示例要素需先经过 `audit`，再进行向量化。

## 结果

| 数据集 | 罪名 Acc. / F1 | 法条 Acc. / F1 | 刑期 Acc. / F1 |
| --- | --- | --- | --- |
| CAIL-Small | 88.51 / 87.29 | 88.22 / 82.37 | 43.20 / 38.20 |
| MultiLJP | 90.70 / 85.45 | 94.09 / 77.86 | 38.23 / 36.30 |

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖多标签解析、CEE 修复、FAISS 检索、重排序、Judge B 选择、刑期转换和评估。
GitHub Actions 自动运行相同检查。

## 引用

```bibtex
@inproceedings{cai2026core,
  title = {Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction},
  author = {Cai, Wei and Mei, Zihan and Huang, Haojun and Wang, Bang},
  year = {2026}
}
```
