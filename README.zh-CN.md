# CoRe

**Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction**，EMNLP 2026。

Wei Cai、Zihan Mei、Haojun Huang、Bang Wang

[English](README.md) · [数据格式](docs/data.md)

![CoRe](assets/overview.png)

CoRe 从案情中抽取行为、后果和事实，分别检索相似先例，再通过 RPR 合并排名。
Judge A 给出初步判断，Judge B 根据 A 预测的罪名和法条集合重新选择先例，修正判断。
匹配先例不足 K 条时，从第一轮先例中补齐。

MultiLJP 为每名被告建立一条记录，输入完整案情和被告信息，保留该被告的全部罪名、
法条及最终刑期。两位法官均输出刑期月数，评估时转换为 11 个区间。

默认使用 DeepSeek-V3，温度为 0.1；向量模型为 `text-embedding-ada-002`，维度为 1536。
检索使用平方 L2 距离，M=5000、K=19、alpha=60。
CEE 使用 DeepSeek-V4-Pro 评估抽取结果，使用 DeepSeek-V4-Flash 修复。
整体评分不低于 4 且无严重幻觉时保留抽取结果。
参数见 [`configs/paper.json`](configs/paper.json)。

## 安装

需要 Python 3.10+ 和 CPU FAISS。

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

CEE 评估和修复默认使用同一组 LLM 密钥与地址，也可通过
`CORE_CEE_EVALUATION_*` 和 `CORE_CEE_REPAIR_*` 分别设置。
变量名见 [`.env.example`](.env.example)，在终端中设置后生效，文件不会自动加载。
模型环境变量优先于 JSON 配置。

## 运行

下面以 MultiLJP 为例。将过滤后的训练集和测试集放入 `data/`，
同一案件的所有被告应放在同一个划分中。

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

CAIL 使用 `prepare --format cail`。标签为数字索引时，还需指定
`--charges resources/cail_charges.txt --laws resources/cail_laws.txt`。
已抽取的数据缺少原始案情时，用 `--source-cases` 补充。
输入格式和评估指标见[数据说明](docs/data.md)。
API 阶段可加 `--limit` 先跑少量样本；已有输出不会被覆盖。

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 示例

被告拾得一部 VIVOX5 手机后，通过支付宝免密支付窃取 3,723 元。
案发后，家属全额赔偿并取得被害人谅解。

[案情与要素](examples/query.elements.jsonl) ·
[先例](examples/train.jsonl) ·
[输出](examples/predictions.jsonl)

| 法官 | 罪名 | 法条 | 刑期（月） |
| --- | --- | --- | --- |
| A | 盗窃 | 264 | 12 |
| B | 抢夺 | 267 | 6 |

## 结果

| 数据集 | 罪名 Acc. / F1 | 法条 Acc. / F1 | 刑期 Acc. / F1 |
| --- | --- | --- | --- |
| CAIL-Small | 88.51 / 87.29 | 88.22 / 82.37 | 43.20 / 38.20 |
| MultiLJP | 90.70 / 85.45 | 94.09 / 77.86 | 38.23 / 36.30 |

## 引用

```bibtex
@inproceedings{cai2026core,
  title = {Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction},
  author = {Cai, Wei and Mei, Zihan and Huang, Haojun and Wang, Bang},
  year = {2026}
}
```
