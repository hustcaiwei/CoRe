# CoRe

**Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction**

Wei Cai, Zihan Mei, Haojun Huang, and Bang Wang · EMNLP 2026

[中文说明](README.zh-CN.md) · [Data format](docs/data.md)

CoRe predicts charges, applicable law articles, and terms of penalty through
case element extraction, multi-facet precedent retrieval, reciprocal reranking,
and collaborative reasoning by two LLM judges.

![CoRe framework](assets/overview.png)

## Method

| Module | Description |
| --- | --- |
| CEE | Extract behavior, consequence, and fact; evaluate and repair extracted elements |
| MPR | Retrieve precedents through three independent `IndexFlatL2` indices |
| RPR | Rerank candidates shared by all three lists using reciprocal rank scores |
| CRJ | Judge A predicts; Judge B reselects precedents and refines the judgment |

MultiLJP uses one instance per defendant, with the complete case description and
target defendant. Charge and law-article labels are sets; each defendant has one
final penalty. Both judges output penalty months, which are mapped to 11 intervals
for evaluation. Judge B selects precedents with matching charge and article sets;
if fewer than K match, unused first-round precedents fill the remaining positions.

Default settings in [`configs/paper.json`](configs/paper.json):

| Setting | Value |
| --- | --- |
| Extraction / judgment model | `deepseek-v3` |
| Extraction / Judge A / Judge B temperature | 0.1 / 0.1 / 0.1 |
| Embedding model / dimensions | `text-embedding-ada-002` / 1536 |
| Retrieval M / selection K / smoothing alpha | 5000 / 19 / 60 |
| Candidate scope / distance | Intersection / squared L2 |
| CEE evaluator / temperature / maximum output tokens | `deepseek-v4-pro` / 0 / 4096 |
| CEE retention | Overall score >=4 and no severe hallucination |
| CEE repair model / temperature | `deepseek-v4-flash` / 0 |

## Installation

Python 3.10+ is required. Retrieval uses CPU FAISS.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m core --help
```

`core-ljp` is also available as a command-line entry point.

## API configuration

Set the LLM and embedding credentials in your shell:

```powershell
$env:CORE_LLM_API_KEY = "YOUR_LLM_KEY"
$env:CORE_LLM_BASE_URL = "https://api.deepseek.com"
$env:CORE_LLM_MODEL = "deepseek-v3"
$env:CORE_EMBEDDING_API_KEY = "YOUR_EMBEDDING_KEY"
$env:CORE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
$env:CORE_EMBEDDING_MODEL = "text-embedding-ada-002"
```

[`.env.example`](.env.example) lists the available variables; it is not loaded
automatically. CEE evaluation and repair reuse the LLM key and endpoint unless
`CORE_CEE_EVALUATION_*` or `CORE_CEE_REPAIR_*` is set. Model environment variables
override configuration values. Use endpoints that support the configured models.

## Run

Place the filtered training and test splits in `data/`. Keep all defendants of
a case in the same split and build indices from training precedents only.

```bash
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

For CAIL, use `prepare --format cail`. Indexed labels require
`--charges resources/cail_charges.txt --laws resources/cail_laws.txt`.
For extracted files without original descriptions, supply `--source-cases`.
See [data formats and adapters](docs/data.md). API stages support `--limit`;
output files and index directories must use new paths.

Charge/article Accuracy measures exact label-set match; macro precision, recall,
and F1 are averaged per label. Penalties are evaluated over 11 interval classes.
Failed or missing predictions count as incorrect, with coverage reported separately.

## Example

The case concerns a found VIVOX5 phone and an unauthorized Alipay withdrawal of
3,723 yuan. The defendant's family compensates the victim in full and obtains
forgiveness.

- [Case and extracted elements](examples/query.elements.jsonl)
- [Training precedents](examples/train.jsonl)
- [Prediction output example](examples/predictions.jsonl)

| Judge | Charge | Article | Penalty (months) |
| --- | --- | --- | --- |
| A | Theft | 264 | 12 |
| B | Snatch | 267 | 6 |

The JSON output stores charge/article labels as arrays and each defendant's
penalty as a scalar. Run `audit` before embedding the example elements.

## Results

| Dataset | Charge Acc. / F1 | Article Acc. / F1 | Penalty Acc. / F1 |
| --- | --- | --- | --- |
| CAIL-Small | 88.51 / 87.29 | 88.22 / 82.37 | 43.20 / 38.20 |
| MultiLJP | 90.70 / 85.45 | 94.09 / 77.86 | 38.23 / 36.30 |

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests exercise multi-label parsing, CEE repair, FAISS retrieval, reranking,
Judge B selection, penalty conversion, and evaluation with offline API fixtures.
GitHub Actions runs the same checks.

## Citation

```bibtex
@inproceedings{cai2026core,
  title = {Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction},
  author = {Cai, Wei and Mei, Zihan and Huang, Haojun and Wang, Bang},
  year = {2026}
}
```
