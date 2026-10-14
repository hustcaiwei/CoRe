# CoRe

Code for **Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction** (EMNLP 2026).

Wei Cai, Zihan Mei, Haojun Huang, and Bang Wang

[中文](README.zh-CN.md) · [Data format](docs/data.md)

![CoRe](assets/overview.png)

CoRe extracts behavior, consequence, and fact from each case, then retrieves
precedents for each of these three facets. RPR combines their rankings over the
shared candidates. Judge A makes an initial prediction; Judge B selects
precedents that match A's charge and article sets, and revises the judgment.
When fewer than K precedents match, the remaining places are filled from the
first round.

For MultiLJP, each record pairs the full case description with one defendant.
It keeps that defendant's complete charge and article sets and final penalty.
Both judges predict penalty months, which are converted to 11 intervals for evaluation.

The default settings are DeepSeek-V3, temperature 0.1, and
`text-embedding-ada-002` with 1536 dimensions. Retrieval uses squared L2 distance,
with M=5000, K=19, and alpha=60. CEE uses DeepSeek-V4-Pro to review extractions
and DeepSeek-V4-Flash for repairs. Extractions are retained when their overall
score is at least 4 and no severe hallucination is detected.
Settings are in [`configs/paper.json`](configs/paper.json).

## Setup

Requires Python 3.10+ and CPU FAISS.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m core --help
```

Set your API keys and endpoints:

```powershell
$env:CORE_LLM_API_KEY = "YOUR_LLM_KEY"
$env:CORE_LLM_BASE_URL = "https://api.deepseek.com"
$env:CORE_LLM_MODEL = "deepseek-v3"
$env:CORE_EMBEDDING_API_KEY = "YOUR_EMBEDDING_KEY"
$env:CORE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
$env:CORE_EMBEDDING_MODEL = "text-embedding-ada-002"
```

CEE review and repair use the same LLM credentials by default. To use separate
endpoints, set `CORE_CEE_EVALUATION_*` and `CORE_CEE_REPAIR_*`.
See [`.env.example`](.env.example) for the variable names. Export them in your
shell; the file is not loaded automatically. Model environment variables take
precedence over the JSON configuration.

## Usage

The commands below use MultiLJP. Put the filtered training and test files in
`data/`, keeping all defendants of a case in the same split.

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

For CAIL, use `prepare --format cail`. If labels are numeric indices, also pass
`--charges resources/cail_charges.txt --laws resources/cail_laws.txt`.
Use `--source-cases` to attach original descriptions to extracted records.
[data.md](docs/data.md) describes the input formats and evaluation metrics.
API stages accept `--limit` for small runs. Outputs are not overwritten.

To run the tests:

```bash
python -m unittest discover -s tests -v
```

## Example

After finding a VIVOX5 phone, the defendant steals 3,723 yuan through Alipay's
password-free payment service. The family later compensates the victim in full
and obtains forgiveness.

[Case and elements](examples/query.elements.jsonl) ·
[Precedents](examples/train.jsonl) ·
[Output](examples/predictions.jsonl)

| Judge | Charge | Article | Months |
| --- | --- | --- | --- |
| A | Theft | 264 | 12 |
| B | Snatch | 267 | 6 |

## Results

| Dataset | Charge Acc. / F1 | Article Acc. / F1 | Penalty Acc. / F1 |
| --- | --- | --- | --- |
| CAIL-Small | 88.51 / 87.29 | 88.22 / 82.37 | 43.20 / 38.20 |
| MultiLJP | 90.70 / 85.45 | 94.09 / 77.86 | 38.23 / 36.30 |

## Citation

```bibtex
@inproceedings{cai2026core,
  title = {Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction},
  author = {Cai, Wei and Mei, Zihan and Huang, Haojun and Wang, Bang},
  year = {2026}
}
```
