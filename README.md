# CoRe

**Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction**

Wei Cai, Zihan Mei, Haojun Huang, and Bang Wang · EMNLP 2026

[中文说明](README.zh-CN.md) · [Data format](docs/data.md) · [Paper correspondence](docs/paper_alignment.md) · [Verification](docs/verification.md)

CoRe extracts **behavior, consequence, and fact**, retrieves historical cases
from three facets, reranks them, and uses **Judge A followed by Judge B** to
predict charges, relevant law articles, and the term of penalty.

MultiLJP runs one instance per defendant using the **complete original case
description and target defendant**. All annotated charges and law articles are
preserved as label sets. Each defendant retains one original final penalty.
Both judges output penalty months; evaluation converts them into 11 intervals.
Multiple defendants therefore have separate records and separate penalties.

![CoRe framework from the paper](assets/overview.png)

## Modules and paper parameters

| Module | Implementation | Behavior |
| --- | --- | --- |
| CEE | `core/cee.py` | Extract, evaluate, apply targeted repair, and validate schema |
| MPR | `core/mpr.py` | Independent behavior/consequence/fact `IndexFlatL2` search, ascending squared L2 distance |
| RPR | `core/rpr.py` | Candidates present in all three lists; sum `1/(rank+alpha)` using one-based ranks |
| CRJ | `core/crj.py` | Judge A predicts; Judge B reselects precedents and can revise the prediction |

| Parameter | `configs/paper.json` default |
| --- | --- |
| Extraction and judge model identifier | `deepseek-v3` |
| Extraction / Judge A / Judge B temperature | 0.1 / 0.1 / 0.1 |
| Embedding model / dimension | `text-embedding-ada-002` / 1536 |
| Maximum candidates per facet M / precedents per judge K / alpha | 5000 / 19 / 60 |
| RPR candidate scope | `intersection` |
| CEE evaluator / temperature / maximum output tokens | `deepseek-v4-pro` / 0 / 4096 |
| CEE keep criterion | Overall score >=4, without detected severe hallucination |
| CEE repair model / temperature | `deepseek-v4-flash` / 0 (from the supplied repair implementation) |
| Records per batch / simultaneous API requests | 16 / 8 |

The PDF's 'appearing in all three precedent lists' is implemented literally as
an intersection. `rpr_scope=union` remains available as an explicit historical
variant, and is not the paper default. Short retrieval lists are not padded;
FAISS invalid row IDs are ignored. Equal scores use training row order.

Judge B matches the **complete charge set and complete law-article set**,
independently of label order. If fewer than K precedents match, unused
first-round precedents fill the remaining positions, as stated in the paper.
The two judges run sequentially within each case; independent cases can run
concurrently. Query ground-truth labels never enter inference or audit prompts.

## Installation and requirements

Python 3.10+ and CPU FAISS are required. CUDA and local model weights are not
needed for the supported API pipeline.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m core --help
```

`core-ljp` provides the same interface as `python -m core`.
Runtime dependencies are `numpy>=1.26,<3`, `faiss-cpu>=1.8,<2`,
`openai>=1.30,<3`, `json-repair>=0.30,<1`, and `scikit-learn>=1.4,<2`.
They are identical in `requirements.txt` and `pyproject.toml`.
`requirements-verified.txt` records the local verification environment.

## API configuration

Set environment variables in your shell; `.env.example` is a template and is
not automatically loaded.

```powershell
$env:CORE_LLM_API_KEY = "YOUR_LLM_KEY"
$env:CORE_LLM_BASE_URL = "https://api.deepseek.com"
$env:CORE_LLM_MODEL = "deepseek-v3"
$env:CORE_EMBEDDING_API_KEY = "YOUR_EMBEDDING_KEY"
$env:CORE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
$env:CORE_EMBEDDING_MODEL = "text-embedding-ada-002"
```

LLM and embedding credentials are separate. The audit and repair clients reuse
`CORE_LLM_API_KEY` and its endpoint unless `CORE_CEE_EVALUATION_API_KEY` /
`CORE_CEE_EVALUATION_BASE_URL` or `CORE_CEE_REPAIR_API_KEY` /
`CORE_CEE_REPAIR_BASE_URL` are set. Their models can be overridden with
`CORE_CEE_EVALUATION_MODEL` and `CORE_CEE_REPAIR_MODEL`.

`--config` overrides `core/config.py` defaults. Model environment variables take
precedence over the corresponding configuration. Unknown configuration keys
are rejected. SDK requests use a 120-second timeout and `max_retries=2`.
The configured endpoint must expose the requested model identifier. Exact paper
reproduction requires the original model snapshots, including the historical
DeepSeek-V3 release; the client cannot verify provider snapshot identity.

## Prepare data and run

Use the verified original filtered training/test splits. The repository contains
no benchmark cases, vectors, model weights, or experiment logs. See
[dataset adapters and schemas](docs/data.md). Train and test indices must never
be combined, and all defendants stay in their original case's split.

```bash
# Example: retain every MultiLJP defendant's charge/article sets and original description.
python -m core prepare --input data/multi-train.raw.jsonl --format multiljp --output data/train.jsonl
python -m core prepare --input data/multi-test.raw.jsonl --format multiljp --output data/test.jsonl

# Derive the actual charge/law space from the verified training split.
python -m core label-space --input data/train.jsonl --charges-out data/charges.txt --laws-out data/laws.txt

# CEE: extract, then evaluate and automatically repair both splits.
python -m core extract --input data/train.jsonl --output data/train.elements.jsonl --config configs/paper.json
python -m core extract --input data/test.jsonl --output data/test.elements.jsonl --config configs/paper.json
python -m core audit --input data/train.elements.jsonl --output data/train.audited.jsonl --config configs/paper.json
python -m core audit --input data/test.elements.jsonl --output data/test.audited.jsonl --config configs/paper.json

# MPR: encode the validated elements and build training-only indices.
python -m core embed --input data/train.audited.jsonl --output data/train.embeddings.jsonl --config configs/paper.json
python -m core embed --input data/test.audited.jsonl --output data/test.embeddings.jsonl --config configs/paper.json
python -m core index --input data/train.embeddings.jsonl --index-dir outputs/train-index --config configs/paper.json

# RPR + CRJ, followed by evaluation by record ID.
python -m core predict --input data/test.embeddings.jsonl --index-dir outputs/train-index --charges data/charges.txt --output outputs/predictions.jsonl --config configs/paper.json
python -m core evaluate --truth data/test.jsonl --predictions outputs/predictions.jsonl --output outputs/metrics.json
```

For CAIL, use `prepare --format cail` (or the extracted-data adapter) with the
matching original ordered label tables when input labels are numeric indices.
For extracted-only CAIL or MultiLJP files, supply `prepare --source-cases` to join the
original descriptions. `embed` requires output from the audit/repair stage.
`extract`, `audit`, `embed`, and `predict` call real APIs; `--limit 5` restricts a
stage to five records. Outputs and index directories are never overwritten.

Charge/article **Accuracy is exact label-set match**. Macro precision, recall,
and F1 are calculated per label using the union of observed truth/prediction
labels. Penalty Accuracy uses the final interval, with macro metrics over the
11 interval classes. Metrics include all labeled records: failed or missing
predictions count as incorrect rather than disappearing. Coverage is reported
separately as a fraction. Metrics are percentages.

## Reported paper results

The following values are transcribed from the supplied PDF, not obtained by
rerunning the revised repository.

| Dataset | Charge Acc. / F1 | Law article Acc. / F1 | Penalty Acc. / F1 |
| --- | --- | --- | --- |
| CAIL-Small | 88.51 / 87.29 | 88.22 / 82.37 | 43.20 / 38.20 |
| MultiLJP | 90.70 / 85.45 | 94.09 / 77.86 | 38.23 / 36.30 |

Full rows are in `results/paper_reported.json`. A verified historical filtered-data manifest
and complete filtered-label manifest were not supplied. The archived MultiLJP
source tables have 36/26 entries, not the paper's 23/22; they are **not used as the
paper's default label space**. `label-space` exports what your verified training
split actually contains; it does not invent a 23/22 filter. See
[paper correspondence](docs/paper_alignment.md) for the remaining reproduction inputs.

## Repository and checks

`core/` holds the supported pipeline; `configs/` holds defaults; `docs/` describes
schemas and correspondence; `reference/` holds sanitized historical scripts as
nonexecuting `.py.txt` references; `examples/` contains synthetic records;
`resources/` contains source label tables. The framework illustration is in
`assets/`, and GitHub Actions runs offline checks.

```bash
python -m compileall -q core tests
python -m unittest discover -s tests -v
python -m pip check
```

Tests use real CPU FAISS and deterministic fake API responses. They cover full
multi-label preservation, unknown extra predictions, exact-set matching,
original defendant-specific descriptions, CEE quality decisions and repairs,
RPR intersection, Judge B reselection/backfill, penalty units, and failed-record
scoring. Live benchmark results and historical API availability require a
separate real experiment.

## Citation and license

See `CITATION.cff`. No license was supplied with the source materials; the
repository does not assign a redistribution license on the authors' behalf.

```bibtex
@inproceedings{cai2026core,
  title = {Collaborative Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction},
  author = {Cai, Wei and Mei, Zihan and Huang, Haojun and Wang, Bang},
  year = {2026}
}
```
