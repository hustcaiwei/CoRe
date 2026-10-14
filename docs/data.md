# Data format

Use UTF-8 JSONL, with one object per line and a unique string `id`.

## Records

```json
{
  "id": "precedent-theft-withdrawal",
  "fact": "The defendant stole the victim's property and made unauthorized withdrawals or transactions.",
  "labels": {
    "charge": [
      "盗窃"
    ],
    "law_article": [
      264
    ],
    "term_months": 8,
    "term_class": 8
  }
}
```

Top-level `fact` is the original case description. `elements` contains behavior,
consequence, and the extracted fact summary. `labels` is required for training
and evaluation, and omitted from unlabeled queries.

Charge and law-article labels are arrays; scalar input labels are normalized to
arrays. Charge names omit the trailing `罪`. `term_months` is one final penalty
per defendant. Multiple unassigned penalties are rejected. `prepare` computes
`term_class` from months.

For MultiLJP, retain `dataset="multiljp"`, `origin_id`, and `defendant`. Each
defendant has a separate ID, the complete original case description, all its
charge/article labels, and its own final penalty. Keep all defendants within
the original case's training/test split.

## Input adapters

| `prepare --format` | Input fields |
| --- | --- |
| `canonical` | `id`, `fact`; optional `elements`, `labels`, `dataset`, `origin_id`, `defendant` |
| `cail` | `id`, `fact`, `meta`; or preprocessed `fact_cut`, `accu`, `law`, `term`, optional `term_cate` |
| `cail-elements` | `id`, original `fact`/`fact_cut`, `behavior`, `consequence`, `fact_summary`; optional true labels |
| `multiljp` | `id`, `fact`, `criminals_info` with `accusations`, `laws`, `term` per defendant |
| `multiljp-elements` | Grouped `id`, `elements`, `crime_info`; or split `case_id`, `origin_id`, `defendant`, `data` |

Official CAIL law articles are article numbers. Preprocessed CAIL `accu` and
`law` indices use the ordered tables in `resources/cail_charges.txt` (119 entries)
and `resources/cail_laws.txt` (103 entries). `term_cate`: 0 death, 1 life,
2 fixed-term imprisonment. Predicted labels are not training/evaluation truth.

If extracted inputs lack original descriptions, join raw cases by `id` (CAIL)
or `origin_id` (MultiLJP):

```bash
python -m core prepare --input data/elements.jsonl --format multiljp-elements --source-cases data/raw.jsonl --output data/prepared.jsonl
```

`--source-cases` requires unique `id` and original `fact` fields. Missing original
cases fail preparation. Use the filtered dataset splits and export their label
catalogs with `label-space`; the adapters perform schema conversion.

## Extraction and embedding

`extract` creates three nonempty text facets. `audit` evaluates them against the
original description and repairs flagged elements. `cee_validated=true` indicates
a valid schema after retention or repair; repaired text is not evaluated again.
`embed` requires this audit flag and adds three vectors plus `embedding_model`.
Training and query vectors must share their model and dimension.

## Penalty intervals

Both judges output months. Death is -1; life imprisonment is 360 months.

| Class | Months |
| --- | --- |
| 0 | Death (-1) |
| 1 | >120, including life |
| 2 | (84, 120] |
| 3 | (60, 84] |
| 4 | (36, 60] |
| 5 | (24, 36] |
| 6 | (12, 24] |
| 7 | (9, 12] |
| 8 | (6, 9] |
| 9 | (0, 6] |
| 10 | 0 |

Chinese terms such as `有期徒刑一年十个月` and `拘役三个月` are accepted.
Suspension periods are excluded; ambiguous combined terms require explicit months.

## Predictions and evaluation

[Prediction example](../examples/predictions.jsonl) shows both judges' labels and
penalties. Inference also records precedent IDs, raw responses, and run configuration.
Query truth never enters model prompts.

API-stage failures are written as `status="error"` with an error message, and
return nonzero CLI status. Failed records require correction/reprocessing before
indexing. Input files and outputs require unique IDs; existing outputs are not overwritten.

Evaluation joins truth and predictions by ID. Charge/article Accuracy is exact
set match; macro precision/recall/F1 average over observed truth and predicted
labels. Penalty Accuracy matches intervals, with macro metrics over all 11 classes.
Failed/missing predictions count as incorrect. Scores are percentages; coverage
is a fraction. Unknown IDs and malformed successful outputs are rejected.
