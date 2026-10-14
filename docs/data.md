# Data format

Data files use UTF-8 JSONL: one record per line, with a unique string `id`.

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

`fact` holds the original case description. After extraction, `elements` holds
three strings: `behavior`, `consequence`, and `fact` (the summary). Training and
evaluation records need `labels`; unlabeled queries do not.

Charges and articles are stored as arrays, including when there is only one
label. Charge names omit the trailing `罪`. Each defendant has one final
`term_months` value. `prepare` converts it to `term_class`. Multiple penalties
without a defendant assignment are rejected.

MultiLJP records also carry `dataset="multiljp"`, `origin_id`, and `defendant`.
Each defendant gets a separate ID and keeps the full case description, complete
label sets, and final penalty. Defendants from the same case stay in the same split.

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
2 fixed-term imprisonment. Training and evaluation use annotated labels.

If extracted inputs lack original descriptions, join raw cases by `id` (CAIL)
or `origin_id` (MultiLJP):

```bash
python -m core prepare --input data/elements.jsonl --format multiljp-elements --source-cases data/raw.jsonl --output data/prepared.jsonl
```

The source file must contain unique `id` values and the original `fact` text.
Every extracted record must have a matching source case. Use `label-space` to
export charge and article lists from the prepared training split.

## Extraction and embedding

`extract` produces the three elements. `audit` checks them against the original
case and repairs those that need correction. It sets `cee_validated=true` when
the retained or repaired output has three nonempty text fields. Repairs do not
receive a second evaluation.

`embed` reads audited records and adds a vector for each element, along with
`embedding_model`. Training and query vectors must use the same model and dimension.

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
Only the custodial term is counted, excluding any suspension period. For
ambiguous combined terms, supply the final penalty in months.

## Predictions and evaluation

[Prediction records](../examples/predictions.jsonl) contain the labels and
penalties from both judges. A run also saves precedent IDs, raw responses, and
configuration. Query labels are kept out of the prompts.

An API-stage failure produces a record with `status="error"` and an error
message. The command returns a nonzero exit code. Fix and rerun failed records
before indexing. Duplicate IDs and existing output paths cause an error.

Evaluation matches predictions to annotated records by ID. Charge and article
Accuracy require an exact set match; macro precision, recall, and F1 average
over the labels present in annotations or predictions. Penalty Accuracy compares
intervals, and its macro metrics use all 11 classes. Failed or missing predictions
count as incorrect. Scores are percentages; coverage is a fraction. Prediction
IDs must appear in the annotations, and successful records must have valid fields.
