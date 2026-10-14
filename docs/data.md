# Data schema and preparation

All pipeline files use UTF-8 JSONL, one object per line. IDs are converted to
strings and must be unique within a file. Preserve the verified original
training/test split; all defendants of a case stay on the same side.

## Canonical records

```json
{
  "id": "case-1:defendant-A",
  "origin_id": "case-1",
  "dataset": "multiljp",
  "defendant": "defendant-A",
  "fact": "The complete original case description.",
  "elements": {
    "behavior": "Description of the target defendant's actions.",
    "consequence": "Direct harm or losses supported by the description.",
    "fact": "Relevant conviction and sentencing facts."
  },
  "labels": {
    "charge": ["盗窃", "诈骗"],
    "law_article": [264, 266],
    "term_months": 12,
    "term_class": 7
  }
}
```

The example is synthetic. `elements` is optional before extraction. `labels`
is required for training precedents and evaluation truth, and optional for
queries. Charge/article arrays represent all true labels, not alternative
candidates. Adapters also accept scalar labels and normalize them to sorted,
deduplicated arrays. A trailing `罪` is removed from charge names.

Each MultiLJP defendant retains **one original final penalty**. Multiple
defendants produce separate records, each with the complete original description
and its target defendant. Multiple unassigned penalties are rejected; the code
does not select the first, sum them, or guess a defendant/charge correspondence.
A singleton penalty list can be normalized without losing information.

Query truth never enters extraction, audit, repair, or judge prompts. Remove any
held-out judgment embedded in input free text upstream; adapters cannot determine
which free-text sentences belong to the judgment.

| Stage | Required inputs or additions |
| --- | --- |
| `prepare` | Stable IDs, genuine original `fact`, optional elements/truth; recomputes interval from months |
| `extract` | Original description, optional target defendant; adds three text facets and `cee_validated=false` |
| `audit` | Original description and three facets; adds evaluation, repair action, raw responses and model IDs |
| `embed` | Three nonempty facets and `cee_validated=true`; adds three vectors and embedding model ID |
| `index` | Training records with truth, three valid vectors and matching embedding model |
| `predict` | Query fact/vectors/model; MultiLJP also retains target defendant; adds both judges' results |
| `evaluate` | Truth labels and predictions joined by unique ID |

`cee_validated=true` means the keep/repair action produced a valid three-facet
schema. It does not claim a second semantic evaluation of repaired text. Neither
schema validation nor the LLM audit guarantees factual correctness.

## Dataset adapters

```bash
# Raw CAIL with id, fact, and official meta.
python -m core prepare --input data/cail.raw.jsonl --format cail --output data/cail.jsonl

# Preprocessed CAIL: numeric accu/law are zero-based table indices.
python -m core prepare --input data/cail.filtered.jsonl --format cail --charges resources/cail_charges.txt --laws resources/cail_laws.txt --output data/cail.jsonl

# Extracted-only CAIL: recover the real original description by id.
python -m core prepare --input data/cail.elements.original.jsonl --format cail-elements --source-cases data/cail.raw.jsonl --output data/cail.elements.jsonl

# Raw MultiLJP: one record per defendant with full charge/article sets.
python -m core prepare --input data/multi.raw.jsonl --format multiljp --output data/multi.jsonl

# Extracted-only MultiLJP: recover the original description by origin_id.
python -m core prepare --input data/multi.elements.original.jsonl --format multiljp-elements --source-cases data/multi.raw.jsonl --output data/multi.elements.jsonl

# Actual training label space, without inventing a historical filter.
python -m core label-space --input data/train.jsonl --charges-out data/charges.txt --laws-out data/laws.txt
```

Assign stable IDs upstream if a source has none. `--source-cases` expects raw
objects with `id` and original `fact`. Missing/duplicate matches fail rather than
substituting an extracted summary. The join can also be used with canonical
records. Original descriptions are required by `prepare` for all formats.

CAIL official `meta.accusation` and `meta.relevant_articles` retain full lists;
articles in `meta` are actual numbers. Locally preprocessed CAIL uses `fact_cut`,
`accu`, `law`, `term`, optional `term_cate` (0 death, 1 life, 2 fixed). Its indexed
label fields must be singleton values and use the original ordered tables.
`cail-elements` reads `behavior`, `consequence`, and `fact_summary`, but never
promotes `fact_summary` to original fact. It adds truth only when genuine label
fields are supplied; `pred_accu` is never ground truth. Extracted test files
without law/penalty truth require a separate labeled file with matching IDs.

`multiljp-elements` accepts grouped `id`, `elements[被告]`,
`crime_info[[被告]]`, or already split `case_id`, `origin_id`, `defendant`,
`data.elements`, `data.crime_info`. Grouped IDs become `original_id:defendant`;
already split IDs remain `case_id`. Each defendant's `accusations`, `laws`, and
`term` are normalized without dropping charge/article labels.

Adapters preserve records and splits. They do not reconstruct the paper's
low-frequency or appeal-case filtering, and do not reduce multi-label cases to
single labels. Use the verified original filtered inputs for score comparisons.
Malformed annotations require explicit upstream correction, not guessing.

## Penalty units

**Both judges output months**, then the same conversion yields an interval for
evaluation. Life imprisonment is represented as 360 months; death as -1.

| Class | Interpretation |
| --- | --- |
| 0 | Death (`term_months=-1`) |
| 1 | More than 120 months, including life |
| 2 | (84, 120] months |
| 3 | (60, 84] months |
| 4 | (36, 60] months |
| 5 | (24, 36] months |
| 6 | (12, 24] months |
| 7 | (9, 12] months |
| 8 | (6, 9] months |
| 9 | (0, 6] months |
| 10 | Zero months |

MultiLJP accepts numeric months and simple Chinese terms such as
`有期徒刑一年十个月`, `拘役三个月`, `死刑`, and `无期徒刑`.
A suspension period is excluded: `有期徒刑三年，缓刑三年六个月` means 36 custodial
months. Ambiguous combined terms are rejected; provide explicit final months.

## Outputs, failures, and metrics

Predictions contain `id`, `status`, both judges' canonical labels/penalty,
first/second-round precedent IDs, candidate count, raw responses, and effective
run configuration. Successful canonical judge fields are `charge` and
`law_article` arrays, scalar `term_months`, and scalar `term_class`.
Model responses use `罪名`, `法条`, `刑期`; original `pred_accusation`,
`pred_relevant_articles`, `pred_imprisonment_months` aliases are accepted.

`audit` performs semantic LLM evaluation/targeted repair separately from JSON
reconstruction. Missing facets, invalid repair schemas, and API failures become
explicit error records. Failed extraction/audit/embedding inputs are preserved.
Per-record processing failures return nonzero CLI status. Invalid JSON files,
duplicate IDs, invalid configuration, and index errors stop the stage.
Reprocess failures into new outputs and join only validated results by ID.

Evaluation rejects duplicate or unknown IDs and malformed successful records.
**All labeled records are scored**: failed/missing charge/article predictions
are empty sets; failed/missing penalty predictions are an incorrect sentinel.
Coverage is reported separately. Charge/article Accuracy is exact set match;
macro precision/recall/F1 are per-label averages over observed truth/prediction
labels, including unknown extra predictions. Penalty Accuracy is interval match,
with macro metrics over all 11 classes. Zero denominators give zero; scores are
percentages, coverage a fraction. With no truth records task metrics are null.
Training label tables do not silently discard unknown predictions.
