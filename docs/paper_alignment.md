# Correspondence with the supplied PDF

The authoritative manuscript for this revision is **CoRe__EMNLP_2026_.pdf**,
18 pages, supplied alongside this repository. Its title is *Collaborative
Reasoning with Reranked Multi-facet Retrieval for Legal Judgment Prediction*.
The PDF and original source directory were left unchanged. The PDF takes
precedence over inconsistent historical scripts and earlier repository notes.

## Implemented correspondence

| PDF location | Requirement | Supported implementation |
| --- | --- | --- |
| Method, pp. 3–4 | Three facets, separate precedent retrieval, reciprocal rank sums | `core/cee.py`, `core/mpr.py`, `core/rpr.py` |
| p. 3 | Candidates appearing in all three lists | Default `intersection`; historical `union` is an explicit alternative |
| p. 4 | Judge A then Judge B; reselect by predicted labels; backfill from first round | Sequential calls, complete charge/article set equality, unused ranked first-round backfill |
| pp. 5, 13 | Defendant-specific MultiLJP inference with complete charge/article sets | One record per defendant; full original case plus target name; arrays preserved throughout |
| pp. 5, 13 | One original final penalty per defendant | Scalar months; separate defendants have separate penalties |
| Setup Table 9, p. 12 | M=5000, K=19, alpha=60 | Matching configuration defaults |
| Setup Table 9, p. 12 | ada-002, 1536 dimensions, IndexFlatL2 | Three CPU FAISS indices; squared L2; no added vector normalization |
| Setup Table 9, p. 12 | DeepSeek-V3; extraction/judge temperatures 0.1 | Matching default model identifier and temperatures |
| Appendix F, p. 15 | Audit all processed splits; DeepSeek-V4-Pro, temperature 0, maximum 4096 tokens | Separate mandatory `audit` stage before CLI embedding |
| Appendix F, p. 15 | Keep overall >=4 and no severe hallucination; targeted repairs otherwise | Explicit validated scores/flag and keep/repair routing |
| Figures 8–9, pp. 17–18 | Behavior, consequence, fact extraction and feedback repair | Three nonempty text facets, evidence-based repair instructions, preserve other facets for consequence-only repair |
| Figure 10, p. 18 | Both judges predict penalty months | Same month parser for both; conversion to 11 intervals for evaluation |
| Evaluation and supplied multi-label evaluator | Charge/article set classification | Exact-set Accuracy and per-label macro metrics; penalty interval classification |

`elements.fact` names the extracted summary, while top-level `fact` names the
original case description. They are distinct. In particular, MultiLJP judge
prompts use the original description and target defendant, not a substituted
summary. Extraction audits likewise compare facets to genuine source text.
Query labels are excluded from all model prompts.

## Changes from earlier scripts and the initial organized version

Earlier repository code rejected multiple charges/articles. This revision
changes the adapters, model output schemas, second-round matching, and evaluator
together; no charge/article set is reduced to its first item. Unknown extra model
labels remain visible and lower precision/Accuracy. Multiple unassigned penalties
are rejected rather than guessed. The paper's defendant-level final penalty is
not a list of independent penalty-class predictions.

Selected older RAG scripts pooled the union of retrieved lists. The PDF explicitly
uses membership in all three, so the default is now intersection. An empty
intersection is an explicit prediction failure. The historical union is available
for comparisons and recorded in run configuration; it is not the paper default.

The first organized version treated Judge B's penalty as an interval code.
Figure 10 uses months for both judges, so both now output months. Explicit
death/life normalization fixes older source routines that confused numeric/string
sentinels or mapped special terms to zero. Suspension periods are not combined
with custodial imprisonment.

CEE now includes the appendix's semantic evaluation and automatic targeted
repair, in addition to JSON reconstruction. The evaluator scores behavior
faithfulness/completeness, consequence validity, summary quality,
non-hallucination, and overall quality on 0–5 scales. Overall >=4 and no severe
hallucination retain the extraction. Other cases use strict evidence
re-extraction, consequence-only repair, feedback re-extraction, or an automated
fallback repair. `cee_repair_model=deepseek-v4-flash` comes from the supplied
later repair implementation; the PDF does not separately mandate that repair
model. The low-score route is automatic rather than a human annotation step.

Repaired output is schema-validated, without a second evaluator call.
`cee_validated` records completion of this keep/repair process, not a guaranteed
post-repair semantic score. Failed calls/repairs remain error records and require
reprocessing before embedding. Appendix repair rates and example counts are
reported historical results, not assertions enforced by this code.

Prompts are consolidated Chinese templates with the PDF's roles and structured
output contract. They are not byte-identical copies of the illustrative English
figures. Charge/article arrays make the multi-label evaluation contract explicit.
Historical reference scripts retain their own variants and are not executable
entry points of `core/`.

## Reproduction inputs still required

The supplied MultiLJP source tables contain **36 charges / 26 articles**, while
the PDF reports **23 / 22** after filtering. No verified filtered label manifest
or complete case-selection artifact was supplied. The source tables remain
clearly identified as historical resources; they are not the paper's default
MultiLJP label space. `label-space` derives labels from a user's verified training
split. It does not choose arbitrary labels to force a 23/22 count.

Exact score reproduction still requires the original filtered splits, label
manifest, historical model snapshots and provider behavior. Model identifiers
can be overridden for current endpoints, but the client cannot verify historical
snapshot identity. These changes have not been benchmarked on full CAIL-Small
or MultiLJP and do not establish equality to the paper's scores.

Reported main-result values remain separate in `results/paper_reported.json`.
Offline tests verify data preservation, mechanics, units, repair decisions,
failure accounting, and the complete CLI pipeline with fake APIs and real FAISS.
Ablations, backbone comparisons, predictive quality, and historical extraction
repair statistics were not rerun. See [verification.md](verification.md).
