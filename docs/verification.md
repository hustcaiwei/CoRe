# Local verification

Verification date: **2026-10-14** (Asia/Shanghai).

This revision changes executable code, configuration, examples, and bilingual
documentation to match the supplied PDF's multi-label/defendant contract,
retrieval scope, penalty units, and CEE evaluation/repair workflow.

## Checks

- `python -m unittest discover -s tests -v`: **30 offline tests passed**.
- `python -m compileall -q core tests` completed successfully.
- `python -m pip check` reports no broken requirements.
- Both `python -m core --help` and `core-ljp --help` expose the unified pipeline.
- GitHub Actions also passed on Ubuntu with Python 3.12 for code commit `cb18e09`: [successful workflow](https://github.com/hustcaiwei/CoRe/actions/runs/38525002790).
- `configs/paper.json` matches all defaults in `core/config.py`.
- Dependency declarations in `pyproject.toml` and `requirements.txt` agree.
- Relative Markdown links and reference-map targets resolve.
- Delivery excludes virtual environments, Git metadata, benchmark cases,
  embeddings, indices, logs, and secret environment files.

The local Windows Python 3.12 environment is recorded in
[`requirements-verified.txt`](../requirements-verified.txt); it is not the
historical experiment environment. The main-result rows were transcribed and
previously checked against the supplied manuscript, not produced by these tests.

## Test evidence

| Area | Verified behavior |
| --- | --- |
| Multi-label data | Preserve every charge/article; normalized label sets; one final penalty per defendant |
| Original descriptions | Full MultiLJP case plus target defendant; CAIL summary is not promoted to original fact; source-case joins for both datasets |
| CEE | Keep threshold, severe-hallucination rejection, targeted consequence repair preserving correct facets, audit required before CLI embedding |
| RPR | One-based reciprocal scores, smoothing, intersection/explicit union, invalid FAISS IDs |
| Judge B | Complete label-set matching independent of order; ranked first-round backfill without duplicates |
| Penalty | Month/interval boundaries, death/life, Chinese terms, suspension exclusion, ambiguous/multiple terms rejected |
| Evaluation | Exact sets; extra predicted labels reduce scores; all failed/missing cases remain in denominators |
| MPR | Real CPU FAISS; small indices; incompatible model/dimension rejection |
| Full CLI | Prepare both splits, derive labels, extract, audit, embed, index, predict both judges, and evaluate synthetic multi-label cases |
| Error handling | Duplicate IDs, existing output refusal, explicit failed records and nonzero CLI exit status |
| Leakage | Query truth excluded from inference prompts |

Tests use fake deterministic API responses and real FAISS. They make no real
embedding, extraction, repair, or judge calls. Expected refusal/error messages
in tests are assertions of failure behavior.

Independent read-only review identified three concrete bugs: suspension-duration
mixing, CAIL audit against its own summary, and an extra CEE retention condition.
All three were fixed, covered by regressions, and independently rechecked. The
review also exercised original-description joins for both CAIL and MultiLJP.

## Remaining limits

Full benchmark scores, semantic quality of live LLM outputs, historical model
availability, and original filtered-label manifests were not established.
No paid API calls were made. Hosted Ubuntu verification covers installation,
compilation, and all 30 offline tests; it does not establish live API behavior. See
[paper_alignment.md](paper_alignment.md) for required reproduction inputs.

The prepared repository was pushed to `hustcaiwei/CoRe` on `main` using the
owner's authenticated Git session. The original remote history was retained.
Local and remote code commits matched, with all 55 curated files published.
