# Source provenance

The repository was organized from the supplied CoRe code directory. Selection
prioritizes the collected scripts under `LJP整理/code` and the complementary
MultiLJP Judge A implementation. The PDF-alignment revision also examined later
multi-label and extraction-repair sources listed below. Relative source paths are retained in
[`reference/source_map.json`](../reference/source_map.json).

| Original source, relative to the CoRe code directory | Organized destination |
| --- | --- |
| `LJP整理/code/element_extract/CAIL_data/{train,test}_element_async.py` | CEE role and `reference/cee/` |
| `LJP整理/code/element_extract/MultiLJP_data/element_extraction.py` | Defendant-level CEE role and `reference/cee/` |
| `LJP整理/code/Index/Multi_index/{index,vectorization}.py` | MPR role and `reference/mpr/` |
| `LJP整理/code/RAG/RAG_plan_B_judge_{A,B}.py` | RPR/CRJ roles and `reference/crj/` |
| `MultiLJP/predict/Judge_A.py` | MultiLJP adaptation and `reference/crj/` |
| `LJP整理/code/log/log解析.py` | Metrics terminology and `reference/evaluation/` |
| `LJP整理/plot_code/` top-K, temperature, smoothing scripts | `reference/plots/` |
| `LJP整理/code/log/new_accu.txt` and `new_law.txt` | `resources/cail_charges.txt` and `cail_laws.txt` |
| `MultiLJP/procress_data/acc.txt` and `law.txt` | `resources/multiljp_charges.txt` and `multiljp_laws.txt` |
| Manuscript `png/overview.png` | `assets/overview.png` |
| Manuscript main-result CoRe rows | `results/paper_reported.json` |

The unified `core/` package is a consolidated implementation of these roles,
with the explicit changes in [paper_alignment.md](paper_alignment.md). It is not
an unchanged copy of any one historical script.

The reference scripts have `.py.txt` extensions because they rely on historical
schemas, intermediate files, and execution conventions. They are readable
references, not supported entry points. Credentials and absolute local path
literals were removed; remaining experimental parameters and prompts illustrate
the original variants. They are not byte-identical archives.

Excluded materials include raw and processed benchmark cases, embeddings,
FAISS indices, logs, model weights, nested Git metadata, IDE settings, paper
build directories, and unrelated baselines/later extensions. Later adaptive retrieval variants are not part of the supported pipeline.
The following later sources were inspected for the multi-label and CEE repair
contracts (their whole files are not bundled as executable entry points):

| Source | Role carried into `core/` |
| --- | --- |
| `CoReV0.1/5_18/retrieval_resume.py` | Array charge/article outputs with one penalty per defendant |
| `CoReV0.1/5_18/evaluate.py` | Exact-set Accuracy and multi-label per-class metrics |
| `CoReV0.1/5_14/evaluate.py` | DeepSeek-V4-Pro extraction evaluation |
| `CoReV0.1/5_14/repair_async.py` | Targeted repair actions and DeepSeek-V4-Flash repair default |

The supplied PDF is the authority where these sources differ. Query ground-truth
metadata from historical audit scripts is deliberately excluded from model
prompts. No historical pipeline is copied unchanged.

The original code and paper files were left intact. This provenance describes the local files; it does not assert that the revised
package is synchronized with an online repository.
