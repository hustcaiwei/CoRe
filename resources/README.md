# Source label tables

These historical tables were copied from the supplied project and transcoded to
UTF-8 without reordering. CAIL table order matters for preprocessed zero-based IDs.

| Resource | Original source | Count |
| --- | --- | --- |
| `cail_charges.txt` | `new_accu.txt` | 119 |
| `cail_laws.txt` | `new_law.txt` | 103 |
| `multiljp_charges.txt` | `MultiLJP/procress_data/acc.txt` | 36 |
| `multiljp_laws.txt` | `MultiLJP/procress_data/law.txt` | 26 |

The MultiLJP files are **source tables**, not the PDF's filtered 23/22 label space.
The main workflow derives its charge catalog from the verified prepared training
split with `python -m core label-space`. It does not use these MultiLJP tables as
paper defaults, invent a frequency filter, or discard unknown predicted labels.

Use the exact ordered CAIL tables for indexed source labels. Official law-article
numbers do not require index decoding. `predict --charges` takes charge names,
one per line; law-table export is a reproducibility artifact, not additional law
text supplied to judges. See [paper correspondence](../docs/paper_alignment.md).
