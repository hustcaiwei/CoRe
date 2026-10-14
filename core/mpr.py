"""MPR: independent behavior/consequence/fact search by squared L2 distance."""
import json
from pathlib import Path

import faiss
import numpy as np

from .data import FACETS
from .io import batches, output_file, read_jsonl, unique_records, write_record


def vector(record, facet, dimension):
    value = np.asarray(record["embeddings"][facet], dtype=np.float32)
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise ValueError(f"Record {record['id']}: invalid {facet} vector, expected dimension {dimension}")
    return value


async def embed_record(record, api, dimension):
    texts = [record["elements"][facet] for facet in FACETS]
    values = await api.embed(texts)
    result = {**record, "embeddings": dict(zip(FACETS, values)),
              "embedding_model": api.model, "status": "ok"}
    for facet in FACETS:
        vector(result, facet, dimension)
    return result


def build_index(input_path, index_dir, config):
    """Build training-only indices and a row-aligned precedent file."""
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=False)
    dimension = config["embedding_dimension"]
    indices = {facet: faiss.IndexFlatL2(dimension) for facet in FACETS}
    count = 0
    model = None
    with output_file(directory / "precedents.jsonl") as handle:
        for batch in batches(unique_records(read_jsonl(input_path)), config["batch_size"]):
            for record in batch:
                if "labels" not in record or record.get("status") == "error":
                    raise ValueError(f"Training precedent {record['id']} has no valid ground-truth labels")
                if not record.get("embedding_model"):
                    raise ValueError("Training vectors must record their embedding_model")
                if model is None:
                    model = record["embedding_model"]
                if record["embedding_model"] != model:
                    raise ValueError("Cannot mix embedding models in one index")
            for facet in FACETS:
                indices[facet].add(np.stack([vector(r, facet, dimension) for r in batch]))
            for record in batch:
                write_record(handle, {key: value for key, value in record.items() if key != "embeddings"})
                count += 1
    if not count:
        raise ValueError("Cannot build an empty precedent index")
    for facet, index in indices.items():
        faiss.write_index(index, str(directory / f"{facet}.index"))
    (directory / "metadata.json").write_text(json.dumps({"count": count, "dimension": dimension,
        "embedding_model": model, "facets": list(FACETS), "metric": "squared_l2"}, indent=2), encoding="utf-8")
    return count


class Retriever:
    def __init__(self, index_dir):
        directory = Path(index_dir)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.precedents = list(unique_records(read_jsonl(directory / "precedents.jsonl")))
        self.indices = {f: faiss.read_index(str(directory / f"{f}.index")) for f in FACETS}
        if self.metadata["count"] != len(self.precedents) or any(
                index.ntotal != len(self.precedents) or index.d != self.metadata["dimension"]
                for index in self.indices.values()):
            raise ValueError("Indices and precedent rows are not aligned")

    def search(self, record, top_m):
        if record.get("embedding_model") != self.metadata["embedding_model"]:
            raise ValueError("Query and training embeddings must use the same model")
        result = []
        for facet in FACETS:
            query = vector(record, facet, self.metadata["dimension"])[None, :]
            # Clamp to ntotal: FAISS uses -1 padding when k exceeds index size.
            _, ids = self.indices[facet].search(query, min(top_m, self.indices[facet].ntotal))
            result.append([int(case_id) for case_id in ids[0] if case_id >= 0])
        return result
