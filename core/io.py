"""Streaming UTF-8 JSONL input/output shared by the experiment stages."""
import json
from itertools import islice
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield value


def batches(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def output_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")


def write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def unique_records(items):
    seen = set()
    for item in items:
        key = str(item["id"])
        if key in seen:
            raise ValueError(f"Duplicate record id: {key}")
        seen.add(key)
        yield item
