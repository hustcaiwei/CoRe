"""Unified entry point: prepare -> extract -> audit -> embed -> index -> predict -> evaluate."""
import argparse
import asyncio
import json
import sys
from itertools import islice
from pathlib import Path

from .config import load_config
from .data import normalize, charge_labels, article_labels
from .io import batches, output_file, read_jsonl, unique_records, write_record


def table(path):
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8-sig").splitlines()


def inputs(args):
    records = unique_records(read_jsonl(args.input))
    return islice(records, args.limit) if getattr(args, "limit", None) else records


async def run_async(args, config):
    from .api import API
    from .cee import extract, audit_and_repair
    from .crj import judge
    from .mpr import Retriever, embed_record
    from .rpr import rerank

    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}; choose a new output path")
    purpose = {"embed": "embed", "audit": "cee-evaluate"}.get(args.command, "chat")
    retriever = Retriever(args.index_dir) if args.command == "predict" else None
    charges = table(args.charges) if args.command == "predict" else None
    if retriever and (retriever.metadata["dimension"] != config["embedding_dimension"]):
        raise ValueError("Configured embedding dimension differs from the precedent index")
    api = API(config, purpose)
    repair_api = None
    failed, count = 0, 0

    async def process(record):
        try:
            if record.get("status") == "error":
                raise ValueError("The input record failed in a previous stage")
            if args.command == "extract":
                return await extract(record, api, config["cee_temperature"])
            if args.command == "audit":
                return await audit_and_repair(record, api, config, repair_api)
            if args.command == "embed":
                if record.get("cee_validated") is not True:
                    raise ValueError("Run the CEE audit/repair stage before embedding")
                return await embed_record(record, api, config["embedding_dimension"])
            ranked = rerank(retriever.search(record, config["top_m"]),
                            config["alpha"], config["rpr_scope"])
            if not ranked:
                raise ValueError("RPR produced no candidate precedents")
            result = await judge(record, ranked, retriever.precedents, charges, api, config)
            result["run_config"] = {**config, "llm_model": api.model,
                                     "embedding_model": retriever.metadata["embedding_model"]}
            return result
        except Exception as error:
            # Preserve failed extraction/embedding inputs for explicit inspection.
            result = {**record} if args.command != "predict" else {"id": record["id"]}
            return {**result, "status": "error", "error_type": type(error).__name__,
                    "error": str(error)}

    try:
        if args.command == "audit": repair_api = API(config, "cee-repair")
        with output_file(args.output) as handle:
            for batch in batches(inputs(args), config["batch_size"]):
                results = await asyncio.gather(*(process(record) for record in batch))
                for result in results:
                    write_record(handle, result)
                    count += 1
                    failed += result["status"] == "error"
                print(f"{args.command}: processed={count}, failed={failed}", file=sys.stderr)
    finally:
        await api.close()
        if repair_api: await repair_api.close()
    return 1 if failed else 0


def parser():
    result = argparse.ArgumentParser(description="CoRe legal judgment prediction research pipeline")
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Normalize CAIL/MultiLJP JSONL records")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--format", required=True,
                         choices=("canonical", "cail", "cail-elements", "multiljp", "multiljp-elements"))
    prepare.add_argument("--charges", help="Zero-based label table for indexed CAIL charges")
    prepare.add_argument("--laws", help="Zero-based label table for indexed CAIL law articles")
    prepare.add_argument("--source-cases", help="Original cases JSONL, joined by origin_id (MultiLJP) or id (CAIL)")
    for command, help_text in (("extract", "CEE: extract three textual case elements"),
                               ("audit", "CEE: evaluate and automatically repair extracted elements"),
                               ("embed", "MPR: encode three case elements"),
                               ("predict", "MPR + RPR + CRJ: run Judge A then Judge B")):
        stage = sub.add_parser(command, help=help_text)
        stage.add_argument("--input", required=True)
        stage.add_argument("--output", required=True)
        stage.add_argument("--config")
        stage.add_argument("--limit", type=int, help="Process at most this many records")
        if command == "predict":
            stage.add_argument("--index-dir", required=True)
            stage.add_argument("--charges", required=True, help="UTF-8 charge names, one per line")
    index = sub.add_parser("index", help="Build three training-only IndexFlatL2 indices")
    index.add_argument("--input", required=True)
    index.add_argument("--index-dir", required=True)
    index.add_argument("--config")
    evaluate = sub.add_parser("evaluate", help="Report both judges' metrics and prediction coverage")
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output")
    catalog = sub.add_parser("label-space", help="Export the actual label space from a prepared training split")
    catalog.add_argument("--input", required=True)
    catalog.add_argument("--charges-out", required=True)
    catalog.add_argument("--laws-out", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "limit", None) is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.command == "prepare":
            charges, laws = table(args.charges), table(args.laws)
            normalized = (item for record in read_jsonl(args.input)
                          for item in normalize(record, args.format, charges, laws))
            descriptions = ({str(r["id"]): r["fact"] for r in unique_records(read_jsonl(args.source_cases))}
                            if args.source_cases else {})
            with output_file(args.output) as handle:
                count = 0
                for item in unique_records(normalized):
                    if args.source_cases:
                        key = str(item.get("origin_id", item["id"]))
                        if key not in descriptions: raise ValueError(f"Missing original case for origin_id={key}")
                        item["fact"] = descriptions[key]
                    if not isinstance(item["fact"], str) or not item["fact"].strip():
                        raise ValueError("The original case description is required; supply --source-cases for extracted-only files")
                    write_record(handle, item)
                    count += 1
            print(f"Prepared {count} records")
            return 0
        if args.command == "label-space":
            charges, articles = set(), set()
            for record in unique_records(read_jsonl(args.input)):
                charges.update(charge_labels(record["labels"]["charge"]))
                articles.update(article_labels(record["labels"]["law_article"]))
            with output_file(args.charges_out) as handle: handle.write("\n".join(sorted(charges)) + "\n")
            with output_file(args.laws_out) as handle: handle.write("\n".join(map(str, sorted(articles))) + "\n")
            print(f"Training label space: {len(charges)} charges, {len(articles)} law articles")
            return 0
        if args.command == "evaluate":
            from .evaluate import evaluate
            report = evaluate(args.truth, args.predictions)
            text = json.dumps(report, ensure_ascii=False, indent=2)
            print(text)
            if args.output:
                with output_file(args.output) as handle:
                    handle.write(text + "\n")
            return 0
        config = load_config(args.config)
        if args.command == "index":
            from .mpr import build_index
            count = build_index(args.input, args.index_dir, config)
            print(f"Indexed {count} training precedents")
            return 0
        return asyncio.run(run_async(args, config))
    except (ValueError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
