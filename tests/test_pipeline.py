"""Real FAISS integration with fake API calls; no credentials or network needed."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.cli import main
from core.config import load_config
from core.crj import judge
from core.evaluate import evaluate
from core.io import read_jsonl
from core.mpr import Retriever, build_index, embed_record
from core.rpr import rerank


class FakeAPI:
    model = "offline-test-model"

    def __init__(self, *args):
        self.prompts = []

    async def embed(self, texts):
        return [[0., 1., 2.] for _ in texts]

    async def chat(self, prompt, temperature):
        self.prompts.append(prompt)
        term = 12
        return json.dumps({"罪名": "盗窃", "法条": 264, "刑期": term}, ensure_ascii=False)

    async def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = {**load_config(), "embedding_dimension": 3}
        self.records = [{"id": f"train-{i}", "fact": "训练事实", "elements": {
            "behavior": "窃取财物", "consequence": "财物损失", "fact": "训练事实"},
            "labels": {"charge": "盗窃", "law_article": 264, "term_months": 12, "term_class": 7},
            "embeddings": {f: [float(i), 1., 2.] for f in ("behavior", "consequence", "fact")},
            "embedding_model": FakeAPI.model} for i in range(2)]
        self.training = self.directory / "training.jsonl"
        self.training.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.records), encoding="utf-8")
        self.index = self.directory / "index"

    def tearDown(self):
        self.temp.cleanup()

    def query(self):
        return {"id": "query", "fact": "查询事实", "elements": {
            "behavior": "窃取财物", "consequence": "财物损失", "fact": "查询事实"}}

    def test_embed_faiss_rpr_two_judges_end_to_end(self):
        self.assertEqual(build_index(self.training, self.index, self.config), 2)
        retriever = Retriever(self.index)
        query = asyncio.run(embed_record(self.query(), FakeAPI(), 3))
        lists = retriever.search(query, 5000)
        self.assertEqual(lists, [[0, 1], [0, 1], [0, 1]])
        ranked = rerank(lists)
        fake = FakeAPI()
        prediction = asyncio.run(judge(query, ranked, retriever.precedents, ["盗窃"], fake, self.config))
        self.assertEqual(len(fake.prompts), 2)
        self.assertIn("法官A的初步预测", fake.prompts[1])
        self.assertEqual(prediction["judge_a"]["term_class"], 7)
        self.assertEqual(prediction["judge_b"]["term_class"], 7)
        self.assertEqual(prediction["second_round_ids"], ["train-0", "train-1"])

    def test_embedding_dimension_and_model_mismatch_are_rejected(self):
        build_index(self.training, self.index, self.config)
        retriever = Retriever(self.index)
        query = asyncio.run(embed_record(self.query(), FakeAPI(), 3))
        query["embedding_model"] = "different"
        with self.assertRaises(ValueError):
            retriever.search(query, 5000)
        with self.assertRaises(ValueError):
            asyncio.run(embed_record(self.query(), FakeAPI(), 1536))

    def test_metrics_report_failures_and_missing_predictions(self):
        truth = [{"id": "a", "labels": self.records[0]["labels"]},
                 {"id": "b", "labels": self.records[0]["labels"]}]
        truth_path = self.directory / "truth.jsonl"
        truth_path.write_text("\n".join(json.dumps(r) for r in truth), encoding="utf-8")
        prediction_path = self.directory / "predictions.jsonl"
        prediction_path.write_text(json.dumps({"id": "a", "status": "ok",
            "judge_a": truth[0]["labels"], "judge_b": truth[0]["labels"]}), encoding="utf-8")
        report = evaluate(truth_path, prediction_path)
        self.assertEqual(report["coverage"], .5)
        self.assertEqual(report["failed_or_missing"], 1)
        self.assertAlmostEqual(report["judge_b"]["charge"]["macro_f1"], 100 * 2 / 3)
        self.assertEqual(report["judge_b"]["charge"]["accuracy"], 50)

    def test_cli_prediction_records_failures_and_returns_nonzero(self):
        build_index(self.training, self.index, self.config)
        query = asyncio.run(embed_record(self.query(), FakeAPI(), 3))
        query["embedding_model"] = "wrong-model"
        input_path = self.directory / "query.jsonl"
        input_path.write_text(json.dumps(query), encoding="utf-8")
        config_path = self.directory / "config.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        charges = self.directory / "charges.txt"
        charges.write_text("盗窃\n", encoding="utf-8")
        output = self.directory / "out.jsonl"
        with patch("core.api.API", FakeAPI):
            result = main(["predict", "--input", str(input_path), "--output", str(output),
                           "--index-dir", str(self.index), "--charges", str(charges),
                           "--config", str(config_path)])
        self.assertEqual(result, 1)
        self.assertEqual(list(read_jsonl(output))[0]["status"], "error")

    def test_cli_refuses_to_overwrite_outputs(self):
        output = self.directory / "existing.jsonl"
        output.write_text("original", encoding="utf-8")
        result = main(["prepare", "--input", str(self.training), "--output", str(output),
                       "--format", "canonical"])
        self.assertEqual(result, 1)
        self.assertEqual(output.read_text(), "original")

    def test_duplicate_input_identity_is_rejected(self):
        self.training.write_text("\n".join([json.dumps(self.records[0])] * 2), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate record"):
            build_index(self.training, self.index, self.config)


if __name__ == "__main__":
    unittest.main()
