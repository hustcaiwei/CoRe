"""Regressions derived from PDF sections 2.3-2.5 and appendices B/F."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch
from core import cee
from core.cli import main
from core.io import read_jsonl
from core.config import load_config
from core.crj import build_prompt, parse_prediction
from core.data import normalize, term_months
from core.evaluate import evaluate
from core.rpr import select_second_round


class PaperContractTests(unittest.TestCase):
    def test_cli_extracted_cases_join_original_descriptions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {"id": "a", "behavior": "行为", "consequence": "后果", "fact_summary": "摘要"}
            (root / "elements").write_text(json.dumps(record), encoding="utf-8")
            (root / "original").write_text(json.dumps({"id": "a", "fact": "完整案情"}), encoding="utf-8")
            args = ["prepare", "--input", str(root / "elements"), "--format", "cail-elements"]
            self.assertEqual(main(args + ["--output", str(root / "missing")]), 1)
            self.assertEqual(main(args + ["--output", str(root / "joined"),
                "--source-cases", str(root / "original")]), 0)
            self.assertEqual(list(read_jsonl(root / "joined"))[0]["fact"], "完整案情")
            multi = {"id": "a", "elements": {"被告": {"行为": "行为", "后果": "后果", "犯罪事实": "摘要"}},
                "crime_info": {"[被告]": {"accusations": ["盗窃", "诈骗"], "laws": [264, 266], "term": 12}}}
            (root / "multi").write_text(json.dumps(multi), encoding="utf-8")
            self.assertEqual(main(["prepare", "--input", str(root / "multi"), "--format", "multiljp-elements",
                "--output", str(root / "multi-joined"), "--source-cases", str(root / "original")]), 0)
            result = list(read_jsonl(root / "multi-joined"))[0]
            self.assertEqual(result["fact"], "完整案情")
            self.assertEqual(result["labels"]["law_article"], [264, 266])

    def test_cli_full_pipeline_requires_audit_and_preserves_multiple_labels(self):
        class Fake:
            model = "offline"
            def __init__(self, config, purpose): self.purpose = purpose
            async def close(self): pass
            async def embed(self, texts): return [[0., 1., 2.] for _ in texts]
            async def chat(self, prompt, temperature, max_tokens=None):
                if self.purpose == "cee-evaluate":
                    return json.dumps({"scores": {k: 5 for k in ("behavior_faithfulness",
                        "behavior_completeness", "consequence_faithfulness", "summary_quality",
                        "non_hallucination", "overall")}, "severe_hallucination": False,
                        "error_types": [], "revision_advice": ""})
                if "罪名库" in prompt:
                    return json.dumps({"罪名": ["盗窃", "诈骗"], "法条": [264, 266], "刑期": 12})
                return json.dumps({"behavior": "行为", "consequence": "后果", "fact": "摘要"})
        with tempfile.TemporaryDirectory() as temp, patch("core.api.API", Fake):
            root = Path(temp)
            (root / "config").write_text(json.dumps({"embedding_dimension": 3}), encoding="utf-8")
            config = ["--config", str(root / "config")]
            for split in ("train", "test"):
                raw = {"id": split, "fact": "完整案情", "criminals_info": {"[被告]": {
                    "accusations": ["盗窃", "诈骗"], "laws": [264, 266], "term": 12}}}
                (root / (split + ".raw")).write_text(json.dumps(raw), encoding="utf-8")
                self.assertEqual(main(["prepare", "--input", str(root / (split + ".raw")),
                    "--format", "multiljp", "--output", str(root / split)]), 0)
                previous = root / split
                for stage in ("extract", "audit", "embed"):
                    output = root / (split + "." + stage)
                    if stage == "audit":
                        self.assertEqual(main(["embed", "--input", str(previous),
                            "--output", str(root / (split + ".unaudited"))] + config), 1)
                    self.assertEqual(main([stage, "--input", str(previous), "--output", str(output)] + config), 0)
                    previous = output
            self.assertEqual(main(["label-space", "--input", str(root / "train"),
                "--charges-out", str(root / "charges"), "--laws-out", str(root / "laws")]), 0)
            self.assertEqual((root / "charges").read_text(encoding="utf-8").splitlines(), ["盗窃", "诈骗"])
            self.assertEqual(main(["index", "--input", str(root / "train.embed"),
                "--index-dir", str(root / "index")] + config), 0)
            self.assertEqual(main(["predict", "--input", str(root / "test.embed"),
                "--index-dir", str(root / "index"), "--charges", str(root / "charges"),
                "--output", str(root / "predictions")] + config), 0)
            report = evaluate(root / "test", root / "predictions")
            self.assertEqual(report["judge_b"]["charge"]["accuracy"], 100)
            self.assertEqual(report["judge_b"]["law_article"]["accuracy"], 100)
            self.assertEqual(report["coverage"], 1)

    def test_suspended_term_does_not_change_custodial_months(self):
        self.assertEqual(term_months("有期徒刑三年，缓刑三年六个月"), 36)
        with self.assertRaises(ValueError):
            term_months("有期徒刑一年，拘役六个月")

    def test_cail_extracted_summary_is_not_original_description(self):
        record = {"id": "a", "behavior": "行为", "consequence": "后果", "fact_summary": "摘要"}
        self.assertEqual(normalize(record, "cail-elements")[0]["fact"], "")

    def test_cee_keep_criterion_is_overall_and_severe_hallucination(self):
        class Fake:
            model = "offline"
            async def chat(self, prompt, temperature, max_tokens=None):
                return json.dumps({"scores": {"behavior_faithfulness": 4, "behavior_completeness": 4,
                    "consequence_faithfulness": 4, "summary_quality": 4, "non_hallucination": 4,
                    "overall": 4}, "severe_hallucination": False,
                    "error_types": ["hallucination"], "revision_advice": ""})
        record = {"id": "a", "fact": "完整案情", "elements": {
            "behavior": "行为", "consequence": "后果", "fact": "摘要"}}
        result = asyncio.run(cee.audit_and_repair(record, Fake(), load_config()))
        self.assertEqual(result["cee_action"], "keep")

    def test_multiljp_preserves_all_labels_and_full_case_description(self):
        record = {"id": 4, "fact": "完整案件原文", "criminals_info": {
            "[被告A]": {"accusations": ["盗窃罪", "诈骗罪"], "laws": [264, 266], "term": "两年"},
            "[被告B]": {"accusations": ["盗窃罪"], "laws": [264], "term": "六个月"}}}
        result = normalize(record, "multiljp")
        self.assertEqual(result[0]["labels"]["charge"], ["盗窃", "诈骗"])
        self.assertEqual(result[0]["labels"]["law_article"], [264, 266])
        self.assertEqual(result[0]["fact"], result[1]["fact"])
        self.assertEqual([r["labels"]["term_months"] for r in result], [24, 6])

    def test_parser_preserves_label_arrays_and_rejects_unassigned_multiple_terms(self):
        result = parse_prediction('{"罪名":["盗窃罪","诈骗罪"],"法条":[264,266],"刑期":24}', "months")
        self.assertEqual(result["charge"], ["盗窃", "诈骗"])
        self.assertEqual(result["law_article"], [264, 266])
        with self.assertRaises(ValueError):
            parse_prediction('{"罪名":["盗窃"],"法条":[264],"刑期":[12,24]}', "months")

    def test_secondary_selection_compares_label_sets_without_order_dependence(self):
        pool = [{"labels": {"charge": ["盗窃", "诈骗"], "law_article": [264, 266]}},
                {"labels": {"charge": ["盗窃"], "law_article": [264]}}]
        self.assertEqual(select_second_round([(1, .2), (0, .1)], pool,
                         ["诈骗", "盗窃"], [266, 264], 1), [0])

    def test_prompt_uses_original_multiljp_fact_and_array_output(self):
        query = {"id": "q", "fact": "完整原始案情", "defendant": "被告A", "dataset": "multiljp",
                 "elements": {"fact": "压缩摘要"}}
        prompt = build_prompt(query, [], ["盗窃", "诈骗"])
        self.assertIn("完整原始案情", prompt)
        self.assertIn('"罪名":[', prompt)
        self.assertNotIn("压缩摘要", prompt)

    def test_paper_defaults_use_intersection(self):
        self.assertEqual(load_config()["rpr_scope"], "intersection")

    def test_extra_predicted_labels_reduce_set_accuracy_and_macro_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = [{"id": "a", "labels": {"charge": ["盗窃"], "law_article": [264], "term_class": 7}},
                     {"id": "b", "labels": {"charge": ["诈骗"], "law_article": [266], "term_class": 7}}]
            predictions = []
            for r in truth:
                pred = dict(r["labels"])
                if r["id"] == "a": pred["charge"] = ["盗窃", "诈骗"]
                predictions.append({"id": r["id"], "status": "ok", "judge_a": pred, "judge_b": pred})
            for name, rows in [("truth", truth), ("pred", predictions)]:
                (root / (name + ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            report = evaluate(root / "truth.jsonl", root / "pred.jsonl")
            self.assertEqual(report["judge_b"]["charge"]["accuracy"], 50)
            self.assertEqual(report["judge_b"]["charge"]["macro_precision"], 75)
            self.assertAlmostEqual(report["judge_b"]["charge"]["macro_f1"], 100 * 5 / 6)

    def test_all_failed_predictions_are_scored_instead_of_disappearing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "truth").write_text(json.dumps({"id": "a", "labels": {
                "charge": ["盗窃"], "law_article": [264], "term_class": 7}}), encoding="utf-8")
            (root / "pred").write_text(json.dumps({"id": "a", "status": "error"}), encoding="utf-8")
            report = evaluate(root / "truth", root / "pred")
            self.assertEqual(report["coverage"], 0)
            for field in ("charge", "law_article", "term_class"):
                self.assertEqual(report["judge_b"][field]["accuracy"], 0)
                self.assertEqual(report["judge_b"][field]["macro_f1"], 0)

    def test_cee_severe_hallucination_cannot_pass_with_a_high_overall_score(self):
        method = getattr(cee, "audit_and_repair", None)
        self.assertIsNotNone(method)
        class Fake:
            model = "offline"
            def __init__(self): self.calls = 0
            async def chat(self, prompt, temperature, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"scores": {"behavior_faithfulness": 4, "behavior_completeness": 4,
                        "consequence_faithfulness": 4, "summary_quality": 4, "non_hallucination": 0,
                        "overall": 4}, "severe_hallucination": True,
                        "error_types": ["hallucination"], "revision_advice": "去除虚构内容"})
                return json.dumps({"behavior": "击打", "consequence": "轻微伤", "fact": "击打致轻微伤"})
        record = {"id": "a", "fact": "击打致轻微伤", "elements": {
            "behavior": "击打", "consequence": "死亡", "fact": "击打致死亡"}}
        result = asyncio.run(method(record, Fake(), load_config()))
        self.assertEqual(result["cee_action"], "re_extract_with_strict_evidence")
        self.assertEqual(result["elements"]["consequence"], "轻微伤")

    def test_raw_model_aliases_and_unknown_extra_labels_are_preserved(self):
        output = parse_prediction('{"pred_accusation":["盗窃","诈骗"],"pred_relevant_articles":[264,266],"pred_imprisonment_months":12}', "months")
        self.assertEqual(output["law_article"], [264, 266])
        self.assertEqual(output["term_class"], 7)

    def test_cee_evaluation_and_targeted_repair_preserve_correct_facets(self):
        method = getattr(cee, "audit_and_repair", None)
        self.assertIsNotNone(method, "CEE must implement the paper's evaluation/repair stage")
        class Fake:
            model = "offline-evaluator"
            def __init__(self): self.calls = 0
            async def chat(self, prompt, temperature, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"scores": {"behavior_faithfulness": 5, "behavior_completeness": 5,
                        "consequence_faithfulness": 2, "summary_quality": 5, "non_hallucination": 5,
                        "overall": 3}, "severe_hallucination": False,
                        "error_types": ["invalid_consequence"], "revision_advice": "纠正后果"})
                return json.dumps({"consequence": "轻微伤"})
        record = {"id": "a", "fact": "造成轻微伤", "elements": {
            "behavior": "击打", "consequence": "死亡", "fact": "击打致轻微伤"}}
        result = asyncio.run(method(record, Fake(), load_config()))
        self.assertEqual(result["elements"]["behavior"], "击打")
        self.assertEqual(result["elements"]["consequence"], "轻微伤")
        self.assertEqual(result["elements"]["fact"], "击打致轻微伤")
        self.assertEqual(result["cee_action"], "repair_consequence_only")


if __name__ == "__main__": unittest.main()
