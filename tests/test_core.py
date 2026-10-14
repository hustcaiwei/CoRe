"""Offline regressions for retrieval, label handling, and dataset adapters."""
import importlib
import unittest
from pathlib import Path


def module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


class CoreTests(unittest.TestCase):
    def require(self, name):
        result = module(name)
        self.assertIsNotNone(result, f"The organized repository must provide {name}")
        return result

    def test_rpr_uses_one_based_ranks_and_missing_facets_contribute_zero(self):
        rpr = self.require("core.rpr")
        ranked = rpr.rerank([[2, 0], [0, 1], [0, 2]], alpha=60, scope="union")
        self.assertEqual(ranked[0][0], 0)
        self.assertAlmostEqual(ranked[0][1], 1 / 62 + 2 / 61)
        self.assertEqual({i for i, _ in ranked}, {0, 1, 2})

    def test_rpr_intersection_and_invalid_faiss_ids(self):
        rpr = self.require("core.rpr")
        ranked = rpr.rerank([[0, 1, -1], [1, 2], [1]], scope="intersection")
        self.assertEqual([i for i, _ in ranked], [1])
        self.assertNotIn(-1, [i for i, _ in rpr.rerank([[0, -1], [1], []])])

    def test_second_round_keeps_matching_labels_and_backfills_without_duplicates(self):
        rpr = self.require("core.rpr")
        pool = [{"labels": {"charge": "盗窃", "law_article": 264}},
                {"labels": {"charge": "诈骗", "law_article": 266}},
                {"labels": {"charge": "盗窃", "law_article": 264}},
                {"labels": {"charge": "盗窃", "law_article": 266}}]
        ranked = [(1, .3), (2, .2), (3, .1), (0, .05)]
        self.assertEqual(rpr.select_second_round(ranked, pool, "盗窃", 264, 3), [2, 0, 1])
        self.assertEqual(rpr.select_second_round(ranked, pool, "无匹配", 999, 3), [1, 2, 3])

    def test_term_boundaries_and_special_penalties(self):
        data = self.require("core.data")
        for months, expected in [(-1, 0), (360, 1), (121, 1), (120, 2),
                                 (84, 3), (60, 4), (36, 5), (24, 6),
                                 (12, 7), (9, 8), (6, 9), (0, 10)]:
            self.assertEqual(data.term_class(months), expected)
        self.assertEqual(data.term_months("有期徒刑一年十个月"), 22)
        self.assertEqual(data.term_months("无期徒刑"), 360)
        self.assertEqual(data.term_months("死刑"), -1)
        with self.assertRaises(ValueError):
            data.term_months("未提供刑期")

    def test_multiljp_is_split_per_defendant_with_stable_identity(self):
        data = self.require("core.data")
        source = {"id": 8, "elements": {
            "被告A": {"行为": "A行为", "后果": "A后果", "犯罪事实": "A事实"},
            "被告B": {"行为": "B行为", "后果": "B后果", "犯罪事实": "B事实"}},
            "crime_info": {
                "[被告A]": {"accusations": ["盗窃罪"], "laws": [264], "term": "一年"},
                "[被告B]": {"accusations": ["诈骗罪"], "laws": [266], "term": "两年"}}}
        records = data.normalize(source, "multiljp-elements")
        self.assertEqual([r["id"] for r in records], ["8:被告A", "8:被告B"])
        self.assertEqual(records[1]["elements"]["fact"], "B事实")
        self.assertEqual(records[0]["labels"]["charge"], ["盗窃"])
        self.assertEqual(records[1]["labels"]["term_class"], 6)

    def test_cail_extracted_elements_do_not_turn_predicted_charge_into_truth(self):
        data = self.require("core.data")
        record = data.normalize({"id": 1, "behavior": "行为", "consequence": "后果",
                                 "fact_summary": "事实", "pred_accu": "盗窃"}, "cail-elements")[0]
        self.assertNotIn("labels", record)

    def test_prediction_prompts_never_include_query_ground_truth(self):
        crj = self.require("core.crj")
        query = {"id": "q", "fact": "可见事实", "labels": {
            "charge": "DO_NOT_LEAK", "law_article": 999999, "term_months": 888888}}
        prompt = crj.build_prompt(query, [], ["盗窃"], stage="a")
        self.assertIn("可见事实", prompt)
        self.assertNotIn("DO_NOT_LEAK", prompt)
        self.assertNotIn("999999", prompt)
        self.assertNotIn("888888", prompt)

    def test_prediction_validates_types_and_rejects_python_expressions(self):
        crj = self.require("core.crj")
        output = crj.parse_prediction('{"罪名":"盗窃罪","法条":264,"刑期":12}', "months")
        self.assertEqual(output, {"charge": ["盗窃"], "law_article": [264],
                                  "term_months": 12, "term_class": 7})
        with self.assertRaises(ValueError):
            crj.parse_prediction('{"罪名":"盗窃","法条":true,"刑期":3}', "months")
        with self.assertRaises(ValueError):
            crj.parse_prediction('{"罪名":"盗窃","法条":"1+263","刑期":3}', "months")

    def test_label_table_lines_preserve_zero_based_cail_mapping(self):
        cli = self.require("core.cli")
        root = Path(__file__).resolve().parents[1]
        charges = cli.table(root / "resources/cail_charges.txt")
        laws = cli.table(root / "resources/cail_laws.txt")
        self.assertEqual(len(charges), 119)
        self.assertEqual(len(laws), 103)
        self.assertTrue(all(charges))
        self.assertTrue(all(laws))
        data = self.require("core.data")
        record = data.normalize({"id": 7, "fact_cut": "事实", "accu": 0, "law": 0,
                                 "term": 360, "term_cate": 1}, "cail", charges, laws)[0]
        self.assertEqual(record["labels"]["law_article"], [int(laws[0])])
        self.assertEqual(record["labels"]["charge"], [data.charge_name(charges[0])])
        self.assertEqual(record["labels"]["term_class"], 1)


if __name__ == "__main__":
    unittest.main()
