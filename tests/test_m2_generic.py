# -*- coding: utf-8 -*-
"""v0.3 M2 通用化研究流程 测试。

验证 run_research 真正由实验定义驱动:
  - 默认 experiment_id 兼容旧调用;
  - build_prompt 使用实验的 description/candidate_field/allowed_values/scoring_hint;
  - 候选校验复用 make_default_validator(类型/有限性/白名单);
  - 评分由程序计算,不使用模型自报分数;
  - 未知实验 ID 明确报错,不静默回退;
  - 不完整/非法实验定义尽早报错。
不调用真实 API(全部 mock),写盘测试使用临时目录。
"""
import json
import math
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, r"c:\Users\lenovo\Desktop\Prometheus")

from prometheus import experiment
from prometheus import experiments
from prometheus.experiments import DEFAULT_EXPERIMENT, validate_experiment_definition

CFG = {"model": "m", "base_url": "http://x", "api_key": "k"}


class TestM2GenericDriver(unittest.TestCase):
    def _research(self, reply, experiment_id=experiments.DEFAULT_EXPERIMENT_ID,
                  goal="goal", rounds=1):
        with mock.patch.object(experiment, "call_model", return_value=reply) as m:
            result = experiment.run_research(CFG, goal, rounds,
                                             experiment_id=experiment_id)
        return result, m

    def test_default_experiment_id_compat(self):
        # 旧调用方式(不传 experiment_id)仍默认 toy_7_3。
        result, _ = self._research('{"candidate_x": 7.5}')
        self.assertEqual(result["best_candidate"]["x"], 7.5)
        self.assertAlmostEqual(result["best_candidate"]["score"], 1 / (1 + 0.2 ** 2))
        self.assertEqual(result["goal"], "goal")
        self.assertEqual(result["rounds_requested"], 1)
        self.assertEqual(len(result["round_results"]), 1)

    def test_round_results_semantics(self):
        result, _ = self._research(
            '{"candidate_x": 7.5, "reason": "nearest"}',
            goal="g", rounds=2)
        self.assertEqual(len(result["round_results"]), 2)
        rr = result["round_results"][0]
        self.assertEqual(rr["round"], 1)
        self.assertEqual(rr["candidate_x"], 7.5)
        self.assertEqual(rr["reason"], "nearest")

    def test_build_prompt_uses_experiment_fields(self):
        prompt = experiment.build_prompt("my goal", DEFAULT_EXPERIMENT)
        self.assertIn(DEFAULT_EXPERIMENT["description"], prompt)
        self.assertIn(DEFAULT_EXPERIMENT["scoring_hint"], prompt)
        self.assertIn(DEFAULT_EXPERIMENT["candidate_field"], prompt)
        self.assertIn("my goal", prompt)

    def test_prompt_candidates_match_allowed_values(self):
        prompt = experiment.build_prompt("g", DEFAULT_EXPERIMENT)
        domain_str = json.dumps(DEFAULT_EXPERIMENT["allowed_values"])
        self.assertIn(domain_str, prompt)

    def test_candidate_field_used_everywhere(self):
        default = DEFAULT_EXPERIMENT
        result, _ = self._research('{"candidate_x": 7.5}')
        rr = result["round_results"][0]
        # history 用 candidate_field 作为键
        self.assertIn(default["candidate_field"], rr)
        self.assertEqual(rr[default["candidate_field"]], 7.5)
        self.assertEqual(rr["candidate_x"], 7.5)

    def test_validator_rejects_bad_candidates(self):
        bads = [
            '{"candidate_x": true}',
            '{"candidate_x": false}',
            '{"candidate_x": "7.5"}',
            '{"candidate_x": null}',
            '{"candidate_x": [7.5]}',
            '{"candidate_x": {"x": 7.5}}',
            '{"candidate_x": NaN}',
            '{"candidate_x": Infinity}',
            '{"candidate_x": -Infinity}',
            '{"candidate_x": 15.0}',
            '{"candidate_x": -1.0}',
            '{"reason": "missing field"}',
            '{"candidate_x": 7.2}',   # 不在白名单(非 0.5 步长)
        ]
        for bad in bads:
            with self.subTest(bad=bad):
                with mock.patch.object(experiment, "call_model",
                                       return_value=bad):
                    with self.assertRaises(ValueError):
                        experiment.run_research(CFG, "g", 1)

    def test_validator_accepts_legit(self):
        # 有限、白名单内、非 bool 的数值应被接受。
        for good in ['{"candidate_x": 7.5}', '{"candidate_x": 0.0}',
                     '{"candidate_x": 10.0}']:
            with self.subTest(good=good):
                result, _ = self._research(good)
                self.assertEqual(len(result["round_results"]), 1)

    def test_score_computed_by_program_not_model(self):
        # 模型返回自报 score=0.0,程序仍按自身评分(7.5 -> 0.9615)。
        result, _ = self._research(
            '{"candidate_x": 7.5, "score": 0.0, "reason": "self-report"}')
        rr = result["round_results"][0]
        self.assertAlmostEqual(rr["score"], 1 / (1 + 0.2 ** 2))
        self.assertNotEqual(rr["score"], 0.0)

    def test_unknown_experiment_id_raises(self):
        with self.assertRaises(KeyError):
            experiment.run_research(CFG, "g", 1, experiment_id="no_such_exp")

    def test_invalid_definition_raises_early(self):
        bad_missing = {"id": "x", "candidate_field": "c", "allowed_values": [1.0],
                       "scoring_hint": "h", "score": lambda x: 1.0}
        with self.assertRaises(ValueError):
            validate_experiment_definition(bad_missing)
        bad_allowed = {"id": "x", "description": "d", "candidate_field": "c",
                       "allowed_values": [], "scoring_hint": "h",
                       "score": lambda x: 1.0}
        with self.assertRaises(ValueError):
            validate_experiment_definition(bad_allowed)
        bad_score = {"id": "x", "description": "d", "candidate_field": "c",
                     "allowed_values": [1.0], "scoring_hint": "h", "score": 3}
        with self.assertRaises(ValueError):
            validate_experiment_definition(bad_score)


class TestM2ExtractJsonRegression(unittest.TestCase):
    def test_valid_forms(self):
        self.assertEqual(experiment._extract_json('{"candidate_x": 1.0}'),
                         {"candidate_x": 1.0})
        self.assertEqual(
            experiment._extract_json(
                '```json\n{"candidate_x": 2.0}\n```'),
            {"candidate_x": 2.0})

    def test_invalid_forms_rejected(self):
        for bad in ["not json", "", "[]"]:
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    experiment._extract_json(bad)


class TestM2StorageIsolation(unittest.TestCase):
    def test_save_run_writes_only_to_temp(self):
        from pathlib import Path
        from prometheus import storage
        with tempfile.TemporaryDirectory() as tmp:
            path = storage.save_run(Path(tmp), {"project": "Prometheus",
                                                "mode": "m2_test", "x": 1})
            self.assertTrue(path.exists())
            self.assertTrue(str(path).startswith(tmp))
        # 临时目录删除后文件不再存在(未写入真实 runs/)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
