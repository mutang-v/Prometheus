# -*- coding: utf-8 -*-
"""v0.3 M3.1:baseline 通用化审查测试。

覆盖:
  - toy_7_3 与 min_5 各自 baseline 候选的合法性(值 2.0 在 allowed_values 内并通过校验器);
  - baseline 候选字段与所选实验的 candidate_field 一致;
  - baseline 分数由所选实验的评分函数计算(两实验不同值均正确);
  - history / best_candidate / round_results 字段与语义一致;
  - 旧默认实验结果与旧接口兼容。
全部写盘走临时目录;不调用真实 API(全部 mock)。
"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

sys.path.insert(0, r"c:\Users\lenovo\Desktop\Prometheus")

import main as main_mod
from prometheus import experiment
from prometheus.experiments import (
    EXPERIMENTS, get_experiment, make_default_validator,
)

CFG = {"model": "m", "base_url": "http://x", "api_key": "k"}
MIN5 = EXPERIMENTS["min_5"]
TOY = EXPERIMENTS["toy_7_3"]

BASELINE_X = 2.0


def _research(reply, experiment_id="toy_7_3", goal="g", rounds=1):
    # reply 可为单个字符串(每轮相同)或与 rounds 等长的字符串列表。
    side_effect = reply if isinstance(reply, list) else [reply] * rounds
    with mock.patch.object(experiment, "call_model",
                           side_effect=side_effect) as m:
        return experiment.run_research(CFG, goal, rounds,
                                       experiment_id=experiment_id), m


class TestBaselineLegality(unittest.TestCase):
    def test_toy_baseline_value_in_allowed_values(self):
        self.assertIn(BASELINE_X, TOY["allowed_values"])

    def test_min5_baseline_value_in_allowed_values(self):
        self.assertIn(BASELINE_X, MIN5["allowed_values"])

    def test_toy_baseline_passes_validator(self):
        # baseline 值 2.0 作为候选应通过所选实验的校验器。
        validate = make_default_validator(TOY)
        self.assertEqual(validate({TOY["candidate_field"]: BASELINE_X}),
                         BASELINE_X)

    def test_min5_baseline_passes_validator(self):
        validate = make_default_validator(MIN5)
        self.assertEqual(validate({MIN5["candidate_field"]: BASELINE_X}),
                         BASELINE_X)


class TestBaselineFieldConsistency(unittest.TestCase):
    def test_demo_toy_baseline_uses_candidate_field(self):
        result = experiment.run_demo(experiment=TOY)
        self.assertIn(TOY["candidate_field"], result["baseline"])
        self.assertEqual(result["baseline"][TOY["candidate_field"]], BASELINE_X)

    def test_demo_min5_baseline_uses_candidate_field(self):
        result = experiment.run_demo(experiment=MIN5)
        self.assertIn(MIN5["candidate_field"], result["baseline"])
        self.assertEqual(result["baseline"][MIN5["candidate_field"]], BASELINE_X)

    def test_research_toy_baseline_uses_candidate_field(self):
        result, _ = _research('{"candidate_x": 7.5}')
        self.assertIn(TOY["candidate_field"], result["baseline"])
        self.assertEqual(result["baseline"][TOY["candidate_field"]], BASELINE_X)

    def test_research_min5_baseline_uses_candidate_field(self):
        result, _ = _research('{"candidate_x": 2.5}', experiment_id="min_5")
        self.assertIn(MIN5["candidate_field"], result["baseline"])
        self.assertEqual(result["baseline"][MIN5["candidate_field"]], BASELINE_X)


class TestBaselineScoreByOwnExperiment(unittest.TestCase):
    def test_demo_toy_baseline_score_by_toy_score(self):
        result = experiment.run_demo(experiment=TOY)
        # toy 评分 1/(1+(2.0-7.3)^2)=1/(1+(-5.3)^2)
        self.assertAlmostEqual(result["baseline"]["score"],
                               TOY["score"](BASELINE_X))
        self.assertAlmostEqual(result["baseline"]["score"],
                               1 / (1 + (2.0 - 7.3) ** 2))

    def test_demo_min5_baseline_score_by_min5_score(self):
        result = experiment.run_demo(experiment=MIN5)
        # min_5 评分 1/(1+(2.0-2.6)^2)=1/(1+0.6^2)
        self.assertAlmostEqual(result["baseline"]["score"],
                               MIN5["score"](BASELINE_X))
        self.assertAlmostEqual(result["baseline"]["score"],
                               1 / (1 + 0.6 ** 2))
        # 两实验 baseline 分数必须不同(评分函数不同)。
        self.assertNotAlmostEqual(result["baseline"]["score"],
                                  TOY["score"](BASELINE_X))

    def test_research_min5_baseline_score_by_min5_score(self):
        result, _ = _research('{"candidate_x": 2.5}', experiment_id="min_5")
        self.assertAlmostEqual(result["baseline"]["score"],
                               MIN5["score"](BASELINE_X))
        self.assertAlmostEqual(result["baseline"]["score"], 1 / (1 + 0.6 ** 2))


class TestHistoryBestRoundConsistency(unittest.TestCase):
    def test_research_toy_rounds_use_candidate_field_and_program_score(self):
        replies = ['{"candidate_x": 7.5}', '{"candidate_x": 7.0}']
        result, m = _research(replies, rounds=2)
        for rr in result["round_results"]:
            self.assertIn(TOY["candidate_field"], rr)
            self.assertAlmostEqual(rr["score"],
                                   TOY["score"](rr[TOY["candidate_field"]]))
        # 最佳候选来自 history 的最高分(公开键 x)。
        self.assertEqual(result["best_candidate"]["x"], 7.5)
        # improved_over_baseline 语义:best 分数 > baseline 分数。
        self.assertEqual(result["improved_over_baseline"],
                         result["best_candidate"]["score"] >
                         result["baseline"]["score"])

    def test_research_min5_rounds_use_candidate_field(self):
        result, _ = _research('{"candidate_x": 2.5}', experiment_id="min_5")
        rr = result["round_results"][0]
        self.assertIn(MIN5["candidate_field"], rr)
        self.assertEqual(rr[MIN5["candidate_field"]], 2.5)
        self.assertAlmostEqual(rr["score"], MIN5["score"](2.5))
        self.assertEqual(result["best_candidate"]["x"], 2.5)


class TestLegacyCompat(unittest.TestCase):
    def test_old_run_research_call_default_toy(self):
        result, _ = _research('{"candidate_x": 7.5}')
        self.assertEqual(result["best_candidate"]["x"], 7.5)
        self.assertAlmostEqual(result["best_candidate"]["score"],
                               1 / (1 + 0.2 ** 2))
        self.assertIn(TOY["candidate_field"], result["baseline"])

    def test_old_cli_demo_still_toy(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = {}

            def fake_save(root, result):
                captured["result"] = result
                return tmp

            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(main_mod, "save_run", side_effect=fake_save):
                with mock.patch.object(sys, "argv", ["prog", "--demo"]):
                    with redirect_stdout(out):
                        with redirect_stderr(err):
                            code = main_mod.run_cli()
            self.assertEqual(code, 0, err)
            self.assertEqual(captured["result"]["best_candidate"]["x"], 7.5)
            self.assertAlmostEqual(captured["result"]["best_candidate"]["score"],
                                   1 / (1 + 0.2 ** 2))
            self.assertIn(TOY["candidate_field"],
                          captured["result"]["baseline"])


class TestIsolation(unittest.TestCase):
    def test_no_real_runs_written(self):
        # 真实隔离验证:将 main.ROOT 重定向到临时目录并执行真实 --demo 写盘路径,
        # 证明运行记录只写入临时目录;再比较项目真实 runs/ 的文件名集合,确认前后一致。
        from pathlib import Path
        real_runs = Path(__file__).resolve().parent.parent / "runs"
        before = (set(p.name for p in real_runs.iterdir())
                  if real_runs.is_dir() else set())
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(main_mod, "ROOT", tmp_root):
                with mock.patch.object(sys, "argv", ["prog", "--demo"]):
                    with redirect_stdout(out):
                        with redirect_stderr(err):
                            code = main_mod.run_cli()
            self.assertEqual(code, 0, err)
            # 写盘确实发生,且被重定向到临时目录 runs/(出现 local_demo 记录)。
            written = list((tmp_root / "runs").glob("*_local_demo.json"))
            self.assertEqual(len(written), 1,
                             "run_cli --demo 未向临时目录写入运行记录")
        after = (set(p.name for p in real_runs.iterdir())
                 if real_runs.is_dir() else set())
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
