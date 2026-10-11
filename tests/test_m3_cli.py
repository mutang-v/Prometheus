# -*- coding: utf-8 -*-
"""v0.3 M3:第二实验(min_5)端到端验证 与 CLI(--experiment)挂接测试。

覆盖:
  - min_5 注册、候选域、评分;toy_7_3 定义不改变;
  - demo 离线用 min_5 自身候选域+评分;默认 demo 仍为 toy_7_3;
  - 研究模式读取各自实验定义,非法候选被拒、伪造 score 被忽略并由程序重算;
  - 未知实验 ID 简洁中文错误 + 非零退出码,不打印冗长 traceback;
  - 旧 CLI 调用、旧 run_research(cfg,goal,rounds) 与默认实验兼容;
  - --goal 不改变程序评分;--experiment min_5 选择正确实验;
  - 全部写盘测试使用临时目录,不触碰真实 runs/。
不调用真实 API(全部 mock)。
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
    EXPERIMENTS, get_experiment, validate_experiment_definition,
)

CFG = {"model": "m", "base_url": "http://x", "api_key": "k"}
MIN5 = EXPERIMENTS["min_5"]
TOY = EXPERIMENTS["toy_7_3"]


class TestMin5Registration(unittest.TestCase):
    def test_min5_registered_with_consistent_id(self):
        self.assertIn("min_5", EXPERIMENTS)
        self.assertEqual(MIN5["id"], "min_5")

    def test_two_experiments_queryable(self):
        self.assertIs(get_experiment("toy_7_3"), TOY)
        self.assertIs(get_experiment("min_5"), MIN5)

    def test_min5_definition_complete(self):
        # 合法定义应通过换检。
        self.assertIs(validate_experiment_definition(MIN5), MIN5)

    def test_min5_allowed_values(self):
        self.assertEqual(MIN5["allowed_values"],
                         [i / 2 for i in range(0, 11)])   # 0.0..5.0, 0.5

    def test_min5_scoring(self):
        self.assertAlmostEqual(MIN5["score"](2.6), 1.0)
        self.assertAlmostEqual(MIN5["score"](2.5), 1 / (1 + 0.1 ** 2))

    def test_min5_scoring_hint_matches_rule(self):
        self.assertIn("2.6", MIN5["scoring_hint"])
        self.assertIn("1/(1+(x-2.6)^2)", MIN5["scoring_hint"])

    def test_toy7_3_unchanged(self):
        self.assertEqual(TOY["allowed_values"], [i / 2 for i in range(0, 21)])
        self.assertAlmostEqual(TOY["score"](7.3), 1.0)
        self.assertAlmostEqual(TOY["score"](7.5), 1 / (1 + 0.2 ** 2))
        self.assertEqual(TOY["candidate_field"], "candidate_x")


class TestDemoMin5(unittest.TestCase):
    def test_demo_min5_offline_uses_own_domain_and_score(self):
        # 无 API:直接调用 run_demo(experiment=min_5)。
        result = experiment.run_demo(experiment=MIN5)
        self.assertEqual(result["best_candidate"]["x"], 2.5)
        self.assertAlmostEqual(result["best_candidate"]["score"], 1 / (1 + 0.1 ** 2))
        # objective 反映 min_5 实验,而非仅改字面文字
        self.assertIn("2.6", result["objective"])

    def test_demo_min5_candidates_within_domain(self):
        result = experiment.run_demo(experiment=MIN5)
        allowed = set(MIN5["allowed_values"])
        for c in result["candidates"]:
            self.assertIn(c["x"], allowed)          # 均在 0~5、0.5 步长候选集合内
            self.assertLessEqual(c["x"], 5.0)
            self.assertGreaterEqual(c["x"], 0.0)

    def test_demo_default_still_toy7_3(self):
        result = experiment.run_demo()
        self.assertEqual(result["best_candidate"]["x"], 7.5)
        self.assertAlmostEqual(result["best_candidate"]["score"], 1 / (1 + 0.2 ** 2))


class TestResearchMin5(unittest.TestCase):
    def _research(self, reply, experiment_id="min_5", goal="goal"):
        with mock.patch.object(experiment, "call_model", return_value=reply) as m:
            result = experiment.run_research(CFG, goal, 1,
                                             experiment_id=experiment_id)
        return result, m

    def test_min5_accepts_own_domain_value(self):
        result, _ = self._research('{"candidate_x": 2.5}')
        rr = result["round_results"][0]
        self.assertEqual(rr["candidate_x"], 2.5)
        self.assertAlmostEqual(rr["score"], 1 / (1 + 0.1 ** 2))

    def test_min5_rejects_value_valid_in_toy_but_out_of_min5(self):
        # 7.5 在 toy_7_3 域内,但在 min_5(0..5) 域外,必须被拒。
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5}'):
            with self.assertRaises(ValueError):
                experiment.run_research(CFG, "g", 1, experiment_id="min_5")

    def test_min5_ignores_forged_score(self):
        # 模型伪造 score=0.99,程序必须忽略并按自身评分函数重算。
        result, _ = self._research('{"candidate_x": 3.0, "score": 0.99}')
        rr = result["round_results"][0]
        self.assertAlmostEqual(rr["score"], 1 / (1 + 0.4 ** 2))   # ~0.8616
        self.assertNotAlmostEqual(rr["score"], 0.99)

    def test_toy_research_ignores_forged_score(self):
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5, "score": 0.1}'):
            result = experiment.run_research(CFG, "g", 1,
                                             experiment_id="toy_7_3")
        rr = result["round_results"][0]
        self.assertAlmostEqual(rr["score"], 1 / (1 + 0.2 ** 2))

    def test_goal_does_not_change_scoring(self):
        # 相同候选,不同 goal 字符串,程序评分一致(评分只由实验定义决定)。
        for goal in ["目标A", "目标B 完全不同"]:
            with mock.patch.object(experiment, "call_model",
                                   return_value='{"candidate_x": 2.5}'):
                result = experiment.run_research(CFG, goal, 1,
                                                 experiment_id="min_5")
            self.assertAlmostEqual(result["round_results"][0]["score"],
                                   1 / (1 + 0.1 ** 2))

    def test_old_run_research_call_compat(self):
        # 旧签名 run_research(cfg, goal, rounds) 默认 toy_7_3。
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5}'):
            result = experiment.run_research(CFG, "g", 1)
        self.assertEqual(result["best_candidate"]["x"], 7.5)
        self.assertAlmostEqual(result["best_candidate"]["score"], 1 / (1 + 0.2 ** 2))


class TestSelectExperiment(unittest.TestCase):
    def test_known_ids_ok(self):
        self.assertIs(main_mod.select_experiment("toy_7_3"), TOY)
        self.assertIs(main_mod.select_experiment("min_5"), MIN5)

    def test_unknown_id_error_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            main_mod.select_experiment("nope")
        msg = str(ctx.exception)
        self.assertIn("未知实验", msg)
        self.assertIn("nope", msg)
        self.assertIn("toy_7_3", msg)
        self.assertIn("min_5", msg)


class TestCliExperiment(unittest.TestCase):
    def _run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with redirect_stdout(out):
                with redirect_stderr(err):
                    code = main_mod.run_cli()
        return code, out.getvalue(), err.getvalue()

    def test_cli_unknown_experiment_exit_1_concise(self):
        # --goal 分支先 select_experiment 抛错,不会触发 load_config/API。
        code, out, err = self._run_cli(
            ["prog", "--goal", "g", "--experiment", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("未知实验", err)
        self.assertNotIn("Traceback", err)

    def test_cli_demo_unknown_experiment_exit_1(self):
        code, out, err = self._run_cli(
            ["prog", "--demo", "--experiment", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("未知实验", err)

    def test_cli_demo_min5_uses_own_experiment_write_to_temp(self):
        # 写临时目录(不污染真实 runs/),并断言结构化结果确为 min_5。
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            captured = {}

            def fake_save(root, result):
                captured["result"] = result
                return Path(tmp) / "fake.json"

            with mock.patch.object(main_mod, "save_run", side_effect=fake_save):
                code, out, err = self._run_cli(
                    ["prog", "--demo", "--experiment", "min_5"])
            self.assertEqual(code, 0, err)
            self.assertEqual(captured["result"]["objective"], MIN5["description"])
            self.assertEqual(captured["result"]["best_candidate"]["x"], 2.5)
            self.assertAlmostEqual(captured["result"]["best_candidate"]["score"],
                                   1 / (1 + 0.1 ** 2))

    def test_cli_demo_default_write_to_temp(self):
        # 旧 CLI 调用(无 --experiment)仍为 toy_7_3。
        with tempfile.TemporaryDirectory() as tmp:
            captured = {}

            def fake_save(root, result):
                captured["result"] = result
                return tmp

            with mock.patch.object(main_mod, "save_run", side_effect=fake_save):
                code, out, err = self._run_cli(["prog", "--demo"])
            self.assertEqual(code, 0, err)
            self.assertEqual(captured["result"]["best_candidate"]["x"], 7.5)
            self.assertAlmostEqual(captured["result"]["best_candidate"]["score"],
                                   1 / (1 + 0.2 ** 2))


class TestIsolation(unittest.TestCase):
    def test_no_real_runs_written_by_m3_tests(self):
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
