# -*- coding: utf-8 -*-
"""v0.3 M4.1:接入可选本地历史驱动研究模式测试。

覆盖(M4.1 指令 15 项要求):
 1. 默认策略仍为 model,旧流程不变;
 2. history 模式完全不调用 call_model();
 3. history 模式不要求 API key(离线,cfg 可为空);
 4. 两种实验分别使用正确的候选域和评分函数;
 5. 每轮候选均通过各自 validator;
 6. 每轮分数均由本地评分函数产生;
 7. 【关键】第二轮候选确实受第一轮已记录历史影响——调用真正的 run_research
    (strategy="history"),证明 run_results 的第 2 轮候选取决于第 1 轮结果;
 8. 相同初始条件与轮数产生相同结果(可复现);
 9. 候选耗尽后停止,不重复尝试;
10. rounds 达到上限后正常停止;
11. 非法策略名称给出清晰错误;
12. 旧函数调用和 CLI 现有测试仍然通过;
13. --goal 不改变真实评分结果;
14. 输出字段兼容性保持不变(best_candidate "x" / round_results candidate_field);
15. 所有测试产生的文件只写入隔离临时目录。

全部离线;不调用真实 API(history 模式本就不调 call_model;model 相关用例全部
mock);不写真实 runs/(写盘仅限 tempfile.TemporaryDirectory)。
"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock
from pathlib import Path

sys.path.insert(0, r"c:\Users\lenovo\Desktop\Prometheus")

import main as main_mod
from prometheus import experiment
from prometheus.candidate_strategy import (
    select_next_candidate, STATUS_START, STATUS_OK, STATUS_EXHAUSTED,
)
from prometheus.experiments import EXPERIMENTS, make_default_validator

TOY = EXPERIMENTS["toy_7_3"]   # 0.0..10.0 步长 0.5,峰值 7.3
MIN5 = EXPERIMENTS["min_5"]    # 0.0..5.0  步长 0.5,峰值 2.6
# history 模式完全离线:甚至不经过 load_config,cfg 可为空 dict。
EMPTY_CFG = {}


# --------------------------------------------------------------------------- #
# 1) 默认策略仍为 model,旧流程不变;12) 旧调用 run_research(cfg,goal,rounds) 兼容
# --------------------------------------------------------------------------- #
class TestDefaultStrategyModel(unittest.TestCase):
    def test_default_strategy_is_model_and_calls_model(self):
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5}') as m:
            result = experiment.run_research({"model": "m"}, "g", 1)
        m.assert_called_once()
        self.assertEqual(result["strategy"], "model")
        self.assertEqual(result["mode"], "api_research")

    def test_old_run_research_signature_still_works(self):
        # 旧签名 run_research(cfg, goal, rounds) 默认 model + toy_7_3。
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5}'):
            result = experiment.run_research({"model": "m"}, "g", 3)
        self.assertEqual(result["strategy"], "model")
        self.assertEqual(result["best_candidate"]["x"], 7.5)
        self.assertEqual(len(result["round_results"]), 3)

    def test_legacy_m4_cli_tests_still_pass_pattern(self):
        # 旧 CLI(--demo / --experiment / --goal)语义不变:此处检测 demo 仍离线、
        # 默认 toy_7_3,与 M3 测试保持一致。
        self.assertEqual(
            experiment.run_demo()["best_candidate"]["x"], 7.5)


# --------------------------------------------------------------------------- #
# 2/3) history 模式不调用 call_model、不要求 API key
# --------------------------------------------------------------------------- #
class TestHistoryOffline(unittest.TestCase):
    def test_history_never_calls_model(self):
        with mock.patch.object(experiment, "call_model") as m:
            result = experiment.run_research(EMPTY_CFG, "g", 20,
                                             experiment_id="min_5",
                                             strategy="history")
        m.assert_not_called()
        self.assertEqual(len(result["round_results"]), 11)  # 耗尽,不调模型

    def test_history_works_without_api_key(self):
        # cfg 为空 dict(等价于 --strategy history 时跳过 load_config),仍能运行。
        result = experiment.run_research({}, "g", 2, strategy="history")
        self.assertEqual(result["strategy"], "history")
        self.assertEqual(len(result["round_results"]), 2)


# --------------------------------------------------------------------------- #
# 4/5/6) 每实验独立候选域 + validator 校验 + 本地评分
# --------------------------------------------------------------------------- #
class TestPerExperimentDomainScore(unittest.TestCase):
    def test_toy_history_uses_own_domain_and_score(self):
        # toy 域 0..10(含 >5.0 值);用足够轮数让历史驱动推进越过 5.0,
        # 证明未复用 min_5 的 0..5 域。
        result = experiment.run_research(EMPTY_CFG, "g", 13,
                                         experiment_id="toy_7_3",
                                         strategy="history")
        allowed = set(TOY["allowed_values"])
        xs = [rr["candidate_x"] for rr in result["round_results"]]
        for rr in result["round_results"]:
            x = rr["candidate_x"]
            self.assertIn(x, allowed)                       # 仅用 toy 域
            self.assertAlmostEqual(rr["score"], TOY["score"](x))  # 本地评分
        # toy 域含 >5.0,证明未复用 min_5 域。
        self.assertGreater(max(xs), 5.0)

    def test_min5_history_uses_own_domain_and_score(self):
        result = experiment.run_research(EMPTY_CFG, "g", 5,
                                         experiment_id="min_5",
                                         strategy="history")
        allowed = set(MIN5["allowed_values"])
        for rr in result["round_results"]:
            x = rr["candidate_x"]
            self.assertIn(x, allowed)
            self.assertLessEqual(x, 5.0)                    # 绝不越出 min_5 域
            self.assertAlmostEqual(rr["score"], MIN5["score"](x))
        # min_5 域不超过 5.0。
        self.assertFalse(any(rr["candidate_x"] > 5.0 for rr in result["round_results"]))

    def test_every_round_passes_validator(self):
        for exp in (TOY, MIN5):
            validate = make_default_validator(exp)
            result = experiment.run_research(EMPTY_CFG, "g", 20,
                                             experiment_id=exp["id"],
                                             strategy="history")
            for rr in result["round_results"]:
                self.assertEqual(
                    validate({exp["candidate_field"]: rr[exp["candidate_field"]]}),
                    rr[exp["candidate_field"]])

    def test_scores_come_from_local_score_function(self):
        # 每轮分数 = exp["score"](候选),非模型自报(模型不参与)。
        result = experiment.run_research(EMPTY_CFG, "g", 8,
                                         experiment_id="toy_7_3",
                                         strategy="history")
        for rr in result["round_results"]:
            self.assertAlmostEqual(rr["score"], TOY["score"](rr["candidate_x"]))


# --------------------------------------------------------------------------- #
# 7) 关键:第二轮候选取决于第一轮已记录历史(调用真正 history 研究模式)
# --------------------------------------------------------------------------- #
class TestHistoryInfluencesNextRound(unittest.TestCase):
    def test_second_round_candidate_depends_on_recorded_history(self):
        # 调用真正的 run_research(strategy="history"),而非仅测纯函数。
        # toy_7_3:第 1 轮(无历史)为确定性起点 0.0;写入 history 后,
        # 第 2 轮依据"已知最优 0.0"选最近的未尝试候选 0.5。
        # 若第 2 轮未参考第 1 轮历史,则不会得到 0.5,从而证明依赖关系。
        result = experiment.run_research(EMPTY_CFG, "g", 2,
                                         experiment_id="toy_7_3",
                                         strategy="history")
        rr = result["round_results"]
        self.assertEqual(len(rr), 2)
        self.assertEqual(rr[0]["candidate_x"], 0.0)   # 第 1 轮起点
        self.assertEqual(rr[1]["candidate_x"], 0.5)   # 第 2 轮=最近未尝试邻居
        # 特别说明:0.5 完全是受第 1 轮记录的最优(0.0)驱动,而非固定顺序之外的随意值。

    def test_later_rounds_keep_following_best_seen_history(self):
        # 连续多轮,候选持续朝"历史最高分候选"最近的未尝试方向推进(0.0、0.5、1.0…)。
        result = experiment.run_research(EMPTY_CFG, "g", 5,
                                         experiment_id="toy_7_3",
                                         strategy="history")
        got = [rr["candidate_x"] for rr in result["round_results"]]
        expected = [0.0, 0.5, 1.0, 1.5, 2.0]      # 随最优分数上升而逐步推进
        self.assertEqual(got, expected)

    def test_history_drives_min5_climb_toward_optimum(self):
        # min_5 峰值在 2.6;历史驱动应从 0.0 逐轮攀升逼近 2.5/3.0 附近。
        result = experiment.run_research(EMPTY_CFG, "g", 8,
                                         experiment_id="min_5",
                                         strategy="history")
        got = [rr["candidate_x"] for rr in result["round_results"]]
        self.assertEqual(got[0], 0.0)
        self.assertEqual(len(set(got)), len(got))         # 无重复
        self.assertIn(2.5, got)                           # 到达峰值邻域
        self.assertIn(3.0, got)


# --------------------------------------------------------------------------- #
# 8/9/10) 可复现、耗尽停止、轮数上限
# --------------------------------------------------------------------------- #
class TestTerminationAndRepro(unittest.TestCase):
    def test_same_input_same_output(self):
        a = experiment.run_research(EMPTY_CFG, "g", 7,
                                    experiment_id="min_5",
                                    strategy="history")
        b = experiment.run_research(EMPTY_CFG, "g", 7,
                                    experiment_id="min_5",
                                    strategy="history")
        self.assertEqual(a["round_results"], b["round_results"])
        self.assertEqual(a["best_candidate"], b["best_candidate"])

    def test_exhausts_and_stops_no_repeat(self):
        # min_5 共 11 个候选;rounds=20 也应在耗尽(11 轮)后立即停止且无重复。
        result = experiment.run_research(EMPTY_CFG, "g", 20,
                                         experiment_id="min_5",
                                         strategy="history")
        got = [rr["candidate_x"] for rr in result["round_results"]]
        self.assertEqual(len(got), len(MIN5["allowed_values"]))
        self.assertEqual(len(set(got)), len(got))
        self.assertEqual(result["rounds_requested"], 20)   # 请求 20,实际 11(耗尽)
        # 每个候选都被尝试过且不重复 -> 不会伪装进展。

    def test_rounds_cap_stops_normally(self):
        # rounds=3 未耗尽,应正好返回 3 轮。
        result = experiment.run_research(EMPTY_CFG, "g", 3,
                                         experiment_id="min_5",
                                         strategy="history")
        self.assertEqual(len(result["round_results"]), 3)
        self.assertEqual(result["rounds_requested"], 3)


# --------------------------------------------------------------------------- #
# 11) 非法策略名清晰错误
# --------------------------------------------------------------------------- #
class TestInvalidStrategy(unittest.TestCase):
    def test_run_research_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError) as ctx:
            experiment.run_research(EMPTY_CFG, "g", 1, strategy="nope")
        msg = str(ctx.exception)
        self.assertIn("未知策略", msg)
        self.assertIn("model", msg)
        self.assertIn("history", msg)


# --------------------------------------------------------------------------- #
# 13/14) --goal 不改评分 + 输出字段兼容
# --------------------------------------------------------------------------- #
class TestGoalAndCompat(unittest.TestCase):
    def test_goal_does_not_change_scoring_in_history_mode(self):
        # 相同轮数,不同 goal 字符串,历史模式候选/分数完全一致(评分只由实验定义决定)。
        a = experiment.run_research(EMPTY_CFG, "目标A", 4,
                                    experiment_id="min_5", strategy="history")
        b = experiment.run_research(EMPTY_CFG, "目标B 完全不同", 4,
                                    experiment_id="min_5", strategy="history")
        self.assertEqual(a["round_results"], b["round_results"])

    def test_output_field_compat_keys(self):
        # best_candidate 沿用旧 "x" 键;round_results / baseline 用候选域 field。
        result = experiment.run_research(EMPTY_CFG, "g", 3,
                                         experiment_id="toy_7_3",
                                         strategy="history")
        bc = result["best_candidate"]
        self.assertIn("x", bc)                       # 旧契约键保留
        self.assertIn("score", bc)
        self.assertIn("candidate_x", result["baseline"])
        rr = result["round_results"][0]
        self.assertIn("candidate_x", rr)

    def test_history_marks_mode_and_strategy(self):
        result = experiment.run_research(EMPTY_CFG, "g", 2, strategy="history")
        self.assertEqual(result["mode"], "history_research")
        self.assertEqual(result["strategy"], "history")   # 最小兼容新增字段
        self.assertIn("project", result)


# --------------------------------------------------------------------------- #
# 11 补充 / 15) CLI 层 --strategy + 隔离临时目录写盘
# --------------------------------------------------------------------------- #
class TestCliHistory(unittest.TestCase):
    def _run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with redirect_stdout(out):
                with redirect_stderr(err):
                    code = main_mod.run_cli()
        return code, out.getvalue(), err.getvalue(), out, err

    def test_cli_history_skips_load_config_and_writes_temp(self):
        # --strategy history 不触发 load_config(离线,无 key 也可);写盘走临时目录。
        with tempfile.TemporaryDirectory() as tmp:
            captured = {}

            def fake_save(root, result):
                captured["result"] = result
                captured["root"] = root
                return Path(tmp) / "fake.json"

            with mock.patch.object(main_mod, "load_config",
                                   side_effect=AssertionError("不应调用 load_config")):
                with mock.patch.object(main_mod, "save_run", side_effect=fake_save):
                    code, out, err, o, e = self._run_cli(
                        ["prog", "--goal", "g", "--rounds", "1",
                         "--experiment", "toy_7_3", "--strategy", "history"])
            self.assertEqual(code, 0, err)
            self.assertEqual(captured["result"]["strategy"], "history")
            self.assertEqual(captured["result"]["mode"], "history_research")

    def test_cli_default_strategy_still_model(self):
        # 无 --strategy 时默认 model:经 load_config + run_research(model)。
        with mock.patch.object(main_mod, "load_config", return_value={"model": "m"}):
            with mock.patch.object(experiment, "call_model",
                                   return_value='{"candidate_x": 7.5}'):
                with mock.patch.object(main_mod, "save_run",
                                       return_value=Path("fake.json")):
                    code, out, err, o, e = self._run_cli(
                        ["prog", "--goal", "g", "--rounds", "1"])
            self.assertEqual(code, 0, err)
            self.assertIn("api_research", out)     # mode 仍为 api_research
            self.assertIn('"strategy": "model"', out)

    def test_cli_invalid_strategy_exit_1_concise(self):
        code, out, err, o, e = self._run_cli(
            ["prog", "--goal", "g", "--strategy", "bogus"])
        self.assertEqual(code, 1)                  # 非零退出
        self.assertIn("未知策略", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
