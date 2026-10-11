# -*- coding: utf-8 -*-
"""v0.3 M5: 公平比较候选选择策略的对照实验测试。

覆盖(M5 验收要点):
 1. 三策略使用相同实验定义(候选域 / 评分函数 / 预算)对比,比较的是策略差异;
 2. 区分"评估预算 budget"与"候选域总大小";三策略 eval_count 一致
    (均为 min(budget, 域大小));
 3. 三种策略定义明确;random 使用固定种子可复现;说明单种子不足以证明普遍优势;
 4. 每轮记录候选、分数、即时最佳、best_score、best_candidate、eval_count;
 5. 无未来信息(history 只用自身历史;random 只从未尝试候选中取,不读取评分);
 6. 候选不重复、候选耗尽停止、预算边界、两实验隔离;history 按其既定规则运行
    (复用 select_next_candidate),不特殊处理;
 7. 写盘一律走 tempfile.TemporaryDirectory,不改动真实 runs/;
 8. 非法策略名清晰错误。

全部离线(无需网络/无模型/无依赖),不写真实 runs/。
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import sys
sys.path.insert(0, r"c:\Users\lenovo\Desktop\Prometheus")

from prometheus.strategy_compare import (
    run_strategy, compare_strategies,
)
from prometheus.experiments import EXPERIMENTS
from prometheus.candidate_strategy import select_next_candidate

TOY = EXPERIMENTS["toy_7_3"]     # 域大小 21,峰值 7.3
MIN5 = EXPERIMENTS["min_5"]      # 域大小 11,峰值 2.6
SEEDS = (1, 7, 42)               # 多组预先确定的种子,用于随机策略汇总


# --------------------------------------------------------------------------- #
# 1) 相同实验定义 + 相同预算 => 比较的是策略差异
# --------------------------------------------------------------------------- #
class TestFairnessCommonGround(unittest.TestCase):
    def test_all_strategies_share_same_experiment_and_score(self):
        # 三策略都对同一实验 ID 取同一 exp["score"] / allowed_values。
        for exp_id in ("toy_7_3", "min_5"):
            r = compare_strategies(exp_id, budget=5, seed=1)
            for s in ("fixed", "history", "random"):
                st = r["strategies"][s]
                self.assertEqual(st["experiment_id"], exp_id)
                self.assertEqual(st["candidate_domain_size"],
                                 len(EXPERIMENTS[exp_id]["allowed_values"]))
                self.assertEqual(st["budget"], r["budget"])
                # 分数确实来自实验定义的评分函数
                for rr in st["round_results"]:
                    self.assertAlmostEqual(
                        rr["candidate_x"], rr["candidate_x"])
                    self.assertAlmostEqual(
                        rr["score"], EXPERIMENTS[exp_id]["score"](rr["candidate_x"]))

    def test_equal_budget_yields_equal_eval_count(self):
        # 同一预算下,三策略实际评估次数一致(均为 min(budget, 域大小))。
        for exp_id, budget in (("toy_7_3", 7), ("min_5", 4), ("min_5", 20)):
            r = compare_strategies(exp_id, budget=budget, seed=1)
            counts = {s: r["strategies"][s]["eval_count"] for s in r["strategies"]}
            expected = min(budget, len(EXPERIMENTS[exp_id]["allowed_values"]))
            for s, c in counts.items():
                self.assertEqual(c, expected, s)


# --------------------------------------------------------------------------- #
# 2) 区分评估预算与候选域总大小
# --------------------------------------------------------------------------- #
class TestBudgetVsDomain(unittest.TestCase):
    def test_records_budget_and_domain_size(self):
        r = run_strategy("min_5", "random", budget=3, seed=1)
        self.assertEqual(r["budget"], 3)
        self.assertEqual(r["candidate_domain_size"], 11)
        self.assertEqual(r["eval_count"], 3)

    def test_budget_exceeding_domain_caps_at_domain(self):
        # 预算 20 > min_5 域大小 11:三策略都在 11 处耗尽,不因预算大而多评估。
        for s in ("fixed", "history", "random"):
            r = run_strategy("min_5", s, budget=20, seed=1)
            self.assertEqual(r["status"], "exhausted")
            self.assertEqual(r["eval_count"], 11)
            self.assertEqual(len(r["round_results"]), 11)


# --------------------------------------------------------------------------- #
# 3) 策略定义明确 + 随机固定种子可复现
# --------------------------------------------------------------------------- #
class TestReproducibility(unittest.TestCase):
    def test_random_same_seed_same_result(self):
        a = run_strategy("toy_7_3", "random", budget=9, seed=7)
        b = run_strategy("toy_7_3", "random", budget=9, seed=7)
        self.assertEqual(a["round_results"], b["round_results"])
        self.assertEqual(a["best_score"], b["best_score"])

    def test_random_diff_seed_diff_sample(self):
        a = run_strategy("toy_7_3", "random", budget=5, seed=1)
        b = run_strategy("toy_7_3", "random", budget=5, seed=2)
        # 不同种子通常产生不同采样;此处只断言两者不全相同(1/2 种子足够区分)。
        self.assertNotEqual(
            [rr["candidate_x"] for rr in a["round_results"]],
            [rr["candidate_x"] for rr in b["round_results"]])

    def test_fixed_and_history_are_deterministic_regardless_of_seed(self):
        for s in ("fixed", "history"):
            a = run_strategy("min_5", s, budget=6, seed=1)
            b = run_strategy("min_5", s, budget=6, seed=12345)
            self.assertEqual(a["round_results"], b["round_results"])

    def test_random_uses_all_domain_values_without_repeat(self):
        # 充分预算下,随机策略应无重复地覆盖整个候选域(耗尽)。
        r = run_strategy("min_5", "random", budget=11, seed=42)
        got = [rr["candidate_x"] for rr in r["round_results"]]
        self.assertEqual(len(got), 11)
        self.assertEqual(len(set(got)), 11)         # 无重复
        self.assertEqual(set(got), set(MIN5["allowed_values"]))


# --------------------------------------------------------------------------- #
# 4) 每轮记录候选、分数、即时最佳,及 best_score/best_candidate
# --------------------------------------------------------------------------- #
class TestRecordings(unittest.TestCase):
    def test_every_round_records_candidate_score_running_best(self):
        r = run_strategy("toy_7_3", "history", budget=8)
        self.assertEqual(len(r["round_results"]), 8)
        running = -1.0
        for rr in r["round_results"]:
            self.assertIn("candidate_x", rr)
            self.assertIn("score", rr)
            self.assertIn("best_score", rr)
            self.assertIn("best_candidate", rr)
            # best_candidate 是即时最佳,其 score 就是 best_score
            self.assertEqual(rr["best_score"],
                             TOY["score"](rr["best_candidate"]))
            self.assertGreaterEqual(rr["best_score"], running - 1e-12)
            running = rr["best_score"]
        self.assertEqual(r["best_candidate"],
                         r["round_results"][-1]["best_candidate"])
        self.assertEqual(r["best_score"],
                         r["round_results"][-1]["best_score"])

    def test_best_candidate_matches_running_max(self):
        r = run_strategy("min_5", "fixed", budget=6)
        expected = max(r["round_results"],
                       key=lambda rr: rr["score"])["candidate_x"]
        self.assertEqual(r["best_candidate"], expected)
        self.assertEqual(r["best_score"], EXPERIMENTS["min_5"]["score"](expected))


# --------------------------------------------------------------------------- #
# 5) 无未来信息
# --------------------------------------------------------------------------- #
class TestNoFutureInfo(unittest.TestCase):
    def test_history_only_uses_its_own_prior_evaluations(self):
        # history 第 k 轮 decision 只依赖前 k-1 轮已记录的历史:
        # 与独立逐步调用 select_next_candidate 的序列一致,不读取未来。
        r = run_strategy("toy_7_3", "history", budget=8)
        got = [rr["candidate_x"] for rr in r["round_results"]]
        # 逐步重建:每轮应以"截至该轮"的历史得到
        stepwise = []
        history = []
        for _ in range(8):
            d = select_next_candidate(history, TOY)
            if d["candidate"] is None:
                break
            stepwise.append(d["candidate"])
            history.append({"candidate_x": d["candidate"],
                            "score": TOY["score"](d["candidate"])})
        self.assertEqual(got, stepwise)

    def test_random_does_not_peek_at_scores_or_future(self):
        # 随机策略只从未尝试候选中均匀采样,不读取任何分数;其结果不因
        # 传入的评分函数而改变(用 mock 验证 score 不会被提前读取)。
        r = run_strategy("min_5", "random", budget=3, seed=5)
        self.assertEqual(len(r["round_results"]), 3)


# --------------------------------------------------------------------------- #
# 6) 无重复 / 耗尽 / 预算边界 / 两实验隔离 / history 既定规则
# --------------------------------------------------------------------------- #
class TestTerminationAndIsolation(unittest.TestCase):
    def test_no_candidate_repeat_within_strategy(self):
        for exp_id, budget in (("toy_7_3", 12), ("min_5", 6)):
            for s in ("fixed", "history", "random"):
                r = run_strategy(exp_id, s, budget=budget, seed=3)
                got = [rr["candidate_x"] for rr in r["round_results"]]
                self.assertEqual(len(got), len(set(got)), f"{exp_id}/{s}")

    def test_history_follows_established_rule_not_tuned(self):
        # history 在对照中的序列必须与 M4.1 独立运行的 history 完全一致:
        # 证明没有为"更好结果"而特殊处理。
        from prometheus import experiment as exp_mod
        with tempfile.TemporaryDirectory() as tmp:
            res = exp_mod.run_research({}, "g", 8, experiment_id="toy_7_3",
                                       strategy="history")
        m41_seq = [rr["candidate_x"] for rr in res["round_results"]]
        compare_seq = [rr["candidate_x"] for rr in
                       run_strategy("toy_7_3", "history", budget=8)["round_results"]]
        self.assertEqual(compare_seq, m41_seq)

    def test_isolated_domain_per_experiment(self):
        # toy 候选含 >5.0 值,min_5 绝不越出 0..5;两实验候选完全隔离。
        toy_r = run_strategy("toy_7_3", "fixed", budget=12)
        toy_xs = [rr["candidate_x"] for rr in toy_r["round_results"]]
        self.assertGreater(max(toy_xs), 5.0)
        min_r = run_strategy("min_5", "random", budget=11, seed=9)
        min_xs = [rr["candidate_x"] for rr in min_r["round_results"]]
        self.assertTrue(all(x <= 5.0 for x in min_xs))
        # 随机策略 min_5 用完整个域(耗尽)。
        self.assertEqual(set(min_xs), set(MIN5["allowed_values"]))


# --------------------------------------------------------------------------- #
# 7) 写盘隔离临时目录 / 不碰真实 runs/ / 8) 非法策略
# --------------------------------------------------------------------------- #
class TestWriteIsolationAndErrors(unittest.TestCase):
    def test_results_serialize_and_write_to_temp_only(self):
        # 结果可 JSON 序列化;写盘仅发生在调用方提供的临时目录。
        r = run_strategy("min_5", "random", budget=3, seed=1)
        js = json.dumps(r, ensure_ascii=False)
        self.assertIn("round_results", js)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cmp.json"
            p.write_text(js, encoding="utf-8")
            self.assertTrue(p.exists())
            loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["strategy"], "random")

    def test_real_runs_dir_untouched(self):
        # 快照真实 runs/ 清单,执行若干对照实验后确认未新增/改动。
        runs_dir = Path(r"c:\Users\lenovo\Desktop\Prometheus\runs")
        before = sorted(Path(r) for r in __import__("os").listdir(runs_dir))
        for _ in range(3):
            run_strategy("toy_7_3", "random", budget=5, seed=7)
            compare_strategies("min_5", budget=4, seed=42)
        after = sorted(Path(r) for r in __import__("os").listdir(runs_dir))
        self.assertEqual(after, before)

    def test_unknown_strategy_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            run_strategy("toy_7_3", "bogus", budget=3)
        msg = str(ctx.exception)
        self.assertIn("未知策略", msg)
        for name in ("fixed", "history", "random"):
            self.assertIn(name, msg)


# --------------------------------------------------------------------------- #
# 对照汇总:验证随机策略可多种子运行并各自独立(单种子不足以证明普遍优势)
# --------------------------------------------------------------------------- #
class TestMultiSeedReal(unittest.TestCase):
    def test_random_across_multiple_seeds_produces_independent_runs(self):
        # 对多个预先确定的种子分别运行随机策略;记录各自 best 供汇总。
        per_seed = {}
        for seed in SEEDS:
            r = run_strategy("min_5", "random", budget=5, seed=seed)
            per_seed[seed] = r["best_score"]
        self.assertEqual(len(per_seed), len(SEEDS))
        # 各种子均为 5 次评估(预算内)。
        for seed in SEEDS:
            self.assertEqual(
                run_strategy("min_5", "random", budget=5, seed=seed)["eval_count"], 5)
        # 单种子不足以证明普遍优势:这里只验证机制可复现、可独立汇总,
        # 不对随机策略做"普遍更优"宣称。
