# -*- coding: utf-8 -*-
"""v0.3 M4:历史经验驱动候选选择策略测试。

覆盖:
  - 空历史 / 正常历史 / 重复候选 / 非法候选 / 缺失候选字段 / 伪造 score;
  - 候选全部耗尽;
  - 两个实验(toy_7_3 与 min_5)候选域不同时均行为正确;
  - 相同输入产生相同输出(可复现,不依赖时间/网络/随机数);
  - 固定基线 / 简单确定性 / 新策略三种策略的对照(合法性/重复率/可复现);
  - 旧接口兼容(默认研究流程仍由模型驱动,未被破坏)。
策略为纯函数,无写盘。本阶段不调用真实 API(全部 mock;策略本就不调用 API)。
"""
import sys
import unittest
from unittest import mock

sys.path.insert(0, r"c:\Users\lenovo\Desktop\Prometheus")

from prometheus import candidate_strategy as strat
from prometheus import experiment
from prometheus.candidate_strategy import STATUS_START, STATUS_OK, STATUS_EXHAUSTED
from prometheus.experiments import EXPERIMENTS, make_default_validator

TOY = EXPERIMENTS["toy_7_3"]      # 0.0..10.0 步长 0.5,最接近 7.3
MIN5 = EXPERIMENTS["min_5"]       # 0.0..5.0  步长 0.5,最接近 2.6
CFG = {"model": "m", "base_url": "http://x", "api_key": "k"}


def safe(history, exp):
    """返回候选值;若耗尽则返回 None。供对照与断言语义统一。"""
    return strat.select_next_candidate(history, exp)["candidate"]


class TestStrategyBasicScenarios(unittest.TestCase):
    def test_empty_history_returns_deterministic_start(self):
        res = strat.select_next_candidate([], TOY)
        self.assertEqual(res["status"], STATUS_START)
        self.assertEqual(res["candidate"], TOY["allowed_values"][0])  # 0.0
        self.assertEqual(res["tried"], [])

    def test_normal_history_advances_toward_known_best(self):
        # toy:历史最优为 4.5,最近未尝试邻居为 5.0。
        history = [
            {"candidate_x": 4.0, "score": TOY["score"](4.0)},
            {"candidate_x": 4.5, "score": TOY["score"](4.5)},
        ]
        res = strat.select_next_candidate(history, TOY)
        self.assertEqual(res["status"], STATUS_OK)
        self.assertEqual(res["candidate"], 5.0)
        self.assertEqual(res["best_seen"], 4.5)
        self.assertEqual(res["tried"], [4.0, 4.5])

    def test_prefers_untried_over_repeat(self):
        # 历史已含 8.0、7.5,新策略不应重复它们,而应取未尝试的最近邻居 7.0。
        history = [
            {"candidate_x": 8.0, "score": TOY["score"](8.0)},
            {"candidate_x": 7.5, "score": TOY["score"](7.5)},
        ]
        res = strat.select_next_candidate(history, TOY)
        self.assertEqual(res["status"], STATUS_OK)
        self.assertEqual(res["candidate"], 7.0)   # 与 7.5 最近
        self.assertNotIn(res["candidate"], [8.0, 7.5])

    def test_ignores_invalid_candidates_in_history(self):
        # 非法候选(不在白名单/非数字)应被忽略,不视为已尝试。
        history = [
            {"candidate_x": 3.0, "score": TOY["score"](3.0)},
            {"candidate_x": 99.0, "score": 0.9},       # 越界
            {"candidate_x": "7.5", "score": 0.9},       # 字符串
            {"candidate_x": True, "score": 0.9},        # bool
        ]
        res = strat.select_next_candidate(history, TOY)
        self.assertEqual(res["tried"], [3.0])
        self.assertEqual(res["best_seen"], 3.0)
        # 最接近 3.0 的未尝试为 2.5(距离 0.5)与 3.5(距离 0.5),平局取较小 2.5。
        self.assertEqual(res["candidate"], 2.5)

    def test_ignores_missing_candidate_field(self):
        # 缺候选字段的条目视为无有效历史。
        history = [{"score": 0.9}, {"reason": "no candidate"}]
        res = strat.select_next_candidate(history, TOY)
        self.assertEqual(res["status"], STATUS_START)
        self.assertEqual(res["candidate"], TOY["allowed_values"][0])
        self.assertEqual(res["tried"], [])

    def test_ignores_nonfinite_and_missing_scores(self):
        # nan / inf score 属格式异常,策略忽略整条(不视为可信历史)。
        history = [
            {"candidate_x": 4.0, "score": float("nan")},
            {"candidate_x": 4.0, "score": float("inf")},
        ]
        res = strat.select_next_candidate(history, TOY)
        # 全部被忽略 -> 视为空历史。
        self.assertEqual(res["status"], STATUS_START)
        self.assertEqual(res["tried"], [])

    def test_trusts_finite_score_from_local_writer(self):
        # 信任边界:策略信任"由本地评分函数写入的有限 score"。一个合法的
        # 有限分数(即使数值看似异常)不在策略防御范围——可信性由写入方保证。
        # 这里的 'score': 0.999 是有限数字,因此会被视为可信历史。
        res = strat.select_next_candidate(
            [{"candidate_x": 4.0, "score": 0.999}], TOY)
        self.assertEqual(res["status"], STATUS_OK)
        self.assertEqual(res["tried"], [4.0])
        self.assertEqual(res["best_seen"], 4.0)

    def test_distrusts_missing_score(self):
        res = strat.select_next_candidate(
            [{"candidate_x": 4.0}], TOY)   # 无 score
        self.assertEqual(res["tried"], [])
        self.assertEqual(res["status"], STATUS_START)

    def test_exhausted_when_all_candidates_tried(self):
        # toy 全部 21 个候选均已尝试 -> exhausted,不返回重复候选。
        history = [{"candidate_x": v, "score": TOY["score"](v)}
                   for v in TOY["allowed_values"]]
        res = strat.select_next_candidate(history, TOY)
        self.assertEqual(res["status"], STATUS_EXHAUSTED)
        self.assertIsNone(res["candidate"])
        self.assertEqual(len(res["tried"]), len(TOY["allowed_values"]))


class TestStrategyPerExperiment(unittest.TestCase):
    def test_validates_candidates_for_both_experiments(self):
        # 两个实验的候选都必须通过各自 validator。
        for exp in (TOY, MIN5):
            validate = make_default_validator(exp)
            for history in ([], [{"candidate_x": exp["allowed_values"][2],
                                  "score": exp["score"](exp["allowed_values"][2])}]):
                cand = safe(history, exp)
                if cand is not None:
                    self.assertEqual(validate({exp["candidate_field"]: cand}), cand)

    def test_domain_difference_min5(self):
        # min_5:历史最优 2.5(最接近 2.6),最近未尝试邻居 3.0。
        history = [
            {"candidate_x": 2.0, "score": MIN5["score"](2.0)},
            {"candidate_x": 2.5, "score": MIN5["score"](2.5)},
        ]
        res = strat.select_next_candidate(history, MIN5)
        self.assertEqual(res["status"], STATUS_OK)
        self.assertEqual(res["candidate"], 3.0)
        # 候选必须在 min_5 域内(0..5)。
        self.assertLessEqual(res["candidate"], 5.0)

    def test_toy_domain_differs_from_min5(self):
        # toy 与 min_5 候选域不同:同一历史里 5.0+ 在 toy 合法、在 min_5 非法(被忽略)。
        hist_toy = [{"candidate_x": 8.0, "score": TOY["score"](8.0)}]
        res_toy = strat.select_next_candidate(hist_toy, TOY)
        self.assertEqual(res_toy["candidate"], 7.5)   # 最近未尝试
        hist_min = [{"candidate_x": 8.0, "score": 0.9}]  # 8.0 不在 min_5 域
        res_min = strat.select_next_candidate(hist_min, MIN5)
        self.assertEqual(res_min["status"], STATUS_START)  # 视为无有效历史
        self.assertEqual(res_min["candidate"], 0.0)

    def test_reproducible_same_input_same_output(self):
        history = [{"candidate_x": 4.0, "score": TOY["score"](4.0)},
                   {"candidate_x": 4.5, "score": TOY["score"](4.5)}]
        runs = [strat.select_next_candidate(history, TOY) for _ in range(20)]
        self.assertTrue(all(r == runs[0] for r in runs))


class TestStrategyComparison(unittest.TestCase):
    """固定基线 / 简单确定性 / 新策略三者在同一玩具实验上的最小对照。"""

    def _fixed_baseline(self, history):
        # 固定基线:不读历史,始终返回固定候选值。
        return 0.0

    def _simple_deterministic(self, history, exp):
        # 简单确定性:顺序取未尝试的第一个候选,不看分数。
        tried = {h[exp["candidate_field"]] for h in history or []
                 if exp["candidate_field"] in h}
        untried = [v for v in exp["allowed_values"] if v not in tried]
        return untried[0] if untried else None

    def test_all_produce_valid_candidates(self):
        exp = MIN5
        history = [{"candidate_x": 1.0, "score": MIN5["score"](1.0)},
                   {"candidate_x": 1.5, "score": MIN5["score"](1.5)}]
        validate = make_default_validator(exp)
        for cand in (self._fixed_baseline(history),
                     self._simple_deterministic(history, exp),
                     self._new_strategy(history, exp)):
            if cand is not None:
                self.assertEqual(validate({exp["candidate_field"]: cand}), cand)

    def _new_strategy(self, history, exp):
        return safe(history, exp)

    def test_first_steps_advance_toward_min5_optimum(self):
        # min_5 最优在 2.5。新策略应朝高分方向(2.6 附近)推进:
        # 从历史 {1.0,1.5} 出发,应选 2.0 -> 2.5 -> 3.0 ... 呈升方向(朝 2.6)。
        exp = MIN5
        history = [{"candidate_x": 1.0, "score": MIN5["score"](1.0)},
                   {"candidate_x": 1.5, "score": MIN5["score"](1.5)}]
        next1 = self._new_strategy(history, exp)
        # 已知最优 1.5,最近未尝试邻居是 1.0? 不,1.0 已尝试;2.0 距离0.5,0.0距离1.5。
        # 最近为 2.0(距离 0.5)< 0.0(距离 1.5)。
        self.assertEqual(next1, 2.0)

    def test_no_repeat_until_exhausted(self):
        # 用尽策略连续推进应不重复候选,直至耗尽返回 None,而不是伪装进展。
        exp = MIN5
        history = []
        seen = []
        for _ in range(200):
            c = safe(history, exp)
            if c is None:
                break
            seen.append(c)
            history.append({"candidate_x": c, "score": exp["score"](c)})
        self.assertEqual(len(seen), len(exp["allowed_values"]))  # 全部不重复走完
        self.assertEqual(len(set(seen)), len(seen))
        self.assertIsNone(safe(history, exp))                     # 耗尽 -> None
        # 耗尽时状态明确。
        self.assertEqual(strat.select_next_candidate(history, exp)["status"],
                         STATUS_EXHAUSTED)

    def test_fixed_baseline_is_reproducible_but_does_not_follow_score(self):
        # 固定基线始终同一值(可复现),但其分数不随历史提升。
        r1 = self._fixed_baseline([{"candidate_x": 2.0, "score": 0.9}])
        r2 = self._fixed_baseline([{"candidate_x": 5.0, "score": 0.99}])
        self.assertEqual(r1, r2)


class TestLegacyCompat(unittest.TestCase):
    def test_run_research_default_still_model_driven(self):
        # 默认研究流程仍由模型生成候选并本地评分,未被策略改动。
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5}'):
            result = experiment.run_research(CFG, "g", 1)
        self.assertEqual(result["best_candidate"]["x"], 7.5)


if __name__ == "__main__":
    unittest.main()
