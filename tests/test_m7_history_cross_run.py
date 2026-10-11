# -*- coding: utf-8 -*-
"""v0.3 M7: 跨运行实验历史记忆(最小实现)测试。

覆盖(M7 要求 1-12):
 1. 空/不存在历史目录能正常启动 history 策略;
 2. 第一轮运行保存结果后,第二次运行能读取并利用第一轮历史;
 3. 不同 experiment_id 的历史严格隔离;
 4. 损坏 JSON、缺字段、不兼容 schema 不会被误当作有效候选;
 5. 非法候选(bool/字符串/None/NaN/正负无穷/越界)与非法 score 不进入有效历史;
 6. 历史中的重复候选不会导致重复评估;
 7. 候选全部耗尽时正确停止;
 8. 现有 run_research(cfg, goal, rounds) 调用兼容;
 9. 默认 model 策略行为不被跨运行历史改变;
10. M5 的 fixed/history/random 对照逻辑、预算约束与固定种子可复现性不回归;
11. 所有涉及文件读写的测试均使用临时目录,且测试结束后清理;
12. history_store.load_history 的输入输出与异常/坏记录处理被直接测试。

全部离线(不调用真实 API / 不联网 / 无依赖安装 / 无 Git 操作),不碰真实 runs/。
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, r"c:\Users\lenovo\Desktop\Prometheus")

import main as main_mod
from prometheus import experiment
from prometheus.experiments import EXPERIMENTS, get_experiment
from prometheus.storage import save_run
from prometheus.strategy_compare import run_strategy, compare_strategies
from prometheus import history_store
from prometheus.history_store import (
    experiment_fingerprint, load_history, load_history_detailed,
)

TOY = EXPERIMENTS["toy_7_3"]
MIN5 = EXPERIMENTS["min_5"]
TOY_FP = experiment_fingerprint(TOY)
MIN5_FP = experiment_fingerprint(MIN5)
EMPTY_CFG = {}


def _make_record(exp_id, exp_fp, rounds, file_name):
    """构造一条可被本项目读取器识别的运行记录 dict。"""
    exp = EXPERIMENTS[exp_id]
    field = exp["candidate_field"]
    return {
        "project": "Prometheus",
        "mode": "history_research",
        "strategy": "history",
        "experiment_id": exp_id,
        "experiment_fingerprint": exp_fp,
        "round_results": [
            {"round": i + 1, field: x, "score": exp["score"](x), "reason": "r"}
            for i, x in enumerate(rounds)
        ],
    }


class TestEmptyOrMissingDir(unittest.TestCase):
    # 要求 1: 空历史目录 / 不存在目录能正常启动 history 策略。
    def test_missing_runs_dir_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_such_dir"
            res = experiment.run_research(EMPTY_CFG, "g", 2,
                                          strategy="history", runs_dir=missing)
        self.assertEqual(len(res["round_results"]), 2)

    def test_empty_runs_dir_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = experiment.run_research(EMPTY_CFG, "g", 2,
                                          strategy="history", runs_dir=tmp)
        self.assertEqual(len(res["round_results"]), 2)

    def test_load_history_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries, skipped = load_history(tmp, "toy_7_3")
        self.assertEqual(entries, [])
        self.assertEqual(skipped, [])
        entries2, skipped2 = load_history(Path(tmp) / "missing", "toy_7_3")
        self.assertEqual(entries2, [])
        self.assertEqual(skipped2, [])


class TestCrossRunReuse(unittest.TestCase):
    # 要求 2: 第一次运行保存结果后,第二次运行能读取并利用第一轮历史。
    def _write_run(self, root, res):
        return save_run(Path(root), res)

    def test_second_run_uses_first_run_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # 第一次运行:空历史 -> 从起点 0.0 开始,rounds=2 得 [0.0, 0.5]。
            first = experiment.run_research(EMPTY_CFG, "g", 2,
                                            experiment_id="toy_7_3",
                                            strategy="history", runs_dir=tmp)
            first_xs = [rr["candidate_x"] for rr in first["round_results"]]
            self.assertEqual(first_xs, [0.0, 0.5])
            # save_run 写到 <tmp>/runs/<stamp>_history_research.json。
            self._write_run(tmp, first)

            # 第二次运行:从 <tmp>/runs 读取第一轮保存的历史(已试 0.0,0.5)继续,
            # 第一个新候选是 1.0,证明历史确实影响了候选选择且未重复。
            second = experiment.run_research(EMPTY_CFG, "g", 2,
                                             experiment_id="toy_7_3",
                                             strategy="history",
                                             runs_dir=tmp / "runs")
            second_xs = [rr["candidate_x"] for rr in second["round_results"]]
            self.assertEqual(second_xs[0], 1.0)
            # 绝不重复评估第一轮已保存的候选。
            self.assertNotIn(second_xs[0], {0.0, 0.5})

    def test_cross_run_result_contains_id_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = experiment.run_research(EMPTY_CFG, "g", 1,
                                            experiment_id="toy_7_3",
                                            strategy="history", runs_dir=tmp)
        self.assertEqual(first["experiment_id"], "toy_7_3")
        self.assertEqual(first["experiment_fingerprint"], TOY_FP)


class TestExperimentIsolation(unittest.TestCase):
    # 要求 3: 不同 experiment_id 的历史严格隔离。
    def test_other_experiment_history_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "min.json").write_text(
                json.dumps(_make_record("min_5", MIN5_FP, [0.0, 0.5, 1.0],
                                        "min.json")),
                encoding="utf-8")
            # 加载 toy 历史时,min_5 记录被跳过。
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(entries, [])
            self.assertEqual(len(skipped), 1)
            # toy 运行不受 min 记录影响,仍从起点开始。
            res = experiment.run_research(EMPTY_CFG, "g", 3,
                                          experiment_id="toy_7_3",
                                          strategy="history", runs_dir=tmp)
            xs = [rr["candidate_x"] for rr in res["round_results"]]
            self.assertEqual(xs, [0.0, 0.5, 1.0])

    def test_correct_experiment_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "toy.json").write_text(
                json.dumps(_make_record("toy_7_3", TOY_FP, [0.0, 0.5],
                                        "toy.json")),
                encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(len(entries), 2)
            self.assertEqual(skipped, [])


class TestBadRecordsSkipped(unittest.TestCase):
    # 要求 4/5: 损坏 JSON、缺字段、不兼容 schema、非法候选/score 不进入有效历史。
    # 且单个坏文件不阻断其余有效历史。
    def test_corrupt_json_skipped_no_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.json").write_text("{not-json", encoding="utf-8")
            Path(tmp, "good.json").write_text(
                json.dumps(_make_record("toy_7_3", TOY_FP, [0.0, 0.5],
                                        "good.json")),
                encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(len(entries), 2)
            reasons = [r for _, r in skipped]
            self.assertTrue(any("无法解析 JSON" in r for r in reasons))

    def test_non_dict_record_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "arr.json").write_text("[1,2,3]", encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(entries, [])
            self.assertEqual(len(skipped), 1)

    def test_record_missing_id_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _make_record("toy_7_3", TOY_FP, [0.0, 0.5], "x.json")
            del rec["experiment_id"]
            Path(tmp, "x.json").write_text(json.dumps(rec), encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(entries, [])
            self.assertTrue(any("experiment_id" in r for _, r in skipped))

    def test_fingerprint_mismatch_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _make_record("toy_7_3", "deadbeef" * 8, [0.0, 0.5], "x.json")
            Path(tmp, "x.json").write_text(json.dumps(rec), encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(entries, [])
            self.assertTrue(any("fingerprint" in r for _, r in skipped))

    def test_missing_round_results_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _make_record("toy_7_3", TOY_FP, [0.0, 0.5], "x.json")
            del rec["round_results"]
            Path(tmp, "x.json").write_text(json.dumps(rec), encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(entries, [])
            self.assertTrue(any("round_results" in r for _, r in skipped))

    def test_bad_candidates_and_scores_skipped(self):
        # 非法候选:bool/字符串/None/NaN/±Inf/越界;非法 score:nan/字符串/bool。
        bad_entries = [
            {TOY["candidate_field"]: True, "score": 0.5},   # bool
            {TOY["candidate_field"]: "7.5", "score": 0.5},  # str
            {TOY["candidate_field"]: None, "score": 0.5},   # None
            {TOY["candidate_field"]: float("nan"), "score": 0.5},
            {TOY["candidate_field"]: float("inf"), "score": 0.5},
            {TOY["candidate_field"]: 99.0, "score": 0.5},   # 越界
            {TOY["candidate_field"]: 7.5, "score": float("nan")},
            {TOY["candidate_field"]: 7.5, "score": "0.5"},
            {TOY["candidate_field"]: 7.5, "score": True},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for i, entry in enumerate(bad_entries):
                rec = _make_record("toy_7_3", TOY_FP, [0.0, 0.5], f"x{i}.json")
                rec["round_results"].append(dict(entry))
                Path(tmp, f"x{i}.json").write_text(
                    json.dumps(rec), encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(entries, [])
            self.assertEqual(len(skipped), len(bad_entries))
            self.assertTrue(all("非法逐轮候选/score" in r for _, r in skipped))

    def test_only_valid_fingerprint_record_used(self):
        # 同目录多个文件中,仅指纹完全匹配的被采用,其余被跳过且不阻断。
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(_make_record("toy_7_3", TOY_FP, [0.0], "a.json")),
                encoding="utf-8")
            Path(tmp, "b.json").write_text(
                json.dumps(_make_record("min_5", MIN5_FP, [1.0], "b.json")),
                encoding="utf-8")
            Path(tmp, "c.json").write_text(
                json.dumps(_make_record("toy_7_3", "bad" * 8, [00.0], "c.json")),
                encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(len(entries), 1)      # 只有 a 被采用
            self.assertEqual(len(skipped), 2)      # b, c 被跳过


class TestNoDuplicateEvaluation(unittest.TestCase):
    # 要求 6: 历史中的重复候选不会导致重复评估。
    def test_duplicate_candidates_in_history_not_re_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 历史含重复候选 0.0(两次),但 select_next_candidate 视其为已尝试一次。
            dup = _make_record("toy_7_3", TOY_FP, [0.0, 0.0, 0.0], "d.json")
            Path(tmp, "d.json").write_text(json.dumps(dup), encoding="utf-8")
            res = experiment.run_research(EMPTY_CFG, "g", 1,
                                          experiment_id="toy_7_3",
                                          strategy="history", runs_dir=tmp)
            xs = [rr["candidate_x"] for rr in res["round_results"]]
            # 0.0 已被历史占用,不会重复评估;首个新候选是与 0.0 最近的未尝试 0.5。
            self.assertEqual(xs, [0.5])


class TestExhaustionStops(unittest.TestCase):
    # 要求 7: 候选全部耗尽时正确停止。
    def test_full_domain_within_one_run_exhausts(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = experiment.run_research(EMPTY_CFG, "g", 100,
                                          experiment_id="min_5",
                                          strategy="history", runs_dir=tmp)
            got = [rr["candidate_x"] for rr in res["round_results"]]
            self.assertEqual(len(got), len(MIN5["allowed_values"]))  # 11
            self.assertEqual(len(set(got)), 11)

    def test_history_plus_new_rounds_exhausts_without_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "h.json").write_text(
                json.dumps(_make_record("min_5", MIN5_FP,
                                        MIN5["allowed_values"][:5], "h.json")),
                encoding="utf-8")
            res = experiment.run_research(EMPTY_CFG, "g", 100,
                                          experiment_id="min_5",
                                          strategy="history", runs_dir=tmp)
            got = [rr["candidate_x"] for rr in res["round_results"]]
            # 已保存 5 个;新一轮 round_results 只含"本次新评估"的候选,故为剩余 6 个。
            self.assertEqual(len(got), 6)
            self.assertEqual(len(set(got)), 6)       # 新评估无重复
            # 历史(已加载 5)与本次新评估(6)合起来覆盖整个 min_5 域,且无重叠。
            loaded, _ = load_history(tmp, "min_5")
            loaded_xs = {e["candidate_x"] for e in loaded}
            self.assertEqual(len(loaded_xs), 5)
            self.assertEqual(loaded_xs | set(got), set(MIN5["allowed_values"]))
            self.assertTrue(loaded_xs.isdisjoint(set(got)))


class TestBackwardCompat(unittest.TestCase):
    # 要求 8/9: 旧签名兼容 + 默认 model 策略不受跨运行历史影响。
    def test_old_signature_without_runs_dir_still_works(self):
        with mock.patch.object(experiment, "call_model",
                               return_value='{"candidate_x": 7.5}'):
            res = experiment.run_research({"m": "x"}, "g", 3)
        self.assertEqual(res["strategy"], "model")
        self.assertEqual(len(res["round_results"]), 3)

    def test_history_without_runs_dir_unchanged_m41(self):
        # runs_dir=None 时,history 行为与 M4.1 完全一致(不从目录读取)。
        before = experiment.run_research(EMPTY_CFG, "g", 5,
                                         experiment_id="min_5",
                                         strategy="history")
        # 即便真实 runs 目录存在,不传 runs_dir 也不会读取(不依赖外部状态)。
        self.assertEqual(len(before["round_results"]), 5)
        self.assertEqual(before["strategy"], "history")

    def test_default_model_not_affected_by_runs_history(self):
        # model 策略不接收 runs_dir,跨运行历史不影响其行为。
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "h.json").write_text(
                json.dumps(_make_record("toy_7_3", TOY_FP, [7.5, 7.0], "h.json")),
                encoding="utf-8")
            with mock.patch.object(experiment, "call_model",
                                   return_value='{"candidate_x": 8.0}'):
                res = experiment.run_research({}, "g", 3, strategy="model")
        self.assertEqual(res["strategy"], "model")
        # 返回 3 轮,完全由 mock 模型决定,未受磁盘历史影响。
        self.assertEqual([rr["candidate_x"] for rr in res["round_results"]],
                         [8.0, 8.0, 8.0])

    def test_history_result_gains_id_fp_but_keeps_old_keys(self):
        res = experiment.run_research(EMPTY_CFG, "g", 2, strategy="history")
        self.assertEqual(res["experiment_id"], "toy_7_3")
        self.assertEqual(res["experiment_fingerprint"], TOY_FP)
        # 旧契约键仍在。
        self.assertIn("best_candidate", res)
        self.assertIn("round_results", res)
        self.assertEqual(res["mode"], "history_research")


class TestLoadHistoryDirect(unittest.TestCase):
    # 要求 12: 直接测试 load_history 的输入输出与异常/坏记录处理。
    def test_load_history_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_history(tmp, "toy_7_3"), ([], []))
            self.assertEqual(load_history(Path(tmp) / "gone", "toy_7_3"),
                             ([], []))
        missing = Path(tempfile.gettempdir()) / "m7_no_such_dir_xyz"
        self.assertEqual(load_history(missing, "toy_7_3"), ([], []))

    def test_load_history_valid_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "v.json").write_text(
                json.dumps(_make_record("toy_7_3", TOY_FP, [0.0, 0.5], "v.json")),
                encoding="utf-8")
            entries, skipped = load_history(tmp, "toy_7_3")
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0], {"candidate_x": 0.0,
                                          "score": TOY["score"](0.0)})
            self.assertEqual(skipped, [])

    def test_load_history_unknown_experiment_raises(self):
        # 未知实验 ID 应明确报错(经 get_experiment),而非静默返回。
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                load_history(tmp, "no_such_experiment")

    def test_detailed_returns_skipped_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.json").write_text("qw{", encoding="utf-8")
            Path(tmp, "ok.json").write_text(
                json.dumps(_make_record("toy_7_3", TOY_FP, [0.0], "ok.json")),
                encoding="utf-8")
            entries, skipped = load_history_detailed(tmp, "toy_7_3")
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(skipped), 1)
            self.assertIn("bad.json", skipped[0][0])


class TestM5NotRegressed(unittest.TestCase):
    # 要求 10: M5 fixed/history/random 对照、预算与固定种子可复现性不回归。
    def test_m5_compare_budget_and_seed_repro(self):
        a = compare_strategies("min_5", budget=4, seed=7)
        b = compare_strategies("min_5", budget=4, seed=7)
        self.assertEqual(a["strategies"]["random"]["round_results"],
                         b["strategies"]["random"]["round_results"])
        self.assertEqual(a["budget"], 4)
        for s in ("fixed", "history", "random"):
            self.assertEqual(a["strategies"][s]["eval_count"], 4)

    def test_m5_isolated_real_runs_untouched(self):
        import os
        from pathlib import Path as P
        runs_dir = P(r"c:\Users\lenovo\Desktop\Prometheus\runs")
        before = sorted(os.listdir(runs_dir))
        for _ in range(2):
            run_strategy("toy_7_3", "random", budget=5, seed=3)
            compare_strategies("min_5", budget=4, seed=42)
        after = sorted(os.listdir(runs_dir))
        self.assertEqual(after, before)


class TestCliLoadsRealRunsDir(unittest.TestCase):
    # 要求: 主程序真实调用链 - history 策略从配置的 runs 目录加载历史。
    # 用临时目录替换 main.ROOT,验证 CLI 把 ROOT/runs 传给 run_research(runs_dir)。
    def test_cli_history_passes_runs_dir_to_run_research(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        captured = {}

        def fake_run_research(cfg, goal, rounds, experiment_id="toy_7_3",
                              strategy="model", runs_dir=None):
            captured["runs_dir"] = runs_dir
            captured["strategy"] = strategy
            return experiment.run_demo()

        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(main_mod, "ROOT", fake_root):
                with mock.patch.object(main_mod, "run_research",
                                       side_effect=fake_run_research):
                    with mock.patch.object(main_mod, "save_run",
                                           return_value=Path(tmp) / "r.json"):
                        with mock.patch.object(sys, "argv",
                                               ["prog", "--goal", "g",
                                                "--rounds", "2",
                                                "--experiment", "toy_7_3",
                                                "--strategy", "history"]):
                            with redirect_stdout(out), redirect_stderr(err):
                                code = main_mod.run_cli()
        self.assertEqual(code, 0, err.getvalue())
        # CLI 确实把 ROOT/runs 作为 history 加载目录传入 run_research。
        self.assertEqual(captured["runs_dir"], fake_root / "runs")
        self.assertEqual(captured["strategy"], "history")

    def test_cli_model_passes_runs_dir_none(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        captured = {}

        def fake_run_research(cfg, goal, rounds, experiment_id="toy_7_3",
                              strategy="model", runs_dir=None):
            captured["runs_dir"] = runs_dir
            self.assertIsNone(runs_dir)
            return experiment.run_demo()

        with tempfile.TemporaryDirectory() as tmp:
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(main_mod, "ROOT", Path(tmp)):
                with mock.patch.object(main_mod, "load_config",
                                       return_value={"model": "m"}):
                    with mock.patch.object(main_mod, "run_research",
                                           side_effect=fake_run_research):
                        with mock.patch.object(main_mod, "save_run",
                                               return_value=Path(tmp) / "r.json"):
                            with mock.patch.object(sys, "argv",
                                                   ["prog", "--goal", "g",
                                                    "--rounds", "1"]):
                                with redirect_stdout(out), redirect_stderr(err):
                                    code = main_mod.run_cli()
        self.assertEqual(code, 0, err.getvalue())


if __name__ == "__main__":
    unittest.main()
