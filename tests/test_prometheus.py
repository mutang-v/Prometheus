# -*- coding: utf-8 -*-
"""Project Prometheus v0.1 — 自动化测试基线 (v0.2-A)

仅使用 Python 标准库 unittest。本套测试:
- 不调用任何真实 API / 不发付费请求 / 不依赖外网;
- 不使用任何第三方依赖;
- 不修改业务代码;任何无法通过的断言会如实报告为失败,不会为了通过而改动实现;
- 涉及文件写入的用例一律使用临时目录,与项目 runs/ 演示数据隔离。
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

# 保证能导入包内模块(与项目根目录无关,支持从任意 cwd 运行)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prometheus import experiment as exp
from prometheus import storage as st
from prometheus import api_client as ac


class TestScoreParameter(unittest.TestCase):
    """score_parameter: 固定玩具评分 1/(1+(x-7.3)^2),分数越高越好,最大为 1。"""

    def test_at_target_is_1(self):
        self.assertAlmostEqual(exp.score_parameter(7.3), 1.0, places=12)

    def test_symmetric_around_target(self):
        # 距离目标相同的两点分数应相等
        d = 0.5
        self.assertAlmostEqual(
            exp.score_parameter(7.3 - d), exp.score_parameter(7.3 + d), places=12)

    def test_score_in_01_range(self):
        for x in [0.0, 2.0, 7.3, 10.0, -100.0, 100.0]:
            s = exp.score_parameter(x)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_monotonic_near_target(self):
        # 越接近 7.3 分越高
        self.assertGreater(
            exp.score_parameter(7.3), exp.score_parameter(7.0))
        self.assertGreater(
            exp.score_parameter(7.0), exp.score_parameter(2.0))


class TestExtractJson(unittest.TestCase):
    """_extract_json: 接受裸 JSON 或 ```json``` 代码块,拒绝非法输入。"""

    def test_bare_json(self):
        text = json.dumps({"candidate_x": 7.0, "reason": "ok"})
        self.assertEqual(exp._extract_json(text),
                         {"candidate_x": 7.0, "reason": "ok"})

    def test_fenced_json_with_lang(self):
        text = "```json\n" + json.dumps({"candidate_x": 7.5}) + "\n```"
        self.assertEqual(exp._extract_json(text), {"candidate_x": 7.5})

    def test_fenced_json_without_lang(self):
        text = "```\n" + json.dumps({"candidate_x": 8.0}) + "\n```"
        self.assertEqual(exp._extract_json(text), {"candidate_x": 8.0})

    def test_leading_trailing_noise(self):
        text = "以下是结果:\n" + json.dumps({"candidate_x": 7.0}) + "\n完毕"
        self.assertEqual(exp._extract_json(text), {"candidate_x": 7.0})

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            exp._extract_json("完全没有 JSON 的文本")

    def test_rejects_malformed_json(self):
        # json.loads 抛 json.JSONDecodeError(继承自 ValueError)
        with self.assertRaises(ValueError):
            exp._extract_json('{"candidate_x": }')


class TestRunResearchValidation(unittest.TestCase):
    """候选值的类型、有限性与 domain 白名单校验(单轮逻辑,mock 掉模型调用)。"""

    def _call_research(self, model_reply, domain=None):
        # mock call_model 使其直接返回给定文本,避免任何真实 API 调用
        cfg = {"model": "fake", "base_url": "http://fake", "api_key": "none"}
        with mock.patch.object(exp, "call_model", return_value=model_reply):
            return exp.run_research(cfg, "toy goal", rounds=1)

    def test_accepts_valid_in_domain(self):
        result = self._call_research('{"candidate_x": 7.5, "reason": "接近最优"}')
        self.assertEqual(result["round_results"][0]["candidate_x"], 7.5)

    def test_rejects_out_of_domain(self):
        # 7.25 不在 0.5 步长网格上
        with self.assertRaises(ValueError):
            self._call_research('{"candidate_x": 7.25}')

    def test_rejects_negative_out_of_domain(self):
        with self.assertRaises(ValueError):
            self._call_research('{"candidate_x": -1.0}')

    def test_rejects_above_domain(self):
        with self.assertRaises(ValueError):
            self._call_research('{"candidate_x": 10.5}')

    def test_rejects_non_numeric_candidate(self):
        with self.assertRaises((ValueError, TypeError)):
            self._call_research('{"candidate_x": "not_a_number"}')

    def test_rejects_non_finite_for_unknown_type(self):
        # 字符串 "nan" 无法转 float,应作为非法候选被拒绝
        with self.assertRaises((ValueError, TypeError)):
            self._call_research('{"candidate_x": "nan"}')


class TestSaveRun(unittest.TestCase):
    """save_run: 基本保存行为与文件名冲突风险(使用临时目录)。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_runs_dir_and_file(self):
        result = {"mode": "test", "x": 1}
        path = st.save_run(self.root, result)
        self.assertTrue(path.exists())
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["x"], 1)

    def test_filename_contains_utc_stamp_and_mode(self):
        result = {"mode": "local_demo", "x": 2}
        path = st.save_run(self.root, result)
        # 格式 <YYYYmmdd>T<HHMMSS>Z_<mode>.json
        name = path.name
        self.assertTrue(name.endswith("_local_demo.json"), name)
        # 时间戳部分为 8 位日期 + T + 6 位时分秒 + Z
        stamp = name.removesuffix("_local_demo.json")
        self.assertRegex(stamp, r"^\d{8}T\d{6}Z$")

    def test_no_collision_within_same_second(self):
        # 同一秒内连续保存多条记录,必须都保留且互不覆盖。
        # 用 mock 固定时钟,确保两次调用得到相同的时间戳,确定性制造"同秒冲突"。
        fixed_dt = lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        results = [
            {"mode": "collision", "x": 1, "tag": "first"},
            {"mode": "collision", "x": 2, "tag": "second"},
            {"mode": "collision", "x": 3, "tag": "third"},
        ]
        with mock.patch.object(st, "datetime", wraps=datetime) as fake_dt:
            fake_dt.now.return_value = fixed_dt()
            paths = [st.save_run(self.root, r) for r in results]

        # 三条路径不同(互不覆盖),文件都存在且数量正确
        self.assertEqual(len(set(paths)), 3, "同秒三条记录路径必须互不相同")
        files = sorted(p.name for p in paths)
        self.assertEqual(len(files), 3)

        # 各条内容完整且彼此独立(未发生互相覆盖)
        saved = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        self.assertEqual([s["tag"] for s in saved],
                         ["first", "second", "third"])
        self.assertEqual([s["x"] for s in saved], [1, 2, 3])

    def test_does_not_overwrite_existing_target(self):
        # 当目标文件(如 <stamp>_<mode>.json)已存在时,不得覆盖其内容,
        # 必须改写到唯一的新文件名。
        fixed_dt = lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        out = self.root / "runs"
        out.mkdir(exist_ok=True)
        stamp = "20260102T030405Z"
        # 预置一个已存在的"第一秒"记录
        existing = out / f"{stamp}_collision.json"
        existing.write_text(json.dumps({"tag": "PREEXISTING"}), encoding="utf-8")

        with mock.patch.object(st, "datetime", wraps=datetime) as fake_dt:
            fake_dt.now.return_value = fixed_dt()
            path = st.save_run(self.root, {"mode": "collision", "tag": "NEW"})

        # 原文件内容保持不变
        self.assertEqual(
            json.loads(existing.read_text(encoding="utf-8"))["tag"],
            "PREEXISTING")
        # 新记录写到了不同的唯一文件,内容完整
        self.assertNotEqual(path, existing)
        self.assertTrue(path.exists())
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["tag"], "NEW")
        # 名字形如 <stamp>_collision_1.json
        self.assertEqual(path.name, f"{stamp}_collision_1.json")


class TestCliBehavior(unittest.TestCase):
    """CLI 参数互斥、rounds 范围与预期错误退出行为。

    不写盘的分支用 subprocess 直接跑真实 main.py,验证进程级退出码;
    需要写盘或会触碰配置的分支用 mock 隔离,避免污染真实 runs/ 与网络。
    """

    def _run_cli(self, *args):
        env = dict(os.environ)
        env.pop("PROMETHEUS_API_KEY", None)  # 确保无密钥,避免假象
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *args],
            capture_output=True, text=True, timeout=30, env=env,
            cwd=str(ROOT))
        return proc

    def test_check_exit_zero(self):
        proc = self._run_cli("--check")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("本地检查通过", proc.stdout)

    def test_demo_exit_zero(self):
        # --demo 会写盘。为避免污染真实 runs/,这里以单元级方式隔离:
        # import main,把 save_run 替换为写入临时目录的假实现,再调用 main()。
        import main as M
        from unittest import mock as _mock
        tmpdir = tempfile.TemporaryDirectory()
        fake_run_path = Path(tmpdir.name) / "runs" / "fake_demo.json"
        try:
            # 替换 main 模块对 save_run 的引用,使其写临时目录
            def fake_save_run(root, result):
                fake_run_path.parent.mkdir(parents=True, exist_ok=True)
                fake_run_path.write_text(
                    json.dumps(result, ensure_ascii=False), encoding="utf-8")
                return fake_run_path

            with _mock.patch.object(M, "save_run", side_effect=fake_save_run), \
                    mock.patch.object(sys, "argv", ["main.py", "--demo"]):
                M.main()
            self.assertTrue(fake_run_path.exists(), "demo 应调用 save_run 写入文件")
            saved = json.loads(fake_run_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "local_demo")
        finally:
            tmpdir.cleanup()

    def test_mutually_exclusive_exit_2(self):
        proc = self._run_cli("--demo", "--goal", "x")
        self.assertEqual(proc.returncode, 2)  # argparse usage error

    def test_rounds_out_of_range_exit_2(self):
        proc = self._run_cli("--goal", "t", "--rounds", "5")
        self.assertEqual(proc.returncode, 2)

    def test_rounds_negative_exit_2(self):
        proc = self._run_cli("--goal", "t", "--rounds", "0")
        self.assertEqual(proc.returncode, 2)

    def test_check_api_without_config_concise_error_exit_1(self):
        proc = self._run_cli("--check-api")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("错误：", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_no_mutual_arg_required(self):
        # 不提供任何互斥参数,argparse 报 usage 错误 exit 2
        proc = self._run_cli()
        self.assertEqual(proc.returncode, 2)


class TestRunCliUnit(unittest.TestCase):
    """run_cli() 异常分类:期望错误被转简洁、非期望异常不被吞掉(纯内存,不碰磁盘)。"""

    def test_expected_runtime_error_returns_1_no_traceback(self):
        import contextlib
        import io
        import main as M
        orig = M.main

        def boom():
            raise RuntimeError("配置缺失(模拟)")
        M.main = boom
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                rc = M.run_cli()
        finally:
            M.main = orig
        self.assertEqual(rc, 1)
        self.assertIn("配置缺失", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_unexpected_exception_not_swallowed(self):
        import main as M
        orig = M.main

        def boom():
            raise KeyError("unexpected boom")
        M.main = boom
        try:
            with self.assertRaises(KeyError):
                M.run_cli()
        finally:
            M.main = orig


if __name__ == "__main__":
    unittest.main()
