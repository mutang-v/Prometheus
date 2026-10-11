# -*- coding: utf-8 -*-
"""Project Prometheus v0.1 — v0.2-C 输入与异常处理加固审查测试

原则:
- 仅使用标准库;不调用真实 API / 不联网 / 不产生付费请求(所有 API 调用均 mock);
- 不修改任何业务代码;本文件用于"审查",通过断言记录当前实现的真实行为;
  凡是当前实现无法满足"期望的加固行为"的用例,会如实 FAIL 并在运行结果中暴露。
- 涉及写文件一律用临时目录。
"""
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prometheus import experiment as exp
from prometheus import storage as st
from prometheus import api_client as ac


def _research_with_reply(reply, domain=None):
    """驱动 run_research 单轮,用假配置 + mock call_model,绝不触发真实网络。"""
    cfg = {"model": "fake", "base_url": "http://fake.local", "api_key": "none"}
    with mock.patch.object(exp, "call_model", return_value=reply):
        return exp.run_research(cfg, "toy goal", rounds=1)


class TestCandidateValidation(unittest.TestCase):
    """候选值校验:有限性、类型(bool/字符串/容器/None)与白名单 domain。"""

    DOMAIN = [i / 2 for i in range(0, 21)]  # 与 run_research 内 domain 一致

    def test_accepts_finite_in_domain(self):
        r = _research_with_reply('{"candidate_x": 7.5}')
        self.assertEqual(r["round_results"][0]["candidate_x"], 7.5)

    def test_accepts_integer_written_value(self):
        r = _research_with_reply('{"candidate_x": 7}')
        self.assertEqual(r["round_results"][0]["candidate_x"], 7.0)

    def test_rejects_nan(self):
        # float("NaN") 可成功,但 NaN 非有限数值且不在白名单 → 期望被拒
        with self.assertRaises(ValueError):
            _research_with_reply('{"candidate_x": NaN}')

    def test_rejects_positive_infinity(self):
        with self.assertRaises(ValueError):
            _research_with_reply('{"candidate_x": Infinity}')

    def test_rejects_negative_infinity(self):
        with self.assertRaises(ValueError):
            _research_with_reply('{"candidate_x": -Infinity}')

    def test_bool_true_should_be_rejected(self):
        # bool 是 int 子类:float(True)=1.0 且 1.0 在 domain 内。
        # 修复后:显式类型校验拒绝布尔值,并给出稳定的 ValueError(不泄漏 TypeError)。
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": true}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_bool_false_should_be_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": false}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_numeric_string_out_of_domain_rejected(self):
        # 数字字符串 "7.5":float("7.5") 可转,修复后必须按类型拒绝。
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": "7.5"}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_plain_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": "abc"}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_none_rejected(self):
        # 缺键/None 应得到稳定的 ValueError,而非 KeyError 泄漏。
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": null}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_list_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": [7.5]}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_dict_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"candidate_x": {"x": 7.5}}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_out_of_domain_rejected(self):
        with self.assertRaises(ValueError):
            _research_with_reply('{"candidate_x": 7.25}')


class TestModelOutputParsing(unittest.TestCase):
    """模型输出解析的容错与可理解错误。"""

    def test_valid_json_parses(self):
        r = _research_with_reply('{"candidate_x": 7.5, "reason": "ok"}')
        self.assertEqual(r["round_results"][0]["reason"], "ok")

    def test_fenced_json_parses(self):
        r = _research_with_reply('```json\n{"candidate_x": 7.5}\n```')
        self.assertEqual(r["round_results"][0]["candidate_x"], 7.5)

    def test_missing_required_field(self):
        # 缺 candidate_x 键 → 修复后给出稳定的 ValueError(不泄漏 KeyError)。
        with self.assertRaises(ValueError) as ctx:
            _research_with_reply('{"reason": "no candidate"}')
        self.assertIn("必须是有限数字", str(ctx.exception))

    def test_field_wrong_type(self):
        with self.assertRaises((ValueError, TypeError)):
            _research_with_reply('{"candidate_x": "not-a-number"}')

    def test_empty_output(self):
        with self.assertRaises(ValueError):
            _research_with_reply("")

    def test_invalid_json(self):
        with self.assertRaises(ValueError):
            _research_with_reply("this is not json")

    def test_json_with_nan_value(self):
        # Python json.loads 默认接受 NaN,期望后续白名单/有限性拒绝
        with self.assertRaises(ValueError):
            _research_with_reply('{"candidate_x": NaN}')

    def test_json_with_infinity_value(self):
        with self.assertRaises(ValueError):
            _research_with_reply('{"candidate_x": Infinity}')


class TestApiExceptionHandling(unittest.TestCase):
    """api_client 异常处理(全部 mock,不触网)。"""

    def _run_call(self, raiser=None, cfg=None):
        cfg = cfg or {"base_url": "https://api.example/v1", "model": "m",
                      "api_key": "k", "timeout_seconds": 5}
        if raiser is not None:
            patcher = mock.patch.object(ac.urllib.request, "urlopen",
                                        side_effect=raiser)
        else:
            patcher = mock.patch.object(
                ac.urllib.request, "urlopen",
                return_value=mock.MagicMock())
        with patcher:
            return ac.call_model(cfg, "hi")

    def test_load_config_missing_file_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(RuntimeError):
                ac.load_config(root)

    def test_load_config_missing_api_key_raises_runtime_error(self):
        with mock.patch.dict(ac.os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "config.json").write_text(
                    json.dumps({"base_url": "https://x/v1", "model": "m",
                                "api_key_env": "PROMETHEUS_API_KEY"}),
                    encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    ac.load_config(root)

    def test_http_error_wrapped_as_runtime_error(self):
        err = urllib.error.HTTPError(
            "url", 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))
        with self.assertRaises(RuntimeError) as ctx:
            self._run_call(raiser=err)
        self.assertIn("HTTP 401", str(ctx.exception))

    def test_connection_error_wrapped_as_runtime_error(self):
        err = urllib.error.URLError("connection refused")
        with self.assertRaises(RuntimeError) as ctx:
            self._run_call(raiser=err)
        self.assertIn("连接 API 失败", str(ctx.exception))

    def test_timeout_wrapped_as_runtime_error(self):
        # urllib 超时抛 socket.timeout(继承 OSError,非 URLError)。
        # 修复后:超时应被包装为可理解的 RuntimeError,而非向外泄漏 socket.timeout。
        with self.assertRaises(RuntimeError) as ctx:
            self._run_call(raiser=socket.timeout("timed out"))
        self.assertIn("超时", str(ctx.exception))

    def test_timeout_error_wrapped_as_runtime_error(self):
        # TimeoutError 与 socket.timeout 同源,同样应被包装。
        with self.assertRaises(RuntimeError) as ctx:
            self._run_call(raiser=TimeoutError("timed out"))
        self.assertIn("超时", str(ctx.exception))

    def test_malformed_response_format(self):
        # 返回非法 JSON 响应 → 期望 RuntimeError(api_client 的格式分支)
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"choices": [broken json'

        cfg = {"base_url": "https://api.example/v1", "model": "m",
               "api_key": "k", "timeout_seconds": 5}
        with mock.patch.object(ac.urllib.request, "urlopen",
                               return_value=FakeResp()), \
             self.assertRaises(RuntimeError):
            ac.call_model(cfg, "hi")

    def test_malformed_response_missing_choices(self):
        # 合法 JSON 但缺少 choices/message 必要字段 → RuntimeError(不泄漏 KeyError)。
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"foo": 1}'

        cfg = {"base_url": "https://api.example/v1", "model": "m",
               "api_key": "k", "timeout_seconds": 5}
        with mock.patch.object(ac.urllib.request, "urlopen",
                               return_value=FakeResp()), \
             self.assertRaises(RuntimeError) as ctx:
            ac.call_model(cfg, "hi")
        self.assertIn("格式", str(ctx.exception))

    def test_unexpected_exception_is_not_reclassified_as_user_error(self):
        # 真实程序错误(如 500 之外的任意异常)不应被归为普通输入错误。
        # 这里用 ValueError 验证:当前实现并不捕获 ValueError(只捕获 HTTP/URL/格式),
        # 因此会原样逃逸——保持程序错误可辨识。
        try:
            self._run_call(raiser=ValueError("boom"))
        except ValueError:
            pass  # 程序错误原样传播,未错误包装
        else:
            self.fail("非网络类异常不应被静默吞掉")


class TestCliErrorBehavior(unittest.TestCase):
    """CLI 预期/意外/参数错误的退出码与提示(仅测不写盘的嗅探路径)。"""

    def _run_cli(self, *args):
        env = dict(PROMETHEUS_API_KEY_DUMMY=None)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *args],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT))
        return proc

    def test_expected_error_concise_and_exit_1(self):
        # 无 config.json → RuntimeError → 简洁中文 + exit 1 + 无 traceback
        proc = self._run_cli("--check-api")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("错误：", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_argparse_error_exit_2(self):
        # --demo 与 --goal 互斥,argparse 处理,exit 2
        proc = self._run_cli("--demo", "--goal", "x")
        self.assertEqual(proc.returncode, 2)

    def test_expected_demo_ok(self):
        # demo 成功路径(写盘已由既有测试以 mock 隔离,这里只验证进程能启动并成功)
        proc = self._run_cli("--check")
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
