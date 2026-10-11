# -*- coding: utf-8 -*-
"""本地端到端验证：model 策略通过真实(本地回环)HTTP 服务完成完整调用链。

目标（不调用真实 API、不需要 API Key、不联网、不产生费用、不安装依赖）：
用 Python 标准库在 127.0.0.1 上起一个 OpenAI-compatible 的本地模拟服务，
验证 run_research(strategy="model") 能完成：
  真实 HTTP 请求 -> 收到响应 -> 解析模型候选 -> 实验定义校验
  -> 本地评分函数评分 -> 候选/分数/轮次进入研究结果。

关键点：
  * 本测试直接向 run_research 传入一个 base_url 指向 127.0.0.1 的 cfg 字典，
    从而绕过 load_config / config.json / 真实环境变量，全程无真实凭据。
  * 不调用 save_run，运行结果不写盘，runs/ 及任何真实记录均不触碰。
  * 服务只绑定回环地址 127.0.0.1；测试结束在 finally 中 shutdown+close，
    即使断言失败也会正确清理。
  * 错误路径按项目现有约定断言 RuntimeError/ValueError，且错误信息不泄露
    模拟的 api_key。
"""
import json
import time
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue, Empty
from socket import socket

from prometheus.experiment import run_research


# ---------------------------------------------------------------------------
# 本地回环 HTTP 模拟服务
# ---------------------------------------------------------------------------
class _LocalModelServer:
    """在 127.0.0.1 上提供 OpenAI-compatible Chat Completions 的模拟服务。

    每个进入的请求从 responses 队列取一个 "responder"（可调用对象，
    签名 responder(request_json_body) -> (status:int, body:str, sleep:float)）。
    同时把每个请求的摘要记入 seen，供测试断言"确实发生了真实 HTTP 请求"。
    线程安全；服务线程为 daemon，测试结束由 contextmanager 负责 shutdown。
    """

    def __init__(self):
        self.responses = Queue()
        self.seen = []
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def _handler_factory(self):
        responses = self.responses
        seen = self.seen
        lock = self._lock

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                req = {
                    "path": self.path,
                    "method": self.command,
                    "auth": self.headers.get("Authorization"),
                    "body": raw.decode("utf-8", errors="replace"),
                }
                with lock:
                    seen.append(req)
                try:
                    responder = responses.get(timeout=5)
                except Empty:
                    self._write(500, "no responder configured", sleep=0)
                    return
                body_json = None
                try:
                    body_json = json.loads(req["body"]) if req["body"] else None
                except json.JSONDecodeError:
                    body_json = None
                status, body, sleep = responder(body_json)
                self._write(status, body, sleep)

            def _write(self, status, body, sleep):
                if sleep and sleep > 0:
                    time.sleep(sleep)
                if isinstance(body, (dict, list)):
                    payload = json.dumps(body).encode("utf-8")
                else:
                    payload = str(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args, **kwargs):
                pass

        return Handler

    def start(self):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_factory())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def base_url(self):
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ---- 便捷建造响应 ----
    # 注意：do_POST 从 responses 队列取出的必须是"可调用对象"（responder），
    # 签名 responder(body) -> (status:int, body:str|dict|list, sleep:float)。
    # 因此 ok/http500 必须像 slow 一样返回一个闭包，而不是直接返回元组；
    # 否则 handler 会 TypeError，进而 handle_error 关闭连接导致 RemoteDisconnected。
    @staticmethod
    def ok(content: str):
        # OpenAI-compatible 成功体：content 是模型回复文本。
        def responder(_body):
            return (200, {"choices": [{"message": {"content": content}}]}, 0)
        return responder

    @staticmethod
    def http500():
        def responder(_body):
            return (500, {"error": {"message": "internal boom"}}, 0)
        return responder

    @staticmethod
    def slow(sleep_seconds):
        # 返回一个睡眠足够久以触发调用方超时的响应。
        def responder(_body):
            return (200, {"choices": [{"message": {"content": "{\"candidate_x\": 3.0, \"reason\": \"late\"}"}}]}, sleep_seconds)
        return responder


def _make_cfg(server, timeout=15):
    return {
        "base_url": server.base_url,
        "model": "toy-fake",
        "api_key": "local-test-key",
        "timeout_seconds": timeout,
    }


class LocalModelServerTestCase(unittest.TestCase):
    def setUp(self):
        self.server = _LocalModelServer().start()
        self.addCleanup(self.server.stop)

    def queue_responses(self, *responders):
        for r in responders:
            self.server.responses.put(r)

    def run_model(self, goal="测试目标", rounds=1, timeout=15):
        cfg = _make_cfg(self.server, timeout=timeout)
        return run_research(cfg, goal, rounds, strategy="model")


# ---------------------------------------------------------------------------
# 成功链路
# ---------------------------------------------------------------------------
class TestSuccessPath(LocalModelServerTestCase):
    def test_real_http_request_reaches_local_server_and_research_succeeds(self):
        # 两个轮次都返回合法候选。
        self.queue_responses(
            self.server.ok('{"candidate_x": 2.0, "reason": "lower half"}'),
            self.server.ok('{"candidate_x": 3.0, "reason": "next"}'),
        )
        result = self.run_model("寻找接近 7.3 的参数", rounds=2)

        # ① 确实发生了 2 次真实 HTTP POST 请求到本地 loopback 服务。
        self.assertEqual(len(self.server.seen), 2)
        for req in self.server.seen:
            self.assertEqual(req["method"], "POST")
            self.assertEqual(req["path"], "/chat/completions")
            # 真实请求确实携带了模拟的 Authorization 头与 JSON 负载。
            self.assertEqual(req["auth"], "Bearer local-test-key")
            body = json.loads(req["body"])
            self.assertEqual(body["model"], "toy-fake")

        # ② 候选/分数/轮次正确进入结果。
        self.assertEqual(result["mode"], "api_research")
        self.assertEqual(result["rounds_requested"], 2)
        self.assertEqual(len(result["round_results"]), 2)
        r1, r2 = result["round_results"]
        self.assertEqual(r1["candidate_x"], 2.0)
        self.assertEqual(r1["score"], 1.0 / (1.0 + (2.0 - 7.3) ** 2))  # 本地评分函数
        self.assertEqual(r2["candidate_x"], 3.0)
        self.assertEqual(r2["score"], 1.0 / (1.0 + (3.0 - 7.3) ** 2))
        # ③ best_candidate 由本地评分决定。
        self.assertEqual(result["best_candidate"]["x"], 3.0)

    def test_min5_experiment_uses_own_validation_and_scoring(self):
        self.queue_responses(
            self.server.ok('{"candidate_x": 2.5, "reason": "near peak"}'),
            self.server.ok('{"candidate_x": 2.0, "reason": "adj"}'),
        )
        result = run_research(_make_cfg(self.server), "min_5 目标", 2,
                              experiment_id="min_5", strategy="model")
        r1, r2 = result["round_results"]
        self.assertEqual(r1["candidate_x"], 2.5)
        self.assertEqual(r1["score"], 1.0 / (1.0 + (2.5 - 2.6) ** 2))
        self.assertEqual(r2["candidate_x"], 2.0)
        self.assertTrue(result["best_candidate"]["x"] in (2.5, 2.0))

    def test_reason_text_is_recorded(self):
        self.queue_responses(self.server.ok('{"candidate_x": 1.5, "reason": "pilot"}'))
        result = self.run_model(rounds=1)
        self.assertEqual(result["round_results"][0]["reason"], "pilot")


# ---------------------------------------------------------------------------
# 异常/错误路径
# ---------------------------------------------------------------------------
class TestErrorPaths(LocalModelServerTestCase):
    def test_http_response_body_is_not_valid_json(self):
        # 服务返回 200 但响应体不是合法 JSON -> call_model 抛 RuntimeError。
        self.queue_responses(lambda _body: (200, "this is not json", 0))
        with self.assertRaises(RuntimeError):
            self.run_model(rounds=1)

    def test_model_content_missing_candidate_field(self):
        # 响应为合法 JSON，但模型内容是有 JSON 却没有 candidate_x 字段 -> 校验失败(ValueError)。
        self.queue_responses(self.server.ok('{"reason": "no candidate"}'))
        with self.assertRaises(ValueError):
            self.run_model(rounds=1)

    def test_candidate_out_of_allowed_range(self):
        # toy_7_3 白名单 0..10；返回 50.0 越界 -> 校验失败(ValueError)，不得作为有效结果。
        self.queue_responses(self.server.ok('{"candidate_x": 50.0, "reason": "wild"}'))
        with self.assertRaises(ValueError):
            self.run_model(rounds=1)

    def test_candidate_wrong_type(self):
        # 候选是字符串，不符合现有验证规则(有限数字) -> ValueError。
        self.queue_responses(self.server.ok('{"candidate_x": "abc", "reason": "oops"}'))
        with self.assertRaises(ValueError):
            self.run_model(rounds=1)

    def test_http_error_500(self):
        self.queue_responses(self.server.http500())
        with self.assertRaises(RuntimeError) as ctx:
            self.run_model(rounds=1)
        # HTTP 错误不被误判为成功；错误信息不得泄露模拟的模拟凭据。
        self.assertNotIn("local-test-key", str(ctx.exception))

    def test_timeout(self):
        # 服务回复前睡眠超过调用方超时(0.3s) -> socket.timeout -> RuntimeError。
        self.queue_responses(self.server.slow(sleep_seconds=1.0))
        with self.assertRaises(RuntimeError) as ctx:
            self.run_model(rounds=1, timeout=0.3)
        self.assertNotIn("local-test-key", str(ctx.exception))

    def test_connection_refused(self):
        # 指向一个已关闭的端口 -> URLError -> RuntimeError。
        # 先起一个临时服务拿一个空闲端口，再立刻停掉它。
        probe = _LocalModelServer().start()
        dead_url = probe.base_url
        probe.stop()
        from prometheus.experiment import run_research as _rr
        cfg = {"base_url": dead_url, "model": "m", "api_key": "k", "timeout_seconds": 1}
        with self.assertRaises(RuntimeError):
            _rr(cfg, "目标", 1, strategy="model")

    def test_errors_do_not_leak_api_key_across_paths(self):
        # 覆盖越界 + 格式两类异常，均不得把模拟 key 泄露进消息。
        self.queue_responses(
            self.server.ok('{"candidate_x": 99.0, "reason": "far"}'))
        with self.assertRaises(ValueError) as v:
            self.run_model(rounds=1)
        self.assertNotIn("local-test-key", str(v.exception))


# ---------------------------------------------------------------------------
# 服务与清理本身
# ---------------------------------------------------------------------------
class TestServerHygiene(unittest.TestCase):
    def test_server_binds_loopback_only(self):
        srv = _LocalModelServer().start()
        host, _port = srv._server.server_address[:2]
        srv.stop()
        self.assertEqual(host, "127.0.0.1")

    def test_no_http_server_left_listening_after_stop(self):
        srv = _LocalModelServer().start()
        port = srv._server.server_address[1]
        srv.stop()
        # 端口应已释放：连接应被拒绝。
        import urllib.error
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/x", timeout=1)
            self.fail("端口不应继续监听")
        except urllib.error.URLError:
            pass  # 预期的连接失败

    def test_stop_is_idempotent_and_clean(self):
        srv = _LocalModelServer().start()
        srv.stop()
        srv.stop()  # 第二次调用不应抛错

    def test_threads_are_daemon(self):
        srv = _LocalModelServer().start()
        self.assertTrue(srv._thread.daemon)
        srv.stop()


if __name__ == "__main__":
    unittest.main()
