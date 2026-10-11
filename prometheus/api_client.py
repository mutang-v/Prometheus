import json
import os
import socket
import urllib.request
import urllib.error
from pathlib import Path

def load_config(root: Path):
    path = root / "config.json"
    if not path.exists():
        raise RuntimeError(
            "找不到 config.json。请先把 config.example.json 复制为 config.json，"
            "并根据 API 服务商文档填写 base_url 和 model。"
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_url", "model", "api_key_env"):
        if not cfg.get(key) or "YOUR_" in str(cfg.get(key)):
            raise RuntimeError(f"config.json 的 {key} 尚未填写。请不要猜接口地址或模型 ID。")
    env_name = cfg["api_key_env"]
    if not os.environ.get(env_name):
        raise RuntimeError(
            f"环境变量 {env_name} 未设置。请在当前 PowerShell 终端设置 API 密钥；"
            "不要把密钥写进 config.json 或发送给他人。"
        )
    cfg["api_key"] = os.environ[env_name]
    cfg["timeout_seconds"] = int(cfg.get("timeout_seconds", 45))
    return cfg

def call_model(cfg, user_text: str) -> str:
    base = cfg["base_url"].rstrip("/")
    url = base + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "你是一个谨慎的研究助手。只根据给定目标提出可验证的候选参数。不要输出代码。"},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cfg["api_key"]
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout_seconds"]) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API 返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接 API 失败：{exc.reason}") from exc
    except socket.timeout as exc:
        # urllib 超时抛 socket.timeout(即 TimeoutError,非 URLError),需单独处理。
        raise RuntimeError("连接 API 超时，请检查网络或增大 config 的 timeout_seconds。") from exc
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("API 响应格式不符合 OpenAI-compatible Chat Completions 预期。") from exc
