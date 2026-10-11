import argparse
import json
import sys
import traceback
from pathlib import Path

from prometheus.experiment import run_demo, run_research
from prometheus.experiments import EXPERIMENTS
from prometheus.api_client import load_config, call_model
from prometheus.storage import save_run

ROOT = Path(__file__).resolve().parent


def select_experiment(exp_id):
    """解析 --experiment;未知 ID 抛 ValueError(简洁中文并列出可用项)。

    落在 run_cli 的 EXPECTED_ERRORS 捕获范围内,输出简洁错误并以非零退出码返回,
    而不是向用户倾倒完整 traceback 或静默回退到默认实验。
    """
    if exp_id not in EXPERIMENTS:
        available = "、".join(EXPERIMENTS)
        raise ValueError(f"未知实验：{exp_id}。可用实验：{available}")
    return EXPERIMENTS[exp_id]


def main():
    parser = argparse.ArgumentParser(description="Project Prometheus v0.1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="检查本地 Python 环境")
    group.add_argument("--demo", action="store_true", help="运行免费本地演示实验")
    group.add_argument("--check-api", action="store_true", help="发送一次 API 测试请求")
    group.add_argument("--goal", type=str, help="研究目标（v0.1 仅支持内置参数搜索实验）")
    parser.add_argument("--rounds", type=int, default=1, help="研究轮数，限制为 1-3")
    parser.add_argument("--experiment", type=str, default="toy_7_3",
                        help="选择已注册实验（toy_7_3 / min_5），默认 toy_7_3")
    parser.add_argument("--strategy", type=str, default="model",
                        help="候选选择策略：model（默认，调用模型）/ history（离线本地历史驱动，无需 API key）")
    args = parser.parse_args()

    if args.check:
        print("Prometheus v0.1 本地环境检查")
        print(f"Python: {sys.version.split()[0]}")
        print(f"项目目录: {ROOT}")
        print(f"配置模板存在: {(ROOT / 'config.example.json').exists()}")
        print("本地检查通过；未调用 API，未修改系统配置。")
        return

    if args.demo:
        exp = select_experiment(args.experiment)
        result = run_demo(experiment=exp)
        path = save_run(ROOT, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n记录已保存：{path}")
        return

    if args.check_api:
        cfg = load_config(ROOT)
        response = call_model(cfg, "Reply with exactly: PROMETHEUS_API_OK")
        print("API 请求成功。模型回复：")
        print(response)
        return

    if args.goal:
        if not 1 <= args.rounds <= 3:
            parser.error("--rounds 只能是 1 到 3，防止意外消耗 API 额度。")
        # 提前解析实验:未知 ID 抛 ValueError(简洁中文+可用列表),由 run_cli 捕获。
        select_experiment(args.experiment)
        # history 策略完全离线,不读取配置、不要求 API key。
        if args.strategy not in ("model", "history"):
            raise ValueError(f"未知策略：{args.strategy}。可用策略：model、history")
        cfg = {} if args.strategy == "history" else load_config(ROOT)
        # M7:history 策略从项目 runs 目录加载跨运行历史;model 策略不传 runs_dir。
        runs_dir = ROOT / "runs" if args.strategy == "history" else None
        result = run_research(cfg, args.goal, args.rounds,
                              experiment_id=args.experiment,
                              strategy=args.strategy,
                              runs_dir=runs_dir)
        path = save_run(ROOT, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n记录已保存：{path}")

# 预期运行错误（配置缺失、API 配置错误、模型返回非法内容等）会抛出 RuntimeError/ValueError。
# 这里将其转换为简洁的中文提示并以非零退出码返回，避免向用户倾倒完整 traceback。
# 未预期的异常（Bug、环境故障等）仍按原样向上抛出，保留完整调试信息。
EXPECTED_ERRORS = (RuntimeError, ValueError)

def run_cli() -> int:
    try:
        main()
        return 0
    except EXPECTED_ERRORS as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_cli()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)
