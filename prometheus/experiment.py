# -*- coding: utf-8 -*-
"""Project Prometheus — 内置实验与研究流程 (v0.3: M2)

M2 将 run_research 通用化为真正的"由实验定义驱动"流程:
  - build_prompt(goal, experiment) 依据实验的 description/candidate_field/
    allowed_values/scoring_hint 生成提示词;
  - 候选校验使用 experiments.make_default_validator(exp) 统一生成;
  - 程序评分统一为 experiment['score'](candidate),绝不用模型自报分数;
  - run_research 新增可选 experiment_id(默认 'toy_7_3'),未知 ID 经
    get_experiment 报 KeyError,不静默回退到默认实验;
  - validate_experiment_definition 在运行前尽早校验定义完整性。

保留兼容接口与行为:score_parameter、run_demo、run_research(cfg,goal,rounds)、
_extract_json。本阶段(M2)不添加 min_5、不新增 CLI 参数。
"""
import json
import re
from datetime import datetime, timezone
from .api_client import call_model
from .experiments import (
    DEFAULT_EXPERIMENT, DEFAULT_EXPERIMENT_ID,
    make_default_validator, get_experiment, validate_experiment_definition,
)


def score_parameter(x: float) -> float:
    # 兼容 v0.1 外部接口:委托给默认实验的实际评分函数。
    return DEFAULT_EXPERIMENT["score"](x)


def build_prompt(goal: str, experiment) -> str:
    """由实验定义生成提示词。使用所选实验的 description / candidate_field
    / allowed_values / scoring_hint,而非任何硬编码值。"""
    candidate_field = experiment["candidate_field"]
    domain = experiment["allowed_values"]
    return (
        "研究目标：\n" + goal + "\n\n"
        "当前实验是一个受控的参数搜索演示。" + experiment["description"] + "\n"
        "可选参数只能从以下集合中选择：" + json.dumps(domain) + "\n"
        + experiment["scoring_hint"] + "\n"
        + f"请只返回 JSON：{{\"{candidate_field}\": 数字, \"reason\": \"简短可检验理由\"}}。\n"
        "不要输出代码，不要增加其他字段。"
    )


def run_demo(experiment=None):
    # 完全离线(demo 不调用 API):从实验定义的候选范围与评分函数本地评分。
    # 缺省使用默认实验 toy_7_3;可传入任意已注册实验对象以复用其候选域与评分。
    exp = experiment if experiment is not None else DEFAULT_EXPERIMENT
    # baseline 是固定初始化参考候选(仅用于 improved_over_baseline 的比较展示)。
    # 其候选字段取自所选实验的 candidate_field,分数由所选实验评分函数计算。
    # 2.0 对 toy_7_3(0..10,0.5) 与 min_5(0..5,0.5) 均为白名单内的合法候选。
    baseline_x = 2.0
    cand_field = exp["candidate_field"]
    # candidates/best_candidate 沿用历史公开键 "x"(被既有测试与 CLI 契约锁定)。
    candidates = [
        {"x": x, "score": exp["score"](x)} for x in exp["allowed_values"]
    ]
    best = max(candidates, key=lambda item: item["score"])
    return {
        "project": "Prometheus",
        "mode": "local_demo",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "objective": exp["description"],
        "baseline": {cand_field: baseline_x, "score": exp["score"](baseline_x)},
        "candidates": candidates,
        "best_candidate": best,
        "improved_over_baseline": best["score"] > exp["score"](baseline_x),
        "note": "仅用于验证实验、评分和记录闭环，不代表通用智能能力。"
    }


def _extract_json(text):
    # 接受裸 JSON 或 ```json ... ```，拒绝其他形式。
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON 对象。")
    return json.loads(cleaned[start:end+1])


def run_research(cfg, goal: str, rounds: int, experiment_id: str = DEFAULT_EXPERIMENT_ID,
                 strategy: str = "model", runs_dir=None):
    # 模型只能提出数字候选；程序负责客观评分与校验，绝不执行模型生成的代码。
    # strategy: "model"(默认,走模型生成候选) / "history"(离线本地历史驱动)。
    # runs_dir: 仅 history 策略使用。提供时,从该目录读取"与当前实验严格一致"
    #           的跨运行历史来播种候选选择;不提供则行为与 M4.1 完全一致。
    exp = get_experiment(experiment_id)          # 未知 ID -> KeyError，不静默回退
    validate_experiment_definition(exp)          # 定义不完整/非法尽早报错
    candidate_field = exp["candidate_field"]
    validate = make_default_validator(exp)
    score_f = exp["score"]

    from .history_store import experiment_fingerprint

    if strategy == "model":
        history = _run_model_strategy(cfg, goal, rounds, exp, validate, score_f)
        mode = "api_research"
    elif strategy == "history":
        history = _run_history_strategy(
            rounds, exp, validate, score_f, runs_dir=runs_dir)
        mode = "history_research"
    else:
        raise ValueError(
            f"未知策略：{strategy}。可用策略：model、history")

    # baseline 是固定初始化参考候选(仅用于 improved_over_baseline 比较展示)。
    # 候选字段取自所选实验的 candidate_field,分数由所选实验评分函数计算;
    # 值 2.0 对 toy_7_3 与 min_5 均为白名单内合法候选。best_candidate 沿用
    # 历史公开键 "x"(契约),round_results 使用实验自身的 candidate_field。
    baseline = {candidate_field: 2.0, "score": score_f(2.0)}
    best = max(history, key=lambda item: item["score"])
    return {
        "project": "Prometheus", "mode": mode, "strategy": strategy,
        # M7:新增 experiment_id 与 experiment_fingerprint,供跨运行历史读取做
        # 严格归因(旧记录无此字段,读取时会被跳过而非猜测归属)。
        "experiment_id": experiment_id,
        "experiment_fingerprint": experiment_fingerprint(exp),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "goal": goal, "baseline": baseline, "rounds_requested": rounds,
        "round_results": history,
        "best_candidate": {"x": best[candidate_field], "score": best["score"]},
        "improved_over_baseline": best["score"] > baseline["score"],
        "note": ("模型负责提出候选参数;本地固定评分器负责评估。当前仅为玩具任务。"
                 if strategy == "model" else
                 "本地历史驱动:候选由 select_next_candidate 依据可信历史选择,"
                 "分数由本地固定评分器评估;离线运行,不调用模型。")
    }


def _run_model_strategy(cfg, goal, rounds, exp, validate, score_f):
    """模型驱动:每轮由模型提出候选,程序校验并本地评分。"""
    candidate_field = exp["candidate_field"]
    history = []
    for idx in range(rounds):
        prompt = build_prompt(goal, exp)
        raw = call_model(cfg, prompt)
        parsed = _extract_json(raw)
        x = validate(parsed)
        score = score_f(x)                       # 程序评分，不用模型自报分数
        history.append({
            "round": idx + 1, candidate_field: x, "score": score,
            "reason": str(parsed.get("reason", ""))[:500]
        })
    return history


def _run_history_strategy(rounds, exp, validate, score_f, runs_dir=None):
    """历史驱动(离线):调用 select_next_candidate 依据可信历史选候选。

    候选/分数由本地策略决定并写入 history,完全离线,不调用 call_model,
    不要求 API key。耗尽后立即停止,不重复尝试;rounds 仍是最大轮数上限,
    不因候选耗尽而无限循环。

    runs_dir 提供时,先通过 history_store.load_history 读取"与当前实验定义严格
    一致"的跨运行可信历史来播种,使新一次运行能利用先前已保存的实验历史;不提供
    则行为与 M4.1 完全一致(不从任何目录读取)。
    读取到的历史 score 仅作 select_next_candidate 的"策略选择依据";每轮最终
    采用的候选仍由当前实验 score_f 重新本地评分,绝不直接沿用读到的历史分数。
    """
    from .candidate_strategy import (
        select_next_candidate, STATUS_EXHAUSTED,
    )
    from .history_store import load_history
    candidate_field = exp["candidate_field"]
    initial_history = []
    if runs_dir is not None:
        loaded, _skipped = load_history(runs_dir, exp["id"])
        initial_history = list(loaded)

    history = []
    for idx in range(rounds):
        # load_history 已返回去重后的可信历史;select_next_candidate 内部会用
        # 传入历史重建 tried 集合,保证已尝试候选不被重复评估。
        decision = select_next_candidate(initial_history + history, exp)
        if decision["candidate"] is None or decision["status"] == STATUS_EXHAUSTED:
            break
        x = validate({candidate_field: decision["candidate"]})
        score = score_f(x)                       # 本地评分,模型不参与
        history.append({
            "round": idx + 1, candidate_field: x, "score": score,
            "reason": decision["reason"],
        })
    return history
