# -*- coding: utf-8 -*-
"""Project Prometheus — v0.3 M5: 公平比较候选选择策略的对照实验。

在**同一个实验定义(候选域 + 评分函数 + 评估预算)**下,比较三种候选选择策略:
  1. fixed   固定顺序基线:按实验定义 allowed_values 的原始顺序逐项评估;
  2. history M4.1 历史驱动:严格复用 candidate_strategy.select_next_candidate 的
            既定规则(不做任何"为了更好结果"的特殊处理);
  3. random  随机策略:在仍未尝试的候选中均匀采样,固定随机种子可复现。

公平性(本模块的对照前提,见 M5 验收):
  - 相同实验:     三策略都由同一个 experiment_id 查 EXPERIMENTS,取同一份
                   allowed_values 与 score 函数;
  - 相同评估预算: 三策略共用同一个 budget(候选评估次数上限);
  - 相同评分函数: 三策略逐轮都调用 exp["score"](候选);
  - 区分"评估预算"与"候选域总大小":结果同时记录 budget 与
                   candidate_domain_size,eval_count = min(budget, 域大小) 对三策略一致;
  - 无未来信息:   每个策略在每轮只依赖"当前轮之前自己已评估"的信息
                   (fixed 按序扫描不回溯;history 只用自身历史;random 只用自身已尝试集合),
                   绝不读取尚未评估的分数或候选。

本模块不写盘、不联网、不调用模型,与真实 runs/ 无关。写盘由调用方(测试/脚本)
决定并一律走隔离临时目录,以遵守保护规则。

诚实说明(沿用 M4.1 限制):history 是向"历史最高分候选最近未尝试候选"推进的
贪心策略,不保证全局最优,也不保证每步分数都上升。它与 fixed / random 的相对
优劣取决于候选域几何与预算;本模块只提供可复现的公平对照数据,不作"普遍更优"
的宣称。单个随机种子不足以证明普遍优势,故 random 接受 seed,可对多个预先确定
的种子分别运行并汇总。
"""
import random

from .experiments import get_experiment, validate_experiment_definition
from .candidate_strategy import select_next_candidate, STATUS_EXHAUSTED


def run_strategy(experiment_id, strategy, budget, seed=None):
    """在单个实验与评估预算下运行某策略,返回结构化对照记录(不写盘)。

    参数:
      experiment_id: 已注册实验 ID(toy_7_3 / min_5)。
      strategy:      "fixed" / "history" / "random"。
      budget:        最大候选评估次数(评估预算;三者上限一致)。
      seed:          仅 random 使用(random.Random(seed));fixed/history 记录
                     以保持透明度,但对其无影响。
    返回 dict 含逐轮候选/分数/即时最佳、best_score、best_candidate、eval_count、
    candidate_domain_size、status(partial/exhausted)。
    非法策略名抛 ValueError。
    """
    exp = get_experiment(experiment_id)
    validate_experiment_definition(exp)
    candidate_field = exp["candidate_field"]
    score_f = exp["score"]
    allowed = list(exp["allowed_values"])

    if strategy == "fixed":
        candidates = _fixed_order(allowed, budget)
        detail = "固定顺序基线:按候选集合原始顺序逐项评估。"
    elif strategy == "history":
        candidates = _history_rule(exp, allowed, budget, candidate_field, score_f)
        detail = "历史驱动:复用 select_next_candidate 既定规则,不特殊处理。"
    elif strategy == "random":
        candidates = _random_sampling(allowed, budget, seed)
        detail = "随机策略:未尝试候选中均匀采样,固定种子可复现。"
    else:
        raise ValueError(f"未知策略：{strategy}。可用策略：fixed、history、random")

    # 逐轮记录候选与分数,并维护"截至本轮"的最佳(允许比较预算内的进展)。
    round_results = []
    best_score = float("-inf")
    best_candidate = None
    for idx, x in enumerate(candidates):
        score = score_f(x)
        if best_candidate is None or score > best_score:
            best_score = score
            best_candidate = x
        round_results.append({
            "round": idx + 1,
            candidate_field: x,
            "score": score,
            "best_score": best_score,
            "best_candidate": best_candidate,
        })

    domain_size = len(allowed)
    if not round_results:
        status = "empty"
    elif len(round_results) == domain_size:
        status = "exhausted"
    else:
        status = "partial"

    return {
        "strategy": strategy,
        "experiment_id": experiment_id,
        "seed": seed,
        "budget": budget,
        "candidate_domain_size": domain_size,
        "eval_count": len(round_results),
        "status": status,
        "round_results": round_results,
        "best_score": best_score if round_results else None,
        "best_candidate": best_candidate,
        "note": detail,
    }


def compare_strategies(experiment_id, budget, seed=None,
                       strategies=("fixed", "history", "random")):
    """对给定实验、预算,依次运行多个策略,返回便于对照的汇总结果。"""
    out = {}
    for s in strategies:
        out[s] = run_strategy(experiment_id, s, budget, seed=seed)
    return {
        "experiment_id": experiment_id,
        "budget": budget,
        "seed": seed,
        "strategies": out,
    }


# --------------------------------------------------------------------------- #
# 三种策略各自的候选选择定义(共享同一 budget / 候选域 / 无未来信息约束)
# --------------------------------------------------------------------------- #
def _fixed_order(allowed, budget):
    """固定顺序基线:按 allowed_values 原始顺序逐项评估,至多 budget 个。"""
    return allowed[:budget]


def _history_rule(exp, allowed, budget, candidate_field, score_f):
    """M4.1 历史驱动:严格按既定规则运行。

    逐轮调用 select_next_candidate(history, exp);返回候选后立即评估并追加到
    history,供下一轮使用;耗尽(candidate None / exhausted)立即停止,
    不重复尝试。与 experiment._run_history_strategy 行为一致。
    """
    history = []
    picks = []
    for _ in range(budget):
        decision = select_next_candidate(history, exp)
        if decision["candidate"] is None or decision["status"] == STATUS_EXHAUSTED:
            break
        picks.append(decision["candidate"])
        history.append({
            candidate_field: decision["candidate"],
            "score": score_f(decision["candidate"]),
        })
    return picks


def _random_sampling(allowed, budget, seed):
    """随机策略:在仍未尝试的候选中均匀采样(无放回),固定种子可复现。

    只依赖"自身已尝试集合",不使用任何分数或未来信息。
    """
    rng = random.Random(seed)
    untried = list(allowed)
    picks = []
    for _ in range(budget):
        if not untried:
            break
        idx = rng.randrange(len(untried))
        picks.append(untried.pop(idx))
    return picks
