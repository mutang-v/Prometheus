# -*- coding: utf-8 -*-
"""Project Prometheus — 历史经验驱动的候选选择策略 (v0.3: M4)

目标:在现有"模型生成候选"流程之外,提供一种清晰、可解释、可复现的
本地候选选择策略,利用同一实验过程中已获得的可信历史结果来确定下一步
搜索方向,而不依赖模型、不联网、不依赖时间或未固定的随机数。

设计约束(与整个项目一致):
  - 候选必须来自当前实验定义的 allowed_values(单一事实来源),绝不硬编码;
  - 使用当前实验的 candidate_field 读取历史候选;
  - 只把"已通过本地校验且由本地评分函数打分的记录"视为可信历史;历史条目若
    缺少候选字段、候选不在 allowed_values、或 score 非有限数字,一律忽略;
  - 不信任任何模型自报的 score(历史里的 score 必须是本地程序写入的有限数);
  - 优先选择尚未尝试过的合法候选,避免无意义重复;
  - 所有候选决策是确定性的:相同实验定义 + 相同历史 -> 相同结果;
  - 候选耗尽时明确返回 "exhausted" + candidate=None,绝不重复尝试伪装进展;
  - 不假装具备通用机器学习能力:当前只是可复现的轻量贪心搜索策略。

策略原理(可解释、可复现):
  1. 从历史有效条目中找出"已知最优候选"(本地评分最高者)。
  2. 在尚未尝试的合法候选中,选择与"已知最优"数值距离最近的那个(平局取较小值)。
  3. 无有效历史时,返回 allowed_values 中的确定性起点(首个候选)。
  4. 所有候选均已尝试时,返回 exhausted。

本模块为纯函数、无副作用,便于独立单元测试与未来接入 run_research。
M4 不改变 run_research 的默认(模型生成候选)行为,以免破坏旧接口兼容。
"""
import math


# 结果状态常量(便于测试与调用方分支判断)
STATUS_START = "start"          # 无有效历史,返回确定性起点
STATUS_OK = "ok"                # 正常返回一个推荐候选
STATUS_EXHAUSTED = "exhausted"  # 所有候选均已尝试,无剩余候选


def select_next_candidate(history, experiment):
    """根据可信历史与实验定义,返回下一个候选值的决策结果。

    参数:
      history:    可信历史记录列表。每条预期为 dict,含 experiment['candidate_field']
                  键(候选值)与 'score'(本地程序评分)。非 dict / 缺字段 /
                  候选非法 / score 非有限数字的条目会被忽略,不视为已尝试。
      experiment: 实验定义 dict,须含 candidate_field / allowed_values。

    返回 dict:
      {
        "candidate": float 或 None,   # None 仅在 exhausted 时出现
        "status": STATUS_*,
        "tried": [已尝试且有效的候选值,排序] ,
        "best_seen": float 或 None,   # 历史中已知最优候选值
        "reason": str,                # 人类可读理由
      }
    """
    field = experiment["candidate_field"]
    allowed = list(experiment["allowed_values"])
    allowed_set = set(allowed)

    tried = set()
    best_seen = None
    best_score = float("-inf")

    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        v = entry.get(field)
        s = entry.get("score")
        # 候选必须是数字(排除 bool),且在 allowed_values 内。
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        # 分数必须为有限数字(只信任本地程序写入的可信分数)。
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            continue
        if not math.isfinite(float(s)):
            continue
        x = float(v)
        if x not in allowed_set:
            continue
        tried.add(x)
        sc = float(s)
        if sc > best_score:
            best_score = sc
            best_seen = x

    untried = [v for v in allowed if v not in tried]

    if not untried:
        return {
            "candidate": None,
            "status": STATUS_EXHAUSTED,
            "tried": sorted(tried),
            "best_seen": best_seen,
            "reason": "所有候选均已尝试过,没有剩余候选。",
        }

    if best_seen is None:
        # 无有效历史:返回确定性起点(取当前未尝试的首个候选)。
        return {
            "candidate": untried[0],
            "status": STATUS_START,
            "tried": sorted(tried),
            "best_seen": None,
            "reason": "无有效历史,取 allowed_values 中的确定性起点。",
        }

    # 贪心:选择与已知最优数值距离最近的未尝试候选(距离相同取较小值)。
    next_candidate = min(untried, key=lambda v: (abs(v - best_seen), v))
    return {
        "candidate": next_candidate,
        "status": STATUS_OK,
        "tried": sorted(tried),
        "best_seen": best_seen,
        "reason": f"向已知最优 {best_seen} 的邻近未尝试候选推进。",
    }
