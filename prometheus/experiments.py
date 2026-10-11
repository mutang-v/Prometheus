# -*- coding: utf-8 -*-
"""Project Prometheus — 实验定义注册表 (v0.3: M1)

单一事实来源:每个实验的候选字段(candidate_field)、允许值(allowed_values)、
与程序实际执行的评分函数(score)都集中定义于此,供提示词构造、候选校验与评分共用,
从而避免提示词、校验器与评分函数各自维护一份可能漂移的副本。

本阶段(M1)仅含默认实验 toy_7_3,不实现通用多实验驱动。
实验定义采用轻量普通 dict + 普通函数,不引入抽象基类、第三方依赖或插件系统。
"""
import math

# 默认实验 ID(当前唯一实验)
DEFAULT_EXPERIMENT_ID = "toy_7_3"


def _score_toy_7_3(x):
    # 固定、可复现的玩具优化任务:寻找最接近 7.3 的参数。分数越高越好,最大为 1。
    return 1.0 / (1.0 + (x - 7.3) ** 2)


def _score_min_5(x):
    # 第二个演示实验:寻找最接近 2.6 的参数(候选 0.0..5.0, 步长 0.5)。分数越高越好,最大为 1。
    return 1.0 / (1.0 + (x - 2.6) ** 2)


EXPERIMENTS = {
    "toy_7_3": {
        "id": "toy_7_3",
        "description": "寻找最接近 7.3 的参数 x(候选范围 0.0..10.0, 步长 0.5)",
        # 模型返回 JSON 中代表"候选值"的键(提示词模板与校验共用的同一来源)
        "candidate_field": "candidate_x",
        # 允许的候选值集合(单一事实来源):0.0..10.0, 步长 0.5
        "allowed_values": [i / 2 for i in range(0, 21)],
        # 给模型看的评分说明(表述性字符串,程序不解析它;score 才是实际执行的评分函数)
        "scoring_hint": "本地评分函数是 score=1/(1+(x-7.3)^2)，越高越好。",
        # 程序实际执行的评分函数
        "score": _score_toy_7_3,
    },
    "min_5": {
        "id": "min_5",
        "description": "在 0.0 至 5.0 范围内，以 0.5 为步长寻找最接近 2.6 的候选值。",
        # 模型返回 JSON 中代表"候选值"的键(提示词模板与校验共用的同一来源)
        "candidate_field": "candidate_x",
        # 允许的候选值集合(单一事实来源):0.0..5.0, 步长 0.5
        "allowed_values": [i / 2 for i in range(0, 11)],
        # 给模型看的评分说明(表述性字符串,程序不解析它;score 才是实际执行的评分函数)
        "scoring_hint": "本地评分函数是 score=1/(1+(x-2.6)^2)，越高越好。",
        # 程序实际执行的评分函数
        "score": _score_min_5,
    },
}


def get_experiment(exp_id):
    """按 ID 返回实验定义;未知 ID 抛 KeyError。"""
    return EXPERIMENTS[exp_id]


# 实验定义必须具备的字段
REQUIRED_EXPERIMENT_KEYS = (
    "id", "description", "candidate_field",
    "allowed_values", "scoring_hint", "score",
)


def validate_experiment_definition(exp):
    """在运行研究流程前尽早校验实验定义完整性/合法性,避免运行中途报错。

    不完整(non-complete)或非法(non-valid)定义在此处抛 ValueError,给出清晰原因。
    校验内容:必需字段齐全、allowed_values 非空且全为有限数字、
    candidate_field 为非空字符串、score 可调用。
    """
    missing = [k for k in REQUIRED_EXPERIMENT_KEYS if k not in exp]
    if missing:
        raise ValueError(
            f"实验定义不完整，缺少字段: {', '.join(missing)}")
    allowed = exp["allowed_values"]
    if not isinstance(allowed, (list, tuple)) or len(allowed) == 0:
        raise ValueError("allowed_values 必须为非空列表或元组")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in allowed):
        raise ValueError("allowed_values 只能包含有限数字(不能为 bool/NaN/Infinity)")
    if not isinstance(exp["candidate_field"], str) or not exp["candidate_field"]:
        raise ValueError("candidate_field 必须为非空字符串")
    if not callable(exp["score"]):
        raise ValueError("score 必须为可调用评分函数")
    return exp



def make_default_validator(exp):
    """由 exp['candidate_field'] 与 exp['allowed_values'] 生成唯一校验器。

    统一从单一事实来源生成,各实验不必重复手写校验规则。
    校验器接收 _extract_json 解析出的 dict,返回合法候选值 float;非法则抛 ValueError。
    安全要求覆盖 v0.2-C 全部强度:
      - 拒绝 bool(即使它是 int 子类)、字符串、None、列表、字典等容器;
      - 显式拒绝 NaN 与正负 Infinity(有限性校验,非仅依赖白名单);
      - 拒绝不在 allowed_values 白名单中的值;缺字段按 None 处理同样拒绝。
    """
    field = exp["candidate_field"]
    allowed = set(exp["allowed_values"])

    def validate(parsed):
        raw = parsed.get(field)
        # 类型校验:仅接受真正的 int/float,排除 bool。
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(
                f"模型候选参数必须是有限数字，实际为 {type(raw).__name__}，已拒绝执行。")
        # 有限性校验:显式拒绝 NaN / ±Infinity。
        if not math.isfinite(raw):
            raise ValueError(
                f"模型候选参数必须是有限数字，实际为 {raw!r}，已拒绝执行。")
        x = float(raw)
        # 白名单校验:仅允许 exp['allowed_values'] 中的值。
        if x not in allowed:
            raise ValueError(f"模型候选参数 {x} 不在允许集合中，已拒绝执行。")
        return x

    return validate


# 默认实验定义(供 experiment.py 直接读取)
DEFAULT_EXPERIMENT = EXPERIMENTS[DEFAULT_EXPERIMENT_ID]
