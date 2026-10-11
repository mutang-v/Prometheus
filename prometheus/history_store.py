# -*- coding: utf-8 -*-
"""Project Prometheus — 跨运行实验历史读取 (v0.3: M7)

职责:从已保存的运行记录 JSON 中,读取"与当前实验定义严格一致"的逐轮可信候选,
供 history 策略在新的程序运行中复用。与 storage.save_run(写入)职责分离:
本模块只负责"读取 + 归因 + 校验",不写任何文件。

安全与隔离设计(M7 约束):
  - 每条新运行记录带 experiment_id 与 experiment_fingerprint(由当前实验关键
    字段规范化后计算的稳定哈希,仅用标准库 hashlib,无第三方依赖)。
  - 读取历史时须同时匹配 experiment_id 与 experiment_fingerprint,二者任一
    缺失 / 不匹配 / 无法计算,一律跳过该记录,绝不用作任何实验的历史(不猜测)。
  - 历史字段来源 = 每条记录的逐轮 round_results[](逐轮候选),绝不信任
    best_candidate / 总评分等汇总字段。
  - 逐轮候选须满足:类型为真数字(排除 bool)、有限、且在目标实验的
    allowed_values 白名单内;score 为有限数字。不合格条目被丢弃。
  - 坏文件(损坏 JSON / 缺字段 / 实验不匹配 / 指纹不匹配 / 非法逐轮数据)采用
    可诊断的"跳过该记录"策略;单个坏文件不会阻断其余有效历史。
  - 本模块返回的 history 条目只作为 select_next_candidate 的"策略选择依据"。
    其 score 是历史记录里当时写入的本地评分,不直接当作当前实验重新测得的分数;
    当前实验对每轮最终采用候选仍会用当前 exp["score"] 重新本地评分(见 experiment)。
"""
import json
import hashlib
import math
from pathlib import Path

from .experiments import get_experiment


# 参与指纹计算的关键字段:实验标识 + 候选字段 + 候选域。
# 规范化后计算方便稳定(不引入第三方;无随机或时间依赖)。
FINGERPRINT_KEYS = ("id", "candidate_field", "allowed_values")


def experiment_fingerprint(experiment) -> str:
    """由实验定义的关键字段计算稳定指纹(SHA-256 hex)。

    规范化:allowed_values 排序后,把每个值转成规范的 float 表示,避免
    0.0 vs 0、小数表示差异导致指纹不稳定。只取上表中的关键字段,
    不把 description / scoring_hint(纯表述)纳入,以便相同候选域与评分
    语义的轻量定义仍可复用历史。
    """
    definition = {}
    for key in FINGERPRINT_KEYS:
        value = experiment[key]
        if key == "allowed_values":
            value = sorted(float(v) for v in value)
        definition[key] = value
    canonical = json.dumps(definition, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_entry(entry, experiment) -> bool:
    """校验单条逐轮候选是否可纳入可信历史。

    严格程度与 candidate_strategy.select_next_candidate 一致:
      候选为真数字(排除 bool)、有限、且在 experiment["allowed_values"] 内;
      score 为有限数字(只信任本地程序写入的可信分数)。
    只要一条记录里任一逐轮候选或 score 非法,即判定该记录不可信
    (由调用方丢弃),而非仅丢弃单条——以免用半可信历史误导搜索。
    """
    field = experiment["candidate_field"]
    allowed = set(experiment["allowed_values"])
    v = entry.get(field)
    s = entry.get("score")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if not isinstance(s, (int, float)) or isinstance(s, bool):
        return False
    if not math.isfinite(float(v)) or not math.isfinite(float(s)):
        return False
    if float(v) not in allowed:
        return False
    return True


def _record_experiment_id(record):
    """返回记录内明确声明的 experiment_id;缺失/非法返回 None。"""
    eid = record.get("experiment_id")
    if isinstance(eid, str) and eid:
        return eid
    return None


def _record_fingerprint(record):
    """返回记录内明确声明的 experiment_fingerprint;缺失/非法返回 None。"""
    fp = record.get("experiment_fingerprint")
    if isinstance(fp, str) and fp:
        return fp
    return None


def load_history(runs_dir, experiment_id):
    """从 runs 目录读取与当前实验定义严格一致的逐轮可信历史。

    参数:
      runs_dir:     包含 *.json 运行记录的目录(Path 或 str)。目录不存在或为空
                    时,安全返回空历史(不报错)。
      experiment_id:目标实验 ID(必须已注册;未知 ID 经 get_experiment 抛 KeyError)。
    返回:
      (history_entries, skipped):history_entries 为可复用历史列表,每条是
        {experiment["candidate_field"]: x, "score": s} 形式;skipped 为
        (path, reason) 列表,记录被跳过文件及原因,便于诊断。
    对单条记录的处理:仅当 experiment_id 与 fingerprint 均与当前实验匹配,
    且全部逐轮候选通过校验时,才作为该实验的可信历史纳入;否则跳过并记入
    skipped。缺失指纹的旧记录(如 M7 之前的 demo 记录)也被跳过,不猜测归属。
    """
    return load_history_detailed(runs_dir, experiment_id)


def load_history_detailed(runs_dir, experiment_id):
    """实际可用的历史读取主体(返回 (entries, skipped))。

    拆出独立函数,便于直接单测其输入/输出与坏记录处理(测试要求 12)。
    runs_dir 不存在或为空 -> 返回 ([], []) 的空历史,不报错。
    """
    runs_dir = Path(runs_dir)
    target_exp = get_experiment(experiment_id)
    target_fp = experiment_fingerprint(target_exp)
    field = target_exp["candidate_field"]

    entries = []
    skipped = []
    if not runs_dir.is_dir():
        return entries, skipped

    for path in sorted(runs_dir.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
            record = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            skipped.append((str(path), f"无法解析 JSON: {exc}"))
            continue

        if not isinstance(record, dict):
            skipped.append((str(path), "记录不是 JSON 对象"))
            continue

        rec_id = _record_experiment_id(record)
        rec_fp = _record_fingerprint(record)
        if rec_id is None:
            skipped.append((str(path), "缺少 experiment_id(可能是 M7 之前的旧记录),不猜测归属"))
            continue
        if rec_id != experiment_id:
            skipped.append((str(path), f"experiment_id 不匹配: {rec_id!r}"))
            continue
        if rec_fp is None:
            skipped.append((str(path), "缺少 experiment_fingerprint,无法确认定义一致"))
            continue
        if rec_fp != target_fp:
            skipped.append((str(path), "experiment_fingerprint 不匹配(定义可能已变化)"))
            continue

        rounds = record.get("round_results")
        if not isinstance(rounds, list) or not rounds:
            skipped.append((str(path), "缺少可用的 round_results[]"))
            continue

        record_entries = []
        ok = True
        for entry in rounds:
            if not isinstance(entry, dict):
                ok = False
                break
            if not _valid_entry(entry, target_exp):
                ok = False
                break
            record_entries.append({
                field: float(entry[field]),
                "score": float(entry["score"]),
            })
        if not ok:
            skipped.append((str(path), "存在非法逐轮候选/score,整条记录不可信,跳过"))
            continue
        entries.extend(record_entries)

    return entries, skipped
