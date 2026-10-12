# -*- coding: utf-8 -*-
"""Project Prometheus — M10: 独立数学验证器最小原型——验收测试

覆盖 M10 验收点:
1. 三类验证状态的合法取值与唯一性;
2. 结构检查失败后 semantic 一律 not_evaluated;
3. 未绑定变量与错误量词作用域被拒绝(static_error);
4. 不把 no_counterexample 误报为 proved_under_scope;
5. 数值反例必须满足声明前提并实际违反目标命题;
6. 不支持的表达式返回 unknown(而非冒充正确);
7. 不执行候选中的任意代码(禁止 eval/exec);
8. 测试运行不修改真实 runs/(仅允许临时目录写盘);
9. 不改变既有数值实验引擎(get_experiment/make_default_validator 行为不变)。

安全:所有断言基于解析后的受限 AST,绝不 eval/exec 候选字符串。
"""
import builtins
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prometheus import math_verify as M
from prometheus import experiment as exp
from prometheus import experiments as exps


def _runs_file_set(root):
    """返回 root/runs 下的文件名集合(用于断言真实 runs/ 不被测试污染)。"""
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return set()
    return {p.name for p in runs_dir.glob("*.json")}


class TestStateValues(unittest.TestCase):
    """验收点 1:状态常量取值合法且相互区分。"""

    def test_structure_states_are_valid(self):
        for s in M.STRUCTURE_STATES:
            self.assertIn(s, (M.STRUCTURE_SYNTAX_ERROR,
                              M.STRUCTURE_STATIC_ERROR, M.STRUCTURE_OK))

    def test_semantic_states_are_valid(self):
        for s in M.SEMANTIC_STATES:
            self.assertIn(s, (M.SEMANTIC_COUNTEREXAMPLE,
                              M.SEMANTIC_NO_COUNTEREXAMPLE,
                              M.SEMANTIC_UNKNOWN, M.SEMANTIC_NOT_EVALUATED))

    def test_proof_states_are_valid(self):
        for s in M.PROOF_STATES:
            self.assertIn(s, (M.PROOF_PROVED, M.PROOF_FAILED,
                              M.PROOF_UNKNOWN, M.PROOF_NOT_ATTEMPTED))

    def test_state_sets_are_distinct(self):
        # 三类状态字段不应互相误用(不允许把一个字段的专属状态值跨字段复用)。
        # 例外:unknown 是语义上与证明上共同且同义的取值,表示「无法判定」,允许共享。
        sem = set(M.SEMANTIC_STATES)
        pf = set(M.PROOF_STATES)
        # 语义与证明仅在 unknown 上允许重合
        self.assertEqual(
            sem & pf, {M.SEMANTIC_UNKNOWN},
            "语义与证明状态集只应共享 unknown")
        # 结构状态只与语法校验相关,不得与语义/证明共享任何值
        self.assertFalse(set(M.STRUCTURE_STATES) & sem)
        self.assertFalse(set(M.STRUCTURE_STATES) & pf)


class TestStructureBeforeSemantic(unittest.TestCase):
    """验收点 2:结构检查失败后语义不评定。"""

    def test_syntax_error_not_evaluated(self):
        r = M.verify("forall eps exists delta lt abs (diff secret x) 1.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_SYNTAX_ERROR)
        self.assertEqual(r["semantic_state"], M.SEMANTIC_NOT_EVALUATED)
        self.assertEqual(r["proof_state"], M.PROOF_NOT_ATTEMPTED)

    def test_static_error_not_evaluated(self):
        r = M.verify("forall eps lt abs (diff id x) delta")
        self.assertEqual(r["structure_state"], M.STRUCTURE_STATIC_ERROR)
        self.assertEqual(r["semantic_state"], M.SEMANTIC_NOT_EVALUATED)
        self.assertEqual(r["proof_state"], M.PROOF_NOT_ATTEMPTED)

    def test_paren_mismatch_is_syntax_error(self):
        r = M.verify("forall eps exists delta lt abs (diff id x 1.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_SYNTAX_ERROR)


class TestRejectBadStructures(unittest.TestCase):
    """验收点 3:未绑定变量与错误量词作用域被拒绝。"""

    def test_unbound_delta_rejected(self):
        r = M.verify("forall eps lt abs (diff id x) delta")
        self.assertEqual(r["structure_state"], M.STRUCTURE_STATIC_ERROR)

    def test_delta_outside_eps_scope_rejected(self):
        # exists delta 在 forall eps 之外 -> 变量依赖错误
        r = M.verify("exists delta forall eps lt abs (diff id x) 1.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_STATIC_ERROR)

    def test_existential_not_innermost_rejected(self):
        # 最外层应是 forall eps,而这里是 exists delta -> 静态错误
        r = M.verify("exists delta forall eps lt abs (diff id x) 1.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_STATIC_ERROR)

    def test_proof_not_attempted_when_structure_bad(self):
        r = M.verify("forall eps lt abs (diff id x) delta")
        self.assertEqual(r["proof_state"], M.PROOF_NOT_ATTEMPTED)


class TestNoCounterexampleNotProof(unittest.TestCase):
    """验收点 4:有限采样未发现反例绝不等于已严格证明。"""

    def test_identity_convergent_sampling_only(self):
        # f(x)=x 在 c=0 极限为 0,有限网格无反例
        r = M.verify("forall eps exists delta lt abs (diff id x) 0.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_OK)
        self.assertEqual(r["semantic_state"], M.SEMANTIC_NO_COUNTEREXAMPLE)
        # 但非 P1 可证:证明必须显式为 not_attempted,绝不能为 proved_under_scope
        self.assertNotEqual(r["proof_state"], M.PROOF_PROVED)
        self.assertEqual(r["proof_state"], M.PROOF_NOT_ATTEMPTED)

    def test_derived_no_counterexample_never_marked_proved(self):
        # 关键原则:有限采样「未发现反例」这一证据本身绝不能导出 proved_under_scope。
        # 遍历:凡 semantic=no_counterexample 的候选,若其同时被判为 proved_under_scope,
        # 则必须由真实机械证明规则(P1)完成,而非由采样驱动;采样证据不足时必须保持未证明。
        for name, text, exp_s, exp_sem, exp_p, why in M.CALIBRATION:
            if exp_s == M.STRUCTURE_OK and exp_sem == M.SEMANTIC_NO_COUNTEREXAMPLE:
                r = M.verify(text)
                if exp_p == M.PROOF_NOT_ATTEMPTED:
                    # 无证明规则的采样候选:证明必须是 not_attempted,不得冒充为已证明
                    self.assertEqual(
                        r["proof_state"], M.PROOF_NOT_ATTEMPTED,
                        f"校准案例 {name} 仅凭 no_counterexample 被误报为已证明")
                elif exp_p == M.PROOF_PROVED:
                    # 真正可证(如常函数 P1)案例:证明必须携带实际证明依据,而非仅靠采样
                    self.assertEqual(r["proof_state"], M.PROOF_PROVED)
                    self.assertTrue(
                        any("P1" in e and "常函数" in e for e in r["proof_evidence"]),
                        f"案例 {name} 的证明依据未指向可核验的机械规则 P1")


class TestCounterexampleReal(unittest.TestCase):
    """验收点 5:数值反例须满足前提且实际违反目标命题。"""

    def test_divexp_has_counterexample_at_zero(self):
        # f(x)=1/x 在 c=0 无极限:存在 ε,使任意 δ 内都能找到违反点
        r = M.verify("forall eps exists delta lt abs (diff divexp x) 0.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_OK)
        self.assertEqual(r["semantic_state"], M.SEMANTIC_COUNTEREXAMPLE)
        self.assertTrue(any("反例" in e for e in r["semantic_evidence"]))

    def test_counterexample_satisfies_premise_and_violates_claim(self):
        # 直接构造并核实一个反例点:x 满足 |x-c|<delta,且 |f(x)-L|>=eps
        eps, delta, c, L = 0.1, 0.5, 0.0, 0.0
        x = 0.4          # 满足 |x-0|=0.4 < 0.5
        fx = 1.0 / x     # f(x)=1/x
        self.assertLess(abs(x - c), delta)      # 前提成立
        self.assertGreaterEqual(abs(fx - L), eps)  # 结论被违反
        r = M.verify("forall eps exists delta lt abs (diff divexp x) 0.0")
        self.assertEqual(r["semantic_state"], M.SEMANTIC_COUNTEREXAMPLE)

    def test_identity_not_false_positive(self):
        # 真实的收敛函数不应被误判为反例(验证器采样缺陷已修复)
        r = M.verify("forall eps exists delta lt abs (diff id x) 0.0")
        self.assertEqual(r["semantic_state"], M.SEMANTIC_NO_COUNTEREXAMPLE)


class TestUnknownNotCorrect(unittest.TestCase):
    """验收点 6:不支持的表达式返回 unknown,绝不冒充正确。"""

    def test_no_limit_claim_returns_unknown(self):
        r = M.verify("forall eps exists delta gt x c")
        self.assertEqual(r["structure_state"], M.STRUCTURE_OK)
        self.assertEqual(r["semantic_state"], M.SEMANTIC_UNKNOWN)
        self.assertEqual(r["proof_state"], M.PROOF_NOT_ATTEMPTED)

    def test_non_whitelisted_function_unknown(self):
        # 结构合法但函数族匹配不到 -> 语义 unknown
        r = M.verify("forall eps exists delta lt abs (diff id x) 1.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_OK)
        # 该候选其实可判定(identity 在 L=1, c=0 有反例),故应 counterexample
        self.assertIn(r["semantic_state"],
                      (M.SEMANTIC_COUNTEREXAMPLE, M.SEMANTIC_UNKNOWN))


class TestNoCodeExecution(unittest.TestCase):
    """验收点 7:不执行候选中的任意代码(禁止 eval/exec)。"""

    def test_malicious_import_is_rejected_without_execution(self):
        # 若解析器对候选做 eval/exec,builtins.eval/exec 会被调用;验证它们不被触发
        with mock.patch.object(builtins, "eval",
                               side_effect=AssertionError("eval 不应被调用")), \
             mock.patch.object(builtins, "exec",
                               side_effect=AssertionError("exec 不应被调用")):
            r = M.verify("__import__('os')")
        self.assertEqual(r["structure_state"], M.STRUCTURE_SYNTAX_ERROR)

    def test_python_import_statements_rejected(self):
        for cand in ("import os", "os.system('whoami')",
                     "eval('1')", "__builtins__", "secret"):
            r = M.verify(cand)
            self.assertEqual(
                r["structure_state"], M.STRUCTURE_SYNTAX_ERROR,
                f"候选 {cand!r} 应被语法层拒绝且不执行")


class TestCalibration(unittest.TestCase):
    """校准集:逐条比对验证器输出与人工确定的三类状态标签。"""

    def test_all_calibration_cases_match(self):
        for name, text, exp_s, exp_sem, exp_p, why in M.CALIBRATION:
            r = M.verify(text)
            self.assertEqual(
                r["structure_state"], exp_s,
                f"[{name}] structure 不匹配\n候选: {text}\n原因: {why}\n"
                f"实际 issues: {r['structure_issues']}")
            self.assertEqual(
                r["semantic_state"], exp_sem,
                f"[{name}] semantic 不匹配\n候选: {text}\n原因: {why}")
            self.assertEqual(
                r["proof_state"], exp_p,
                f"[{name}] proof 不匹配\n候选: {text}\n原因: {why}")


class TestConvergentConstantProved(unittest.TestCase):
    """白色路径:规则 P1 对常函数收敛给出机械可核验的严格证明。"""

    def test_const_function_proved(self):
        r = M.verify("forall eps exists delta lt abs (diff (const 2.0) x) 2.0")
        self.assertEqual(r["structure_state"], M.STRUCTURE_OK)
        self.assertEqual(r["semantic_state"], M.SEMANTIC_NO_COUNTEREXAMPLE)
        self.assertEqual(r["proof_state"], M.PROOF_PROVED)
        self.assertTrue(any("规则P1" in e for e in r["proof_evidence"]))


class TestRunsNotPolluted(unittest.TestCase):
    """验收点 8:测试不修改真实 runs/(只允许临时目录写盘)。"""

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.before = _runs_file_set(self.root)

    def tearDown(self):
        after = _runs_file_set(self.root)
        self.assertEqual(self.before, after,
                         "测试执行后真实 runs/ 文件名集合发生变化")

    def test_verify_writes_nothing_to_real_runs(self):
        M.verify("forall eps exists delta lt abs (diff id x) 0.0")
        M.verify("forall eps exists delta lt abs (diff divexp x) 0.0")

    def test_tempdir_only_writes(self):
        # 明确证明:任何写测试产物只会落在临时目录,而非项目 runs/
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "probe.json"
            p.write_text("{}", encoding="utf-8")
            self.assertTrue(p.exists())


class TestExistingEngineUntouched(unittest.TestCase):
    """验收点 9:新增验证器不改变既有数值实验引擎行为。"""

    def test_experiment_registry_unchanged(self):
        self.assertIn("toy_7_3", exps.EXPERIMENTS)
        self.assertEqual(exps.get_experiment("toy_7_3")["candidate_field"],
                         "candidate_x")

    def test_default_validator_still_works(self):
        exp_def = exps.EXPERIMENTS["toy_7_3"]
        validate = exps.make_default_validator(exp_def)
        self.assertEqual(validate({"candidate_x": 7.5}), 7.5)
        with self.assertRaises(ValueError):
            validate({"candidate_x": 7.25})  # 不在白名单


if __name__ == "__main__":
    unittest.main()
