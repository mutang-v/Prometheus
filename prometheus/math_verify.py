# -*- coding: utf-8 -*-
"""Project Prometheus — M10: 独立数学验证器最小原型

本模块是 M8/M9/M9.1 审查结论后的第一个"隔离的数学验证器"。
目标:在受限范围内,把"表达式结构检查 / 数学反例搜索 / 严格证明"三者明确分开,
以三个独立字段输出,绝不把有限采样未发现反例当作数学正确,也绝不用"结构良好"冒充证明。

本阶段边界(已实现 / 暂不支持):
- 已实现:受限 DSL 解析(仅白名单 token,绝不 eval/exec)、量词作用域与变量绑定检查、
  量词次序检查、受限函数族上的有限反例搜索、一条白名单内的机械严格证明规则(P1 常数收敛)。
- 暂不支持:任意函数的极限判定、极限唯一性/连续性/导数等任何推广、
  通用数学定理证明器、模型生成候选(本模块只接收受控结构化候选字符串)。

设计原则(对照 M9.1):
- structure_state / semantic_state / proof_state 为三个独立字段,互不混同。
- 有限数值采样未发现反例 -> 只记 no_counterexample,永不等于 proved_under_scope。
- 反例必须是"前提成立且结论被违反"才算 counterexample_found,判定用标准否命题
  "存在 eps>0,使得对任意 delta>0 都能找到 x (|x-c|<delta 且 |f(x)-L|>=eps)"。
- syntax_error / static_error 为结构性失败,结构未通过时 semantic 一律 not_evaluated。
"""

import math

# ---------------------------------------------------------------------------
# 1. 三类验证状态的合法取值常量
# ---------------------------------------------------------------------------

# structure_state:语法、变量绑定、类型、规定的量词结构
STRUCTURE_SYNTAX_ERROR = "syntax_error"
STRUCTURE_STATIC_ERROR = "static_error"
STRUCTURE_OK = "statically_ok"
STRUCTURE_STATES = (STRUCTURE_SYNTAX_ERROR, STRUCTURE_STATIC_ERROR, STRUCTURE_OK)

# semantic_state:数学反例搜索
SEMANTIC_COUNTEREXAMPLE = "counterexample_found"
SEMANTIC_NO_COUNTEREXAMPLE = "no_counterexample"
SEMANTIC_UNKNOWN = "unknown"
SEMANTIC_NOT_EVALUATED = "not_evaluated"
SEMANTIC_STATES = (SEMANTIC_COUNTEREXAMPLE, SEMANTIC_NO_COUNTEREXAMPLE,
                   SEMANTIC_UNKNOWN, SEMANTIC_NOT_EVALUATED)

# proof_state:白名单内的机械严格证明
PROOF_PROVED = "proved_under_scope"
PROOF_FAILED = "proof_failed"
PROOF_UNKNOWN = "unknown"
PROOF_NOT_ATTEMPTED = "not_attempted"
PROOF_STATES = (PROOF_PROVED, PROOF_FAILED, PROOF_UNKNOWN, PROOF_NOT_ATTEMPTED)


# ---------------------------------------------------------------------------
# 2. 受限 DSL:解析 -> AST(绝不对文本执行 eval/exec)
#    候选表达式是"数据",解析结果为嵌套 tuple,只做结构化的解释与校验。
# ---------------------------------------------------------------------------

# 受限 token 白名单(识别失败即语法错误;可解析符号与变量名被限定在此集合内)
# diff 为严格二元减法/函数作用: diff <expr> <expr>。
#   若左操作数是函数叶子 -> 表示求 f(右操作数);否则表示数值差(供前提如 |x-c|<δ 用)。
_BINARY_OPS = {"lt", "gt", "lte", "gte", "and", "or", "implies", "diff"}
_UNARY_OPS = {"abs"}
_QUANTIFIERS = {"forall", "exists"}
_QUANTIFIER_VARS = {"eps", "delta"}   # 量词变量唯一合法名
_FREE_VARS = {"x", "c"}               # 自由变量唯一合法名
_FUNCTION_NAMES = {"id", "divexp"}    # 受限函数族叶子(名称)
_FUNCTION_KEYWORD = "const"           # 常函数关键字,后接有限数字

_ATOM_TOKENS = (_BINARY_OPS | _UNARY_OPS | _QUANTIFIERS
                | _QUANTIFIER_VARS | _FREE_VARS
                | _FUNCTION_NAMES | {_FUNCTION_KEYWORD})


class _LexError(ValueError):
    """词法/语法解析错误。"""


def _tokenize(text):
    """把候选字符串切成 token 列表。只接受白名单符号/变量/数字/括号。"""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            num = text[i:j]
            try:
                val = float(num)
            except ValueError:
                raise _LexError(f"非法数字字面量: {num!r}")
            if not math.isfinite(val):
                raise _LexError(f"非有限数字: {num!r}")
            tokens.append(val)
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            if word not in _ATOM_TOKENS:
                raise _LexError(f"未知符号(不在白名单): {word!r}")
            tokens.append(word)
            i = j
            continue
        raise _LexError(f"非法字符: {ch!r}")
    return tokens


class _Parser:
    """基于 token 列表的递归下降解析,仅操作白名单 token,不执行任何代码。"""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        ast = self.parse_expr()
        if self.pos != len(self.tokens):
            raise _LexError("表达式后存在多余 token")
        return ast

    def parse_expr(self):
        tok = self.peek()
        if tok is None:
            raise _LexError("表达式为空")
        if tok == "(":
            self.next()
            expr = self.parse_expr()
            if self.peek() != ")":
                raise _LexError("缺少右括号 )")
            self.next()  # 消费右括号
            return expr
        if tok == ")":
            raise _LexError("意外的右括号 )")
        if tok in _QUANTIFIERS:
            return self.parse_quantifier()
        if tok in _BINARY_OPS:
            return self.parse_binary(tok)
        if tok in _UNARY_OPS:
            self.next()
            return ("abs", self.parse_expr())
        if tok == _FUNCTION_KEYWORD:  # const <数>
            self.next()
            num = self.next()
            if not isinstance(num, float):
                raise _LexError("const 后必须接一个有限数字")
            return ("f", ("const", num))
        if tok in _FUNCTION_NAMES:    # id / divexp
            self.next()
            return ("f", tok)
        if tok in _QUANTIFIER_VARS:
            self.next()
            return ("var", tok)
        if tok in _FREE_VARS:
            self.next()
            return ("var", tok)
        if isinstance(tok, float):
            self.next()
            return ("const", tok)
        raise _LexError(f"意外的起始 token: {tok!r}")

    def parse_quantifier(self):
        q = self.next()
        var = self.next()
        if var not in _QUANTIFIER_VARS:
            raise _LexError(f"量词变量必须是 eps/delta,实际 {var!r}")
        body = self.parse_expr()
        return (q, var, body)

    def parse_binary(self, op):
        self.next()
        left = self.parse_expr()
        right = self.parse_expr()
        return (op, left, right)


def parse_expression(text):
    """把受限候选字符串解析为 AST(tuple 树)。

    只接受白名单 token;任何未知符号/非法结构都会抛 _LexError。
    绝不 eval/exec:候选仅作为数据被解析和结构化解释。
    """
    tokens = _tokenize(text)
    if not tokens:
        raise _LexError("空表达式")
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# 3. 结构检查:语法、变量绑定、量词作用域、量词次序
#    这是完全可判定的静态检查,与数值/语义无关。
# ---------------------------------------------------------------------------

class _Scope:
    """跟踪量词作用域:已绑定变量集合。"""

    def __init__(self, parent=None):
        self.parent = parent
        self.bound = set()

    def is_bound(self, var):
        node = self
        while node is not None:
            if var in node.bound:
                return True
            node = node.parent
        return False


def _scope_has_eps(scope):
    node = scope
    while node is not None:
        if "eps" in node.bound:
            return True
        node = node.parent
    return False


def _structure_check(ast):
    """返回 (structure_state, issues 列表)。

    只做语法/绑定/作用域/量词次序的静态检查,完全可判定。
    """
    issues = []
    top_order = _traverse(ast, _Scope(), issues)
    if issues:
        return STRUCTURE_STATIC_ERROR, list(issues)
    # 量词次序:第一版要求最外层量词为 forall eps(外 forall,内 exists delta)。
    if top_order and top_order[0] != "eps":
        return STRUCTURE_STATIC_ERROR, [
            "量词次序错误:最外层量词必须是 forall eps(外 forall eps,内 exists delta)"]
    return STRUCTURE_OK, []


def _traverse(node, scope, issues):
    """遍历 AST 做绑定/作用域/类型检查,返回子树顶层量词绑定顺序。"""
    if isinstance(node, tuple) and node[0] in ("var", "const", "f"):
        if node[0] == "var":
            var = node[1]
            if var in _QUANTIFIER_VARS:
                if not scope.is_bound(var):
                    issues.append(f"未绑定量词变量: {var}")
            elif var not in _FREE_VARS:
                issues.append(f"非法自由变量: {var}")
        return []
    if isinstance(node, tuple) and node[0] in _QUANTIFIERS:
        var = node[1]
        if var in scope.bound:
            issues.append(f"变量重复绑定: {var}")
        sub = _Scope(scope)
        sub.bound.add(var)
        inner_order = _traverse(node[2], sub, issues)
        if node[0] == "exists" and var == "delta" and not _scope_has_eps(scope):
            issues.append(
                "变量依赖错误: delta 必须处于 forall eps 的作用域之内")
        return [var] + inner_order
    if isinstance(node, tuple) and node[0] in _BINARY_OPS:
        order = []
        for child in node[1:]:
            order.extend(_traverse(child, scope, issues))
        return order
    if isinstance(node, tuple) and node[0] in _UNARY_OPS:
        return _traverse(node[1], scope, issues)
    issues.append(f"未知 AST 节点: {node!r}")
    return []


# ---------------------------------------------------------------------------
# 4. 受限函数族求值(仅第一版白名单,用于反例搜索)
# ---------------------------------------------------------------------------

def _fname_of(func_node):
    """把函数叶子 node 规约为 ('const', k) 或字符串名。"""
    if isinstance(func_node, tuple) and func_node[0] == "f":
        return func_node[1]
    return None


def _eval_function(func_node, x, c):
    """对受限函数族在输入 x 处求值。仅支持 const(k)/id/divexp。"""
    fname = _fname_of(func_node)
    if isinstance(fname, tuple) and fname[0] == "const":
        return float(fname[1])
    if fname == "id":
        return float(x)
    if fname == "divexp":
        if x == 0:
            return float("inf")  # 未定义,记录为无穷(可用于构造无极限反例)
        return 1.0 / x
    raise ValueError(f"未知函数: {func_node!r}")


# ---------------------------------------------------------------------------
# 5. 声明提取 + 语义反例搜索
#    反例判定采用标准否命题:存在 eps>0,对所有 delta>0 都能找到 x
#    (|x-c|<delta 且 |f(x)-L|>=eps) -> counterexample_found。
#    有限采样未找到这样的 eps -> no_counterexample(不证明正确)。
# ---------------------------------------------------------------------------

_EPS_GRID = (1.0, 0.5, 0.1)
# δ 网格需覆盖足够小的量,以判断"是否存在能使前提内所有点满足结论的 δ";
# 采样排除开区间端点(|x-c|=δ 不属于 |x-c|<δ),避免边界误判。
_DELTA_GRID = (0.5, 0.1, 0.01, 0.001)
_C_VALUE = 0.0  # 第一版参考极限点固定取 0


def _extract_claim(ast):
    """从候选 AST 中查找极限声明,返回 (func_node, L) 或 None。

    匹配形如  lt(abs(diff(F, x, c)), L)  的核心断言:
    F 为函数叶子, |x| < delta 为前提,声明极限值 L。
    """
    result = []

    def walk(node):
        if isinstance(node, tuple) and node[0] == "lt":
            lhs, rhs = node[1], node[2]
            if isinstance(lhs, tuple) and lhs[0] == "abs":
                inner = lhs[1]
                if isinstance(inner, tuple) and inner[0] == "diff":
                    func = inner[1]
                    L = _const_of(rhs)
                    if isinstance(func, tuple) and func[0] == "f" and L is not None:
                        result.append((func, L))
            walk(lhs)
            walk(rhs)
        elif isinstance(node, tuple) and node[0] in _QUANTIFIERS:
            walk(node[2])
        elif isinstance(node, tuple) and node[0] in _BINARY_OPS:
            for child in node[1:]:
                walk(child)
        elif isinstance(node, tuple) and node[0] in _UNARY_OPS:
            walk(node[1])

    walk(ast)
    return result[0] if result else None


def _const_of(node):
    if isinstance(node, tuple) and node[0] == "const":
        return node[1]
    return None


def semantic_check(ast, structure_state):
    """对已通过结构检查的候选做语义反例搜索。

    返回 (semantic_state, evidence)。结构未通过时一律 not_evaluated。
    """
    if structure_state != STRUCTURE_OK:
        return SEMANTIC_NOT_EVALUATED, []

    claim = _extract_claim(ast)
    if claim is None:
        return SEMANTIC_UNKNOWN, ["无法从候选提取受限极限声明"]

    func_node, L = claim
    c = _C_VALUE
    try:
        for eps in _EPS_GRID:
            violated_for_all_delta = True
            for delta in _DELTA_GRID:
                violated = False
                # 在 (c-delta, c+delta) 开区间内采样 x(排除端点 i=±10)
                for i in range(-9, 10):
                    if i == 0:
                        continue
                    x = c + (delta * i) / 10.0
                    try:
                        fx = _eval_function(func_node, x, c)
                    except ValueError:
                        return SEMANTIC_UNKNOWN, ["未知函数族"]
                    if not math.isfinite(fx) or abs(fx - L) >= eps:
                        violated = True
                        break
                if not violated:
                    violated_for_all_delta = False
            if violated_for_all_delta:
                return SEMANTIC_COUNTEREXAMPLE, [
                    f"反例: 存在 ε={eps},使所有 δ∈{_DELTA_GRID} 采样内"
                    "都能找到 |x-c|<δ 且 |f(x)-L|≥ε 的 x"]
    except Exception:
        return SEMANTIC_UNKNOWN, ["语义判定过程异常"]

    return SEMANTIC_NO_COUNTEREXAMPLE, [
        "有限网格未发现反例(不构成数学正确性的证明)"]


# ---------------------------------------------------------------------------
# 6. 严格证明:仅对白名单内、可机械核验的规则开放 proved_under_scope
#    规则 P1(常数收敛):|const(k)-k|=0<eps 恒成立,前提用任意 delta>0。
#    "结构良好/绑定合法/量词次序正确" 本身绝不等于已证明。
# ---------------------------------------------------------------------------

def proof_check(ast, structure_state):
    """返回 (proof_state, evidence)。结构未通过时不做任何证明尝试。"""
    if structure_state != STRUCTURE_OK:
        return PROOF_NOT_ATTEMPTED, ["结构检查未通过,不做证明"]
    return _try_proof_p1(ast)


def _try_proof_p1(ast):
    """规则 P1:常数函数收敛。"""
    claim = _extract_claim(ast)
    if claim is None:
        return PROOF_NOT_ATTEMPTED, ["无候选极限声明,不做证明"]
    func_node, L = claim
    fname = _fname_of(func_node)
    if isinstance(fname, tuple) and fname[0] == "const":
        k = fname[1]
        if abs(k - L) < 1e-12:
            return PROOF_PROVED, [
                f"规则P1: 常函数 const({k}) 声明极限 L={L}=k,"
                "|f(x)-L|=|k-k|=0<eps 对任意 eps>0 恒成立(机械可核验)"]
        return PROOF_FAILED, [
            f"规则P1: 常函数 const({k}) 声明极限 L={L}≠k,不成立"]
    return PROOF_NOT_ATTEMPTED, ["非规则P1可证结构,不做证明尝试"]


# ---------------------------------------------------------------------------
# 7. 顶层统一入口:三字段验证
# ---------------------------------------------------------------------------

def verify(expression_text):
    """对受限候选执行三字段验证,返回独立的三字段 dict。"""
    try:
        ast = parse_expression(expression_text)
    except _LexError as exc:
        return {
            "structure_state": STRUCTURE_SYNTAX_ERROR,
            "semantic_state": SEMANTIC_NOT_EVALUATED,
            "proof_state": PROOF_NOT_ATTEMPTED,
            "structure_issues": [str(exc)],
            "semantic_evidence": [],
            "proof_evidence": [],
        }

    structure_state, structure_issues = _structure_check(ast)
    semantic_state, semantic_evidence = semantic_check(ast, structure_state)
    proof_state, proof_evidence = proof_check(ast, structure_state)

    return {
        "structure_state": structure_state,
        "semantic_state": semantic_state,
        "proof_state": proof_state,
        "structure_issues": structure_issues,
        "semantic_evidence": semantic_evidence,
        "proof_evidence": proof_evidence,
    }


# ---------------------------------------------------------------------------
# 8. 手工校准集(与生成逻辑完全无关,标签由人工确定;供测试逐条比对)
# ---------------------------------------------------------------------------

CALIBRATION = [
    # (名称, 候选文本, 期望 structure, 期望 semantic, 期望 proof, 人工解释)
    (
        "合法结构_常函数收敛(可证)",
        "forall eps exists delta lt abs (diff (const 2.0) x) 2.0",
        STRUCTURE_OK,
        SEMANTIC_NO_COUNTEREXAMPLE,
        PROOF_PROVED,
        "常函数 const(2.0) 声明极限 L=2.0,命中规则P1(严格可证);有限采样亦无反例。",
    ),
    (
        "合法结构_identity收敛(仅采样证据)",
        "forall eps exists delta lt abs (diff id x) 0.0",
        STRUCTURE_OK,
        SEMANTIC_NO_COUNTEREXAMPLE,
        PROOF_NOT_ATTEMPTED,
        "f(x)=x 在 c=0 声明 L=0:极限确实为 0,有限网格无反例;但非 P1 可证,"
        "故只记 no_counterexample 且 proof 未尝试(绝不能当已证明)。",
    ),
    (
        "合法结构_divexp无极限(反例)",
        "forall eps exists delta lt abs (diff divexp x) 0.0",
        STRUCTURE_OK,
        SEMANTIC_COUNTEREXAMPLE,
        PROOF_NOT_ATTEMPTED,
        "f(x)=1/x 在 c=0 无极限:存在 ε,使所有 δ 内都能找到 |x|<δ 且 |1/x|≥ε 的 x。",
    ),
    (
        "语法错误_未知符号",
        "forall eps exists delta lt abs (diff secret x) 1.0",
        STRUCTURE_SYNTAX_ERROR,
        SEMANTIC_NOT_EVALUATED,
        PROOF_NOT_ATTEMPTED,
        "'secret' 不在白名单 -> 词法拒绝,structure=syntax_error。",
    ),
    (
        "语法错误_括号不匹配",
        "forall eps exists delta lt abs (diff id x 1.0",
        STRUCTURE_SYNTAX_ERROR,
        SEMANTIC_NOT_EVALUATED,
        PROOF_NOT_ATTEMPTED,
        "token 流残损 -> 语法错误。",
    ),
    (
        "语法错误_非法字符",
        "forall eps exists delta lt abs (diff id x) 宿0.0",
        STRUCTURE_SYNTAX_ERROR,
        SEMANTIC_NOT_EVALUATED,
        PROOF_NOT_ATTEMPTED,
        "非 ASCII 非法字符'宿' -> 词法拒绝。",
    ),
    (
        "未绑定变量_delta无forall",
        "forall eps lt abs (diff id x) delta",
        STRUCTURE_STATIC_ERROR,
        SEMANTIC_NOT_EVALUATED,
        PROOF_NOT_ATTEMPTED,
        "forall 只绑定 eps,delta 未绑定即被使用 -> 静态错误。",
    ),
    (
        "量词次序错误_exists在外",
        "exists delta forall eps lt abs (diff id x) 1.0",
        STRUCTURE_STATIC_ERROR,
        SEMANTIC_NOT_EVALUATED,
        PROOF_NOT_ATTEMPTED,
        "最外层量词是 exists delta,违背 '外 forall eps 内 exists delta' -> 静态错误。",
    ),
    (
        "变量依赖错误_delta在eps作用域外",
        "lt abs (diff id x) delta",
        STRUCTURE_STATIC_ERROR,
        SEMANTIC_NOT_EVALUATED,
        PROOF_NOT_ATTEMPTED,
        "delta 未处于任何 forall eps 作用域 -> 变量依赖错误。",
    ),
    (
        "未知_无可提取极限声明",
        "forall eps exists delta gt x c",
        STRUCTURE_OK,
        SEMANTIC_UNKNOWN,
        PROOF_NOT_ATTEMPTED,
        "结构合法但无 lt(abs(diff(...))) 极限声明 -> semantic=unknown,不冒充正确。",
    ),
    (
        "仅采样证据_nonP1不可证",
        "forall eps exists delta lt abs (diff id x) 0.0",
        STRUCTURE_OK,
        SEMANTIC_NO_COUNTEREXAMPLE,
        PROOF_NOT_ATTEMPTED,
        "identity 收敛但不在 P1 白名单 -> 虽无反例,proof 仍 not_attempted(证明未完成)。",
    ),
]
