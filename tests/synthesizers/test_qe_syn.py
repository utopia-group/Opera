import ast

import pytest
import z3

from b2s.const import VAR_NAME
from b2s.synthesizers.qe_syn import *
from b2s.synthesizers.qe_types import to_reduce_formula
from b2s.utils import parse_stmt, parse_term

compute_wp = ImperativeQEGrammarConstructor.compute_wp
compute_wp_stmt = ImperativeQEGrammarConstructor.compute_wp_stmt
qe_encode_rel_sig_formula = ImperativeQEGrammarConstructor.qe_encode_rel_sig_formula


def _formula_from_ast_str(s: str) -> Formula:
    return to_formula(parse_term(s))


def _expr_from_ast_str(s: str) -> Expression:
    return to_expression(parse_term(s))


@pytest.mark.parametrize(
    "src,depth,output,last_var",
    [
        (
            """\
def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)""",
            4,
            """\
def mean(xs):
    s = 0
    s += xs0
    s += xs1
    s += xs2
    s += xs3
    ___out___ = s / 4""",
            None,
        ),
        (
            """\
def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)""",
            4,
            """\
def mean(xs):
    s = 0
    s += xs0
    s += xs1
    s += xs2
    s += random_name
    ___out___ = s / 4""",
            "random_name",
        ),
    ],
)
def test_unroll_loop(src: str, depth: int, output: str, last_var: str | None):
    func = ast.parse(src).body[0]
    assert isinstance(func, ast.FunctionDef)

    unrolled = unroll_loop(func, depth, last_var)
    assert ast.unparse(unrolled) == output


@pytest.mark.parametrize(
    "expr_str,expected",
    [
        ("varname1", Variable(name="varname1")),
        ("varname2", Variable(name="varname2")),
        (
            "a + b",
            FunctionApp(
                func=FunctionKinds.ADD, args=(Variable(name="a"), Variable(name="b"))
            ),
        ),
        (
            "a + b + c",
            FunctionApp(
                func=FunctionKinds.ADD,
                args=(
                    FunctionApp(
                        func=FunctionKinds.ADD,
                        args=(Variable(name="a"), Variable(name="b")),
                    ),
                    Variable(name="c"),
                ),
            ),
        ),
        (
            "a + b * c",
            FunctionApp(
                func=FunctionKinds.ADD,
                args=(
                    Variable(name="a"),
                    FunctionApp(
                        func=FunctionKinds.MUL,
                        args=(Variable(name="b"), Variable(name="c")),
                    ),
                ),
            ),
        ),
    ],
)
def test_ast_to_expr(expr_str: str, expected: Expression) -> None:
    ast_expr = parse_term(expr_str)
    assert to_expression(ast_expr) == expected


@pytest.mark.parametrize(
    "stmt_str,expected",
    [
        (
            "a == b",
            PredicateApp(
                PredicateKinds.EQ,
                (Variable(name="a"), Variable(name="b")),
            ),
        ),
        (
            "a < b",
            PredicateApp(
                PredicateKinds.LT,
                (Variable(name="a"), Variable(name="b")),
            ),
        ),
    ],
)
def test_ast_to_formula(stmt_str: str, expected: Formula) -> None:
    ast_expr = parse_term(stmt_str)
    assert to_formula(ast_expr) == expected


@pytest.mark.parametrize(
    "stmt_str,post_cond,expected",
    [
        (
            "x = y + 1",
            _formula_from_ast_str("x == 1"),
            _formula_from_ast_str("y + 1 == 1"),
        ),
        (
            r"""if z == 1:
    x = y + 1
else:
    x = y - 1
""",
            _formula_from_ast_str("x == 114514"),
            Conjunction.create(
                Implication(
                    _formula_from_ast_str("z == 1"),
                    _formula_from_ast_str("y + 1 == 114514"),
                ),
                Implication(
                    Negation(_formula_from_ast_str("z == 1")),
                    _formula_from_ast_str("y - 1 == 114514"),
                ),
            ),
        ),
    ],
)
def test_compute_wp_stmt(stmt_str: str, post_cond: Formula, expected: Formula) -> None:
    assert compute_wp_stmt(parse_stmt(stmt_str), post_cond) == expected


@pytest.mark.parametrize(
    "prog_str,post_cond,expected",
    [
        (
            r"""\
x = y + 1
if x > 0:
    z = 1
else:
    z = -1
ret = z
""",
            _formula_from_ast_str("ret > 0"),
            Conjunction.create(
                Implication(
                    _formula_from_ast_str("y + 1 > 0"),
                    _formula_from_ast_str("1 > 0"),
                ),
                Implication(
                    Negation(_formula_from_ast_str("y + 1 > 0")),
                    _formula_from_ast_str("-1 > 0"),
                ),
            ),
        ),
    ],
)
def test_compute_wp(prog_str: str, post_cond: Formula, expected: Formula) -> None:
    stmts = ast.parse(prog_str).body
    assert compute_wp(stmts, post_cond) == expected


def test_to_z3_1():
    code = r"""\
x = y + 1
if x > 0:
    z = 1
else:
    z = -1
ret = z
"""
    stmts = ast.parse(code).body
    post_cond = _formula_from_ast_str("ret == -1")
    wp = compute_wp(stmts, post_cond)
    z3_expr = to_z3_formula(wp)
    solver = z3.Solver()
    print(z3.simplify(z3_expr))
    solver.add(z3_expr)
    assert solver.check() == z3.sat
    model = solver.model()
    print(model)


def test_to_z3_2():
    src = r"""
def mean(xs):
    s = 0
    for n in xs:
        s += n
    r = s / len(xs)
"""
    depth = 3

    func = ast.parse(src).body[0]
    assert isinstance(func, ast.FunctionDef)

    unrolled = unroll_loop(func, depth)
    assert isinstance(unrolled, ast.FunctionDef)
    body_stmts = unrolled.body
    post_cond = _formula_from_ast_str("ret == r")

    wp = compute_wp(body_stmts, post_cond)
    z3_expr = to_z3_formula(wp)
    print(z3.simplify(z3_expr))


@pytest.mark.parametrize(
    "src,rel_sig,unroll_depth,expected",
    [
        (
            r"""\
def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)""",
            ImperativeRelationalSignature(
                {"prev_s": "s", "prev_len": "len(xs)", "prev_out": VAR_NAME.RET_TEMP_VAR}
            ),
            4,
            "((prev_s = 0 + xs0 + xs1 + xs2 + xs3) and (prev_len = 4) and (prev_out = ((0 + xs0 + xs1 + xs2 + xs3) / 4)))",
        ),
        (
            r"""\
def variance_twopass(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)
    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs)""",
            ImperativeRelationalSignature(
                {"prev_s": "s", "prev_len": "len(xs)", "prev_sq_s": "sq_s"}
            ),
            4,
            r"((prev_s = 0 + xs0 + xs1 + xs2 + xs3) and (prev_len = 4) and "
            r"(prev_sq_s = 0 + (xs0 - (((0 + xs0 + xs1 + xs2 + xs3) / 4))) ** 2 + "
            r"(xs1 - (((0 + xs0 + xs1 + xs2 + xs3) / 4))) ** 2 + (xs2 - (((0 + xs0 + xs1 + xs2 + xs3) / 4))) ** 2 + "
            r"(xs3 - (((0 + xs0 + xs1 + xs2 + xs3) / 4))) ** 2))",
        ),
    ],
)
def test_qe_encode_rel_sig_formula(
    src: str,
    rel_sig: ImperativeRelationalSignature,
    unroll_depth: int,
    expected: str,
) -> None:
    func = ast.parse(src).body[0]
    assert isinstance(func, ast.FunctionDef)

    unrolled = unroll_loop(func, unroll_depth)
    print(ast.unparse(unrolled))
    assert isinstance(unrolled, ast.FunctionDef)
    formula = qe_encode_rel_sig_formula(unrolled, rel_sig, unroll_depth)

    print(formula)
    assert to_reduce_formula(formula) == expected


# @pytest.mark.parametrize(
#     "src,rel_sig,unk_map,unroll_depth,expected",
#     [
#         (
#             r"""\
# def mean(xs):
#     s = 0
#     for n in xs:
#         s += n
#     return s / len(xs)""",
#             ImperativeRelationalSignature(
#                 {"prev_s": "s", "prev_len": "len(xs)", "prev_out": OUT_KEYWORD}
#             ),
#             {"unk_1": "s", "unk_2": "len(xs)"},
#             4,
#             "((prev_s == 0 + xs0 + xs1 + xs2 ∧ prev_len == 3 ∧ prev_out == (0 + xs0 + xs1 + xs2) / 3) ∧ "
#             "unk_1 == 0 + xs0 + xs1 + xs2 + xs3 ∧ unk_2 == 4)",
#         ),
#     ],
# )
# def test_qe_encode_program(
#     src: str,
#     rel_sig: ImperativeRelationalSignature,
#     unk_map: dict[str, str],
#     unroll_depth: int,
#     expected: str,
# ) -> None:
#     func = ast.parse(src).body[0]
#     assert isinstance(func, ast.FunctionDef)

#     formula = qe_encode_program_plain(func, unroll_depth, rel_sig, unk_map)

#     print(formula)
#     assert str(formula) == expected
