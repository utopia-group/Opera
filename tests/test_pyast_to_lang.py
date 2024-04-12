import pytest

import b2s.ast_extra as ast
from b2s.const import VAR_NAME
from b2s.converters.irs.inference import (
    get_smallest_stream_exprs,
    infer_relational_signature,
    sketchify,
)
from b2s.lang import *
from b2s.pyast_to_lang import *
from b2s.rfs import RelationalSignature
from b2s.utils import parse_stmt


@pytest.mark.parametrize(
    "stmt_str, expected_map",
    [
        ("x = 0", {"x": 0}),
        ("y = True", {"y": True}),
        ("x, y = 1, 2", {"x": 1, "y": 2}),
    ],
)
def test_init_extractor(stmt_str, expected_map):
    stmt = parse_stmt(stmt_str)
    actual_map = ExtractInitializers.extract(stmt)
    assert actual_map == expected_map


def test_error_init_extractor():
    stmt = parse_stmt("x, y = 1")
    with pytest.raises(NotImplementedError):
        ExtractInitializers.extract(stmt)


@pytest.mark.parametrize(
    "py_src, func, stream_param, expected_str",
    [
        (
            r"""
def f(xs, y):
    s = 0
    for x in xs:
        s += x
    return s + y
""",
            "f",
            "xs",
            r"\xs y -> let S = {} in let S = foldl(\S x -> S{s = (s + x)}, S{s = 0}, xs) in (S[s] + y)",
        ),
        (
            r"""
def minmax(xs):
    mn = 1000
    l = 0
    for n in xs:
        if n < mn:
            mn = n
        l += 1
    return l, mn
""",
            "minmax",
            "xs",
            r"\xs -> let S = {} in let S = foldl(\S n -> S{mn = if (n < mn) then n else mn; l = (l + 1)}, S{mn = 1000; l = 0}, xs) in (S[l], S[mn])",
        ),
        (
            r"""
def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)
""",
            "mean",
            "xs",
            r"\xs -> let S = {} in let S = foldl(\S n -> S{s = (s + n)}, S{s = 0}, xs) in (S[s] / len(xs))",
        ),
        (
            r"""
def variance_twopass(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs)
""",
            "variance_twopass",
            "xs",
            r"\xs -> let S = {} in"
            r" let S = foldl(\S x -> S{s = (s + x)}, S{s = 0}, xs) in"
            r" let S = foldl(\S x -> S{sq_s = (sq_s + ((x - avg) ^ 2))}, S{avg = (s / len(xs)); sq_s = 0}, xs) in"
            r" (S[sq_s] / len(xs))",
        ),
    ],
)
def test_convert(py_src, func, stream_param, expected_str):
    src_ast = ast.parse(py_src)
    expr = PyAstToIntermLang(func, stream_param).visit(src_ast)
    assert pprint(expr) == expected_str


@pytest.mark.parametrize(
    "py_src, func, stream_param, expected_str",
    [
        (
            r"""
def f(xs, y):
    s = 0
    for x in xs:
        s += x
    return s + y
""",
            "f",
            "xs",
            r"\prev_out prev_s x__ -> (let S = {} in let S = S{s = 0} in let S = S{s = ??new_s} in (S[s] + y), (??new_s))",
        ),
        (
            r"""
def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)
""",
            "mean",
            "xs",
            r"\prev_out prev_s prev_len x__ -> (let S = {} in let S = S{s = 0} in let S = S{s = ??new_s} in (S[s] / ??new_len), (??new_s, ??new_len))",
        ),
        (
            r"""
def variance_twopass(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)
    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs)""",
            "variance_twopass",
            "xs",
            r"\prev_out prev_s prev_sq_s prev_len x__ -> (let S = {} in let S = S{s = 0} in let S = S{s = ??new_s} in let S = S{avg = (s / ??new_len); sq_s = 0} in let S = S{sq_s = ??new_sq_s} in (S[sq_s] / ??new_len), (??new_s, ??new_sq_s, ??new_len))",
        ),
    ],
)
def test_sketchify_2(py_src, func, stream_param, expected_str):
    src_ast = ast.parse(py_src)
    prog = PyAstToIntermLang(func, stream_param).visit(src_ast)

    rel_sig = infer_relational_signature(prog)
    sketch = sketchify(prog, rel_sig, False, False)
    sketch_prog = sketch.sketch
    assert isinstance(sketch_prog, ELam)
    assert fold_expr(
        sketch_prog,
        f_var=lambda _: True,
        f_stream=lambda _: False,
        f_int=lambda _: True,
        f_bool=lambda _: True,
        f_str=lambda _: True,
        f_binop=lambda left, _, right: left and right,
        f_call=lambda func, args: all(args),
        f_lam=lambda params, body: body,
        f_let=lambda name, expr, body: expr and body,
        f_ite=lambda cond, then, els: cond and then and els,
        f_nil=lambda: True,
        f_pair=lambda elts: all(elts),
        f_unk=lambda _: True,
        f_map=lambda _, exprs: all(exprs.values()),
        f_map_nil=lambda: True,
        f_map_get=lambda ex, _: ex,
        f_python_expr = lambda _: True,
    ), f"sketch contains stream expressions: {pprint(sketch_prog)}"
    assert pprint(sketch.sketch) == expected_str
