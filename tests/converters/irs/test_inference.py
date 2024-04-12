import pytest

from b2s.const import VAR_NAME
from b2s.converters.irs.inference import (
    get_smallest_stream_exprs,
    infer_relational_signature,
    sketchify,
)
from b2s.lang import *
from b2s.rfs import RelationalSignature, RelationalSignatureExpr

_variance_loop_1 = ELam(("s", "x"), EBinOp(EVar("s"), BinOpKinds.ADD, EVar("x")))
_variance_loop_2 = ELam(
    ("sq_s", "x"),
    EBinOp(
        EVar("sq_s"),
        BinOpKinds.ADD,
        EBinOp(
            EBinOp(EVar("x"), BinOpKinds.SUB, EVar("avg")),
            BinOpKinds.POW,
            ENumber(2),
        ),
    ),
)
_variance_prog = ELam(
    ("xs",),
    ELet(
        "s",
        ECall(
            EVar("foldl"),
            (
                _variance_loop_1,
                ENumber(0),
                EStream("xs"),
            ),
        ),
        ELet(
            "avg",
            EBinOp(EVar("s"), BinOpKinds.DIV, ECall(EVar("len"), (EStream("xs"),))),
            ELet(
                "sq_s",
                ECall(
                    EVar("foldl"),
                    (
                        _variance_loop_2,
                        ENumber(0),
                        EStream("xs"),
                    ),
                ),
                EBinOp(
                    EVar("sq_s"),
                    BinOpKinds.DIV,
                    ECall(EVar("len"), (EStream("xs"),)),
                ),
            ),
        ),
    ),
)
_variance_prog_sketch = EPair(
    (
        ELet(
            "s",
            EUnknown("new_s"),
            ELet(
                "avg",
                EBinOp(EVar("s"), BinOpKinds.DIV, EUnknown("new_len")),
                ELet(
                    "sq_s",
                    EUnknown("new_sq_s"),
                    EBinOp(EVar("sq_s"), BinOpKinds.DIV, EUnknown("new_len")),
                ),
            ),
        ),
        EPair((EUnknown("new_s"), EUnknown("new_len"), EUnknown("new_sq_s"))),
    ),
)


@pytest.mark.parametrize(
    "expr, expected",
    [
        (EVar("x"), {}),
        (ECall(EVar("f"), (EVar("x"),)), {}),
        (
            ECall(EVar("f"), (EStream("xs"),)),
            {ECall(EVar("f"), (EStream("xs"),)): ("f", [])},
        ),
        (
            ECall(EVar("foldl"), (EVar("f"), ENumber(0), EStream("xs"))),
            {ECall(EVar("foldl"), (EVar("f"), ENumber(0), EStream("xs"))): (None, [])},
        ),
        (
            ELet(
                "sum",
                ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))),
                EVar("sum"),
            ),
            {
                ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))): (
                    "sum",
                    [],
                )
            },
        ),
        (
            _variance_prog,
            {
                ECall(
                    EVar(name="foldl"),
                    (_variance_loop_1, ENumber(value=0), EStream(name="xs")),
                ): ("s", []),
                ECall(
                    EVar(name="foldl"),
                    (_variance_loop_2, ENumber(value=0), EStream(name="xs")),
                ): (
                    "sq_s",
                    (
                        (
                            "s",
                            ECall(
                                EVar(name="foldl"),
                                (
                                    _variance_loop_1,
                                    ENumber(value=0),
                                    EStream(name="xs"),
                                ),
                            ),
                        ),
                        (
                            "avg",
                            EBinOp(
                                EVar(name="s"),
                                BinOpKinds.DIV,
                                ECall(EVar("len"), (EStream("xs"),)),
                            ),
                        ),
                    ),
                ),
                ECall(EVar("len"), (EStream(name="xs"),)): (
                    "len",
                    (
                        (
                            "s",
                            ECall(
                                EVar(name="foldl"),
                                (
                                    _variance_loop_1,
                                    ENumber(value=0),
                                    EStream(name="xs"),
                                ),
                            ),
                        ),
                        (
                            "avg",
                            EBinOp(
                                EVar(name="s"),
                                BinOpKinds.DIV,
                                ECall(EVar("len"), (EStream("xs"),)),
                            ),
                        ),
                        (
                            "sq_s",
                            ECall(
                                EVar(name="foldl"),
                                (
                                    _variance_loop_2,
                                    ENumber(value=0),
                                    EStream(name="xs"),
                                ),
                            ),
                        ),
                    ),
                ),
            },
        ),
    ],
)
def test_get_stream_exprs(expr, expected):
    assert get_smallest_stream_exprs(expr, []) == expected


def _mk_sig_base(expr_body: Expr, additional: dict[str, RelationalSignatureExpr]):
    prog = ELam(("xs",), expr_body)
    sig = RelationalSignature(
        {
            VAR_NAME.PREVIOUS_OUTPUT: RelationalSignatureExpr(expr_body),
        }
    )
    sig.update(additional)
    return (prog, sig)


@pytest.mark.parametrize(
    "expr_l, sig",
    [
        _mk_sig_base(EVar("x"), {}),
        _mk_sig_base(ECall(EVar("f"), (EVar("x"),)), {}),
        _mk_sig_base(
            ECall(EVar("f"), (EStream("xs"),)),
            {"prev_f": RelationalSignatureExpr(ECall(EVar("f"), (EStream("xs"),)))},
        ),
        _mk_sig_base(
            ECall(EVar("foldl"), (EVar("f"), ENumber(0), EStream("xs"))),
            {
                "prev_var0": RelationalSignatureExpr(
                    ECall(EVar("foldl"), (EVar("f"), ENumber(0), EStream("xs")))
                )
            },
        ),
        _mk_sig_base(
            ELet(
                "sum",
                ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))),
                EVar("sum"),
            ),
            {
                "prev_sum": RelationalSignatureExpr(
                    ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs")))
                )
            },
        ),
        # F(xs) = f(g(xs, foldr(+, 0, xs)))
        _mk_sig_base(
            ELet(
                "sum",
                ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))),
                ELet(
                    "g_val",
                    ECall(EVar("g"), (EStream("xs"), EVar("sum"))),
                    ECall(EVar("f"), (EVar("g_val"),)),
                ),
            ),
            {
                "prev_sum": RelationalSignatureExpr(
                    ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs")))
                ),
                "prev_g": RelationalSignatureExpr(
                    ECall(EVar("g"), (EStream("xs"), EVar("sum"))),
                    (
                        (
                            "sum",
                            ECall(
                                EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))
                            ),
                        ),
                    ),
                ),
            },
        ),
        # F(xs) = f(g(xs, foldr(+, 0, xs)))
        _mk_sig_base(
            ECall(
                EVar("f"),
                (
                    ECall(
                        EVar("g"),
                        (
                            EStream("xs"),
                            ECall(
                                EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))
                            ),
                        ),
                    ),
                ),
            ),
            {
                "prev_var0": RelationalSignatureExpr(
                    ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs")))
                ),
                "prev_g": RelationalSignatureExpr(
                    ECall(
                        EVar("g"),
                        (
                            EStream("xs"),
                            ECall(
                                EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))
                            ),
                        ),
                    )
                ),
            },
        ),
        _mk_sig_base(
            _variance_prog,
            {
                "prev_s": RelationalSignatureExpr(
                    ECall(
                        EVar(name="foldl"),
                        (_variance_loop_1, ENumber(value=0), EStream(name="xs")),
                    )
                ),
                "prev_sq_s": RelationalSignatureExpr(
                    ECall(
                        EVar(name="foldl"),
                        (_variance_loop_2, ENumber(value=0), EStream(name="xs")),
                    ),
                    (
                        (
                            "s",
                            ECall(
                                EVar(name="foldl"),
                                (
                                    _variance_loop_1,
                                    ENumber(value=0),
                                    EStream(name="xs"),
                                ),
                            ),
                        ),
                        (
                            "avg",
                            EBinOp(
                                EVar(name="s"),
                                BinOpKinds.DIV,
                                ECall(EVar("len"), (EStream("xs"),)),
                            ),
                        ),
                    ),
                ),
                "prev_len": RelationalSignatureExpr(
                    ECall(EVar("len"), (EStream("xs"),)),
                    (
                        (
                            "s",
                            ECall(
                                EVar(name="foldl"),
                                (
                                    _variance_loop_1,
                                    ENumber(value=0),
                                    EStream(name="xs"),
                                ),
                            ),
                        ),
                        (
                            "avg",
                            EBinOp(
                                EVar(name="s"),
                                BinOpKinds.DIV,
                                ECall(EVar("len"), (EStream("xs"),)),
                            ),
                        ),
                        (
                            "sq_s",
                            ECall(
                                EVar(name="foldl"),
                                (
                                    _variance_loop_2,
                                    ENumber(value=0),
                                    EStream(name="xs"),
                                ),
                            ),
                        ),
                    ),
                ),
            },
        ),
    ],
)
def test_infer_rel_sig(expr_l: ELam, sig):
    assert infer_relational_signature(expr_l) == sig


@pytest.mark.parametrize(
    "expr_body, expected_body, expected_unk_map",
    [
        (EVar("x"), EPair(elts=(EVar(name="x"), EPair(elts=tuple()))), {}),
        (
            ECall(EVar("f"), (EVar("x"),)),
            EPair(elts=(ECall(EVar("f"), (EVar("x"),)), EPair(elts=tuple()))),
            {},
        ),
        (
            ECall(EVar("f"), (EStream("xs"),)),
            EPair(elts=(EUnknown("new_f"), EPair(elts=(EUnknown("new_f"),)))),
            {
                "new_f": ECall(EVar("f"), (EStream("xs"),)),
            },
        ),
        (
            ELet(
                "sum",
                ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))),
                ELet(
                    "g_val",
                    ECall(EVar("g"), (EStream("xs"), EVar("sum"))),
                    ECall(EVar("f"), (EVar("g_val"),)),
                ),
            ),
            EPair(
                (
                    ELet(
                        "sum",
                        EUnknown("new_sum"),
                        ELet(
                            "g_val",
                            EUnknown("new_g"),
                            ECall(EVar("f"), (EVar("g_val"),)),
                        ),
                    ),
                    EPair(elts=(EUnknown(unk_id="new_sum"), EUnknown(unk_id="new_g"))),
                )
            ),
            {
                "new_sum": ECall(
                    EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))
                ),
                "new_g": ECall(EVar("g"), (EStream("xs"), EVar("sum"))),
            },
        ),
        (
            ECall(
                EVar("f"),
                (
                    ECall(
                        EVar("g"),
                        (
                            EStream("xs"),
                            ECall(
                                EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))
                            ),
                        ),
                    ),
                ),
            ),
            EPair(
                (
                    ECall(EVar("f"), (EUnknown(unk_id="new_g"),)),
                    EPair(elts=(EUnknown(unk_id="new_g"), EUnknown(unk_id="new_var0"))),
                )
            ),
            {
                "new_var0": ECall(
                    EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))
                ),
                "new_g": ECall(
                    EVar("g"),
                    (
                        EStream("xs"),
                        ECall(EVar("foldl"), (EVar("add"), ENumber(0), EStream("xs"))),
                    ),
                ),
            },
        ),
        (
            _variance_prog.body,
            _variance_prog_sketch,
            {
                "new_s": ECall(
                    EVar(name="foldl"),
                    (_variance_loop_1, ENumber(value=0), EStream(name="xs")),
                ),
                "new_sq_s": ECall(
                    EVar(name="foldl"),
                    (_variance_loop_2, ENumber(value=0), EStream(name="xs")),
                ),
                "new_len": ECall(EVar("len"), (EStream("xs"),)),
            },
        ),
    ],
)
def test_sketchify(
    expr_body: ELam, expected_body: Expr, expected_unk_map: dict[str, Expr]
):
    prog = ELam(("xs",), expr_body)
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
        f_python_expr=lambda _: True,
    ), f"sketch contains stream expressions: {pprint(sketch_prog)}"
    assert sketch_prog.params == tuple([*rel_sig, VAR_NAME.CURRENT_ELEMENT])
    assert sketch_prog.body == expected_body

    for unk_id, unk_spec in sketch.unknowns.items():
        assert unk_id in expected_unk_map
        assert unk_spec.equivalent_expr.expr == expected_unk_map[unk_id]
        assert unk_spec.io_examples is None
