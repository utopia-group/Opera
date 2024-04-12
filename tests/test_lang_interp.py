from typing import *

import pytest

from b2s.lang import *
from b2s.lang_interp import *


@pytest.mark.parametrize(
    "expr, expected_val, env",
    [
        (ENumber(1), VNumber(1), {}),
        (EBool(True), VBool(True), {}),
        (EString("str"), VString("str"), {}),
        (EVar("x"), VNumber(1), {"x": VNumber(1)}),
        (EVar("x"), VNumber(2), {"x": VNumber(2)}),
        (EVar("x"), VNumber(1), {"x": VNumber(1), "y": VNumber(2)}),
        (EVar("y"), VNumber(2), {"x": VNumber(1), "y": VNumber(2)}),
        (EVar("x"), VError("x is not defined"), {}),
        (EPair((ENumber(1), ENumber(2))), VPair((VNumber(1), VNumber(2))), {}),
        (to_expr_map({"x": 1, "y": 2}), to_value({"x": 1, "y": 2}), {}),
        (EVar("S"), to_value({"x": 1, "y": 2}), {"S": to_value_map({"x": 1, "y": 2})}),
        (EMapGet(EVar("S"), "x"), to_value(1), {"S": to_value_map({"x": 1, "y": 2})}),
        (
            EMapUpdate(
                EVar("S"),
                frozendict(
                    x=EBinOp(EVar("y"), BinOpKinds.ADD, to_const_expr(1)), y=EVar("x")
                ),
            ),
            to_value_map({"x": 5, "y": 5}),
            {"S": to_value_map({"x": 0, "y": 4})},
        ),
    ],
)
def test_eval(expr, expected_val, env) -> None:
    actual_val = eval_lang(expr, env)
    assert actual_val == expected_val, f"expected {expected_val}, got {actual_val}"


def test_int_binops() -> None:
    @dataclass
    class BinOpTest:
        binop: BinOpKinds
        left: int
        right: int
        expected: Value

        def check(self):
            assert self.expected == eval_binop(
                self.binop, VNumber(self.left), VNumber(self.right)
            )

    tests: List[BinOpTest] = [
        BinOpTest(BinOpKinds.ADD, 4, 3, VNumber(7)),
        BinOpTest(BinOpKinds.SUB, 4, 3, VNumber(1)),
        BinOpTest(BinOpKinds.MUL, 4, 3, VNumber(12)),
        BinOpTest(BinOpKinds.DIV, 4, 2, VNumber(2)),
        BinOpTest(BinOpKinds.DIV, 3, 2, VNumber(1.5)),
        BinOpTest(BinOpKinds.EQ, 4, 3, VBool(False)),
        BinOpTest(BinOpKinds.NEQ, 4, 3, VBool(True)),
        BinOpTest(BinOpKinds.LT, 4, 3, VBool(False)),
        BinOpTest(BinOpKinds.GT, 4, 3, VBool(True)),
        BinOpTest(BinOpKinds.LTE, 4, 3, VBool(False)),
        BinOpTest(BinOpKinds.GTE, 4, 3, VBool(True)),
    ]

    for test in tests:
        test.check()


def test_bool_binops() -> None:
    @dataclass
    class BinOpTest:
        binop: BinOpKinds
        left: bool
        right: bool
        expected: bool

        def check(self):
            assert VBool(self.expected) == eval_binop(
                self.binop, VBool(self.left), VBool(self.right)
            )

    tests: List[BinOpTest] = [
        # EQ tests
        BinOpTest(BinOpKinds.EQ, True, True, True),
        BinOpTest(BinOpKinds.EQ, True, False, False),
        BinOpTest(BinOpKinds.EQ, False, True, False),
        BinOpTest(BinOpKinds.EQ, False, False, True),
        # NEQ tests
        BinOpTest(BinOpKinds.NEQ, True, True, False),
        BinOpTest(BinOpKinds.NEQ, True, False, True),
        BinOpTest(BinOpKinds.NEQ, False, True, True),
        BinOpTest(BinOpKinds.NEQ, False, False, False),
        # AND tests
        BinOpTest(BinOpKinds.AND, True, True, True),
        BinOpTest(BinOpKinds.AND, True, False, False),
        BinOpTest(BinOpKinds.AND, False, True, False),
        BinOpTest(BinOpKinds.AND, False, False, False),
        # OR tests
        BinOpTest(BinOpKinds.OR, True, True, True),
        BinOpTest(BinOpKinds.OR, True, False, True),
        BinOpTest(BinOpKinds.OR, False, True, True),
        BinOpTest(BinOpKinds.OR, False, False, False),
    ]

    for test in tests:
        test.check()


def test_cons() -> None:
    @dataclass
    class BinOpTest:
        left: Value
        right: Value
        expected: Value

        def check(self):
            assert self.expected == eval_binop(BinOpKinds.CONS, self.left, self.right)

    tests = [
        BinOpTest(VNumber(1), VNumber(2), VCons(VNumber(1), VNumber(2))),
        BinOpTest(
            VNumber(1),
            VCons(VNumber(2), VNil()),
            VCons(VNumber(1), VCons(VNumber(2), VNil())),
        ),
    ]

    for test in tests:
        test.check()


def test_let() -> None:
    let1 = ELet("x", ENumber(1), EVar("x"))
    assert eval_lang(let1, {}) == VNumber(1)
    assert eval_lang(let1, {"x": VNumber(2)}) == VNumber(1)  # shadowing


def test_ite() -> None:
    assert eval_lang(EIte(EBool(True), ENumber(1), ENumber(2)), {}) == VNumber(1)
    assert eval_lang(EIte(EBool(False), ENumber(1), ENumber(2)), {}) == VNumber(2)
    assert isinstance(eval_lang(EIte(ENil(), ENumber(1), ENumber(2)), {}), VError)


def test_closure() -> None:
    closure1 = ELam(("x",), EBinOp(EVar("x"), BinOpKinds.ADD, ENumber(1)))
    assert eval_lang(ECall(closure1, (ENumber(10),)), {}) == VNumber(11)
    assert eval_lang(ECall(closure1, (ENumber(10),)), {"x": VNumber(1)}) == VNumber(11)

    closure_curry = ELam(("x", "y"), EBinOp(EVar("x"), BinOpKinds.ADD, EVar("y")))
    assert eval_lang(ECall(closure_curry, (ENumber(1), ENumber(2))), {}) == VNumber(3)

    capture_env = ELet("x", ENumber(1), ELam(("_",), EVar("x")))
    assert eval_lang(ECall(capture_env, (ENil(),)), {}) == VNumber(1)

    shadow_let = ELet("x", ENumber(1), ELam(("x",), EVar("x")))
    assert eval_lang(ECall(shadow_let, (ENumber(2),)), {}) == VNumber(2)


## Prelude functions


def test_trace(capsys: pytest.CaptureFixture) -> None:
    eval_lang(ECall(EVar("trace"), (EString("str"), ENil())), prelude)
    captured = capsys.readouterr()
    assert captured.out == "str\n"


def test_head() -> None:
    lst = EBinOp(
        ENumber(2), BinOpKinds.CONS, EBinOp(ENumber(1), BinOpKinds.CONS, ENil())
    )
    assert VNumber(2) == eval_lang(ECall(EVar("head"), (lst,)), prelude)

    empty = ENil()
    assert isinstance(eval_lang(ECall(EVar("head"), (empty,)), prelude), VError)


def test_foldl() -> None:
    lst = EBinOp(
        ENumber(2), BinOpKinds.CONS, EBinOp(ENumber(1), BinOpKinds.CONS, ENil())
    )
    add = ELam(("a", "b"), EBinOp(EVar("a"), BinOpKinds.ADD, EVar("b")))
    assert to_value(3) == eval_lang(
        ECall(EVar("foldl"), (add, ENumber(0), lst)), prelude
    )
