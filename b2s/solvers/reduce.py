# ruff: noqa: F403, F405

import logging
import os
import re
import textwrap
from numbers import Number
from shutil import which
from typing import Sequence

import pexpect as px
from joblib import Memory
from lark import Lark, Token, Transformer, UnexpectedToken

from b2s.synthesizers.qe_types import *

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

CACHE_DIR = ".cache"

reduce_memory = Memory(CACHE_DIR, verbose=0)

REDLOG_GRAMMAR = r"""

!comp_op: "<" | ">" | "=" | ">=" | "<=" | "<>"
!_unary_op: "+" | "-"
!_add_op: "+" | "-"
!_mult_op: "*" | "/" | "%"

?start: "{" or_test ("," or_test)* "}" | or_test
?or_test: and_test ("or" and_test)*
?and_test: not_test_ ("and" not_test_)*
?not_test_: "not" not_test_ -> not_test
            | comparison
?comparison: expr (comp_op expr)*

?expr: term (_add_op term)*
?term: factor (_mult_op factor)*
?factor: _unary_op factor | power

?power: atom_expr ("**" factor)*
    | atom_expr ("^" factor)*

?atom_expr: atom_expr "(" [arguments] ")" -> funccall
    | atom

?atom: name -> var
    | NUMBER -> number
    | "(" or_test ")"
    | "true" -> const_true
    | "false" -> const_false

arguments: argument ("," argument)*
?argument: or_test

?name: NAME
NAME: /[^\W\d]\w*/

%import common.SIGNED_NUMBER -> NUMBER
%import common.WS
%ignore WS
"""


class FormulaTransformer(Transformer):
    def const_true(self, _):
        return ValueTop()

    def const_false(self, _):
        return ValueBottom()

    def number(self, n):
        (n,) = n
        if isinstance(n, Number):
            return Constant(n)
        match n:
            case Token(type="NUMBER", value=n):
                return Constant(float(n))

    def comp_op(self, op):
        (op,) = op
        match op:
            case Token(type="EQUAL", value="="):
                return PredicateKinds.EQ
        return PredicateKinds(op)

    def start(self, args):
        return args[0] if len(args) == 1 else Disjunction.create(*args)

    def or_test(self, args):
        return Disjunction.create(*args) if len(args) > 1 else args[0]

    def and_test(self, args):
        return Conjunction.create(*args) if len(args) > 1 else args[0]

    def not_test(self, args):
        return Negation(args)

    def comparison(self, args):
        assert len(args) == 3, "Comparison must have 3 arguments"
        return PredicateApp(args[1], (args[0], args[2]))

    def expr(self, args):
        assert len(args) % 2 == 1, "Expression must have odd number of arguments"
        base = args[0]
        for op, arg in zip(args[1::2], args[2::2]):
            base = FunctionApp(FunctionKinds(op.value), (base, arg))
        return base

    def term(self, args):
        assert len(args) % 2 == 1, "Term must have odd number of arguments"
        base = args[0]
        for op, arg in zip(args[1::2], args[2::2]):
            base = FunctionApp(FunctionKinds(op.value), (base, arg))
        return base

    def power(self, args):
        base = args[0]
        for arg in args[1:]:
            base = FunctionApp(FunctionKinds.POW, (base, arg))
        return base

    def factor(self, args):
        if len(args) == 1:
            return args[0]

        assert len(args) == 2, "Factor must have 1 or 2 arguments"
        match args[0].value:
            case "-":
                op = FunctionKinds.NEG
            case "+":
                op = FunctionKinds.POS
            case _:
                raise ValueError("Invalid unary operator")
        return FunctionApp(FunctionKinds(op), (args[1],))

    def funccall(self, args):
        assert len(args) == 2, "Function call must have 2 arguments"
        return FunctionApp(args[0], args[1])

    def arguments(self, args):
        return tuple(args)

    def var(self, args):
        match args:
            case [Token("NAME", name)]:
                return Variable(name)

        raise ValueError("Invalid variable")


REDLOG_PARSER = Lark(
    REDLOG_GRAMMAR, start="start", parser="lalr", transformer=FormulaTransformer()
)


def remove_ansi_escape(txt: str) -> str:
    return ANSI_ESCAPE.sub("", txt)


@reduce_memory.cache
def run_reduce(cmds: list[str], timeout: int) -> list[str]:
    redcsl_path = os.getenv("REDUCE_PATH", which("redcsl"))
    if redcsl_path is None:
        raise FileNotFoundError("REDUCE_PATH not set and redcsl not found in PATH")


    cmds = ["off nat;", "rlset reals;"] + [cmd.strip() for cmd in cmds]
    r = px.spawn(
        redcsl_path,
        encoding="utf-8",
        dimensions=(1000, 1000),
        env={
            **os.environ,
            "TERM": "xterm-mono",
        },  # type: ignore
        maxread=20000,
        searchwindowsize=20000,
    )
    res = []

    if any(not cmd.endswith(";") for cmd in cmds):
        raise ValueError("All commands must end with ';'")

    r.expect(r"\d+: $", timeout=timeout)
    for line in cmds:
        for wrapped_line in textwrap.wrap(
            line, 512, break_long_words=False, break_on_hyphens=False
        ):
            r.sendline(wrapped_line)
            r.expect(r"\d+: $", timeout=timeout)

        resp_lines = [
            resp_line.strip()
            for resp_line in remove_ansi_escape(r.before).splitlines()
            if resp_line
        ]

        if not resp_lines or "$" != resp_lines[-1][-1] or "{}$" == resp_lines[-1]:
            continue

        resp_line = " ".join(resp_lines).removeprefix(line).removesuffix("$").strip()
        res.append(resp_line)

    return res


def solve_batch_eqns(
    eqns: Sequence[str], var: str, timeout: int = 20
) -> list[Expression]:
    cmds = [f"solve({eqn}, {var});" for eqn in eqns]
    reduce_output = run_reduce(cmds, timeout)

    result = []
    for line in reduce_output:
        try:
            f = REDLOG_PARSER.parse(line)
        except UnexpectedToken:
            logging.error(f"Failed to parse: {line}")
            continue
        match f:
            case PredicateApp(PredicateKinds.EQ, (Variable(var), arg)):
                match arg:
                    case FunctionApp(func=Variable("root_of")):
                        logging.error(f"Ignoring root_of: {line}")
                        continue
                result.append(arg)
            case _:
                logging.error(f"Unexpected output from Reduce: {line}")

    return result


def run_rlqe_variant(f_str: str, params: list[str], timeout: int, variant: str) -> Formula:
    assert variant in {"rlqe", "rlposqe"}, f"Invalid variant: {variant}"
    TEMP_FORMULA_NAME = "phi_temp"
    QE_COMMAND = f"{variant} {TEMP_FORMULA_NAME};"

    params_str = ", ".join(params)
    cmds = [
        f"{TEMP_FORMULA_NAME} := ex({{{params_str}}}, {f_str});",
        QE_COMMAND,
    ]
    try:
        res = run_reduce(cmds, timeout)
    except px.exceptions.TIMEOUT as ex:
        raise TimeoutError("Reduce timed out", ex)

    if not res:
        raise TimeoutError("Reduce timed out (no output)")

    # the transformer converts the parse tree to a formula
    return REDLOG_PARSER.parse(res[-1])  # type: ignore

def run_rlqe(f_str: str, params: list[str], timeout: int = 20) -> Formula:
    try:
        return run_rlqe_variant(f_str, params, timeout, "rlqe")
    except TimeoutError:
        return run_rlqe_variant(f_str, params, timeout, "rlposqe")
    return run_rlqe_variant(f_str, params, timeout, "rlposqe")
