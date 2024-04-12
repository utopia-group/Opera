import ast
import copy
import functools
import itertools
import random
import re
import signal
import warnings
from dataclasses import replace
from datetime import datetime
from fractions import Fraction
from itertools import chain, combinations
from typing import Callable, Generator, Optional, Sequence, TypeVar, Union

from frozendict import frozendict
from hypothesis import strategies as st
from sympy import fraction, lcm

import b2s.lang as ir
from b2s.const import MAGIC_ID, VAR_NAME
from b2s.lang import EStream, replace_variable
from b2s.rfs import RelationalSignatureExpr


class PyAstSizeCounter(ast.NodeVisitor):
    def __init__(self):
        self.size = 0

    def visit(self, node):
        self.size += 1
        return super().visit(node)

    def run(self, s):
        self.visit(parse_term(s))
        return self.size // 2


def ast_size(expr: ir.Expr) -> int:
    return ir.fold_expr(
        expr,
        f_var=lambda _: 1,
        f_stream=lambda _: 1,
        f_int=lambda _: 1,
        f_bool=lambda _: 1,
        f_str=lambda _: 1,
        f_binop=lambda left, _, right: left + right,
        f_call=lambda func, args: func + sum(args),
        f_lam=lambda _, body: body,
        f_let=lambda _, expr, body: expr + body,
        f_ite=lambda cond, then, els: cond + then + els,
        f_nil=lambda: 1,
        f_pair=lambda elts: sum(elts),
        f_unk=lambda _: 1,
        f_map=lambda e, updates: e + sum(updates.values()),
        f_map_nil=lambda: 1,
        f_map_get=lambda e, _: e,
        f_python_expr=lambda _: PyAstSizeCounter().run(_),
    )


def powerset(iterable, max_size=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(max_size or (len(s) + 1))
    )


def map_expr_to_partial_stream(
    expr: RelationalSignatureExpr,
) -> RelationalSignatureExpr:
    """
    Map the stream expression to a secondary name.
    Used to capture the partial loop value.
    """

    def _replace_fold_expr(ex: ir.Expr) -> ir.Expr:
        match ex:
            case ir.ECall(
                func=ir.EVar("foldl"), args=(f, s, EStream(VAR_NAME.INPUT_STREAM))
            ):
                return ir.ECall(
                    func=ir.EVar("foldl"),
                    args=[f, s, ir.EVar("xs'")],  # TODO: remove constant here
                )
            case ir.ECall(func=ir.EVar("len"), args=(EStream(VAR_NAME.INPUT_STREAM),)):
                return ex  # should not change b/c no explicit update to len

        raise ValueError(f"Unsupported expression {ex}")

    return replace(
        expr,
        expr=_replace_fold_expr(expr.expr),
    )


def find_denominator_lcm(fractions):
    # this sometimes fails -- no idea
    # denominators = [fraction(f)[1] for f in fractions]
    denominators = [
        Fraction(float(f)).limit_denominator().denominator for f in fractions
    ]
    return lcm(denominators)


def sympy_const_to_number(v) -> int | float:
    if isinstance(v, float) or isinstance(v, int):
        return v
    if v.is_Float:
        return float(v)
    elif v.is_Integer:
        return int(v)
    elif v.is_infinite:
        return float("inf")
    raise ValueError(f"Unsupported constant {v}")


def generate_float_tuples(tl: int) -> tuple[float, ...]:
    return tuple(random.sample(range(1, 100), tl))


def generate_bid_object(num_obj: int) -> tuple[frozendict, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tuple(
            frozendict(
                {
                    "auction": random.randint(1, 5),
                    "bidder": random.randint(1, 5),
                    "dateTime": datetime.now(),
                    "price": random.randint(1, 5),
                    "extra": random.randint(1, 5),
                    "url": "http://url/to/some/path/here/and/further/here?query=string&channel_id=other",
                    "channel": random.choice("apple google facebook baidu".split()),
                }
            )
            for _ in range(num_obj)
        )


def generate_person_object(num_obj: int) -> tuple[frozendict, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tuple(
            frozendict(
                {
                    "id": random.randint(1, 5),
                    "state": random.choice(["OR", "ID", "CA", "TX", "WA", "NM", "NH"]),
                    "city": random.choice(
                        ["Portland", "Seattle", "Austin", "Albuquerque"]
                    ),
                }
            )
            for _ in range(num_obj)
        )


def generate_auction_object(num_obj: int) -> tuple[frozendict, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tuple(
            frozendict(
                {
                    "id": random.randint(1, 5),
                    "seller": random.randint(1, 5),
                    "category": random.randint(1, 5),
                    "expires": datetime.now(),
                    "dateTime": datetime.now(),
                    "itemName": "n/a",
                    "description": "n/a",
                    "initialBid": random.randint(1, 5),
                    "reserve": random.randint(1, 5),
                    "extra": random.randint(1, 5),
                }
            )
            for _ in range(num_obj)
        )


def generate_side_input_object(num_obj: int) -> tuple[frozendict, ...]:
    return tuple(
        frozendict(
            {
                "key": random.randint(1, 5),
                "value": random.randint(1, 5),
            }
        )
        for _ in range(num_obj)
    )


def throw_(ex):
    raise ex


def flip(func):
    @functools.wraps(func)
    def newfunc(x, y):
        return func(y, x)

    return newfunc


def foldr(func, acc, xs):
    return functools.reduce(flip(func), reversed(xs), acc)


def get_variables_in_scope(node: ast.stmt | ast.expr) -> set[str]:
    """Returns a list of variables in the scope of the given node."""
    vs = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            vs.add(n.id)
    return vs


def get_docstring(m: ast.mod) -> Optional[str]:
    """Returns the docstring of a module."""
    match m:
        case ast.Module(body=body) if body:
            match body[0]:
                case ast.Expr(value=ast.Str(s=s)):
                    return s
    return None


def parse_stmt(expr: str) -> ast.stmt:
    """Parses an expression."""
    module = ast.parse(expr, mode="exec")
    assert (
        isinstance(module, ast.Module) and len(module.body) == 1
    ), "Invalid expression"
    return module.body[0]


def parse_term(term: str) -> ast.expr:
    """Parses a term."""
    return ast.parse(term, mode="eval").body


def extract_left_values(expr: ast.stmt) -> list[str]:
    """Extracts the left values of an assignment expression."""
    results: list[str] = []
    for node in ast.walk(expr):
        match node:
            case ast.Assign(targets=targets):
                results.extend(
                    itertools.chain.from_iterable(map(extract_top_var_decls, targets))
                )
            case ast.AnnAssign(target=target) | ast.AugAssign(target=target):
                results.extend(extract_top_var_decls(target))

    return list(dict.fromkeys(results).keys())


def extract_top_var_decls(node: ast.expr) -> set[str]:
    match node:
        case ast.Name(id=name):
            return set([name])
        case ast.Tuple(elts=elts):
            return set.union(*(extract_top_var_decls(elt) for elt in elts))
        case ast.Subscript(value=ast.Name(id=name)):
            return set([name])
        case _:
            return set()


T = TypeVar("T", bound=ast.AST)


def rewrite_variable_name(get_name: Callable[[str], Optional[str]], node: T) -> T:
    """Rewrites a variable name (including return statement) in a statement."""

    class RewriteName(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            var_name = get_name(node.id)
            if var_name is not None:
                return ast.Name(var_name, node.ctx)
            return node

        def visit_Return(self, node: ast.Return) -> Union[ast.Return, ast.stmt]:
            """Rewrites the return statement if `get_name` returns a new name."""
            assert node.value is not None
            node_value = self.visit(node.value)
            var_name = get_name(MAGIC_ID.RETURN)
            if var_name is not None:
                return ast.fix_missing_locations(
                    ast.Assign(
                        value=node_value, targets=[ast.Name(var_name, ast.Store())]
                    )
                )
            return node

    return RewriteName().visit(copy.deepcopy(node))


def rewrite_call(
    node: ast.stmt, get_call: Callable[[ast.Call], Optional[ast.expr]]
) -> ast.stmt:
    """Rewrites a function application in a statement."""

    class RewriteCall(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.expr:
            new_call_term = get_call(node)
            if new_call_term is not None:
                return new_call_term
            return node

    return RewriteCall().visit(copy.deepcopy(node))


def rewrite_subscript(
    get_subscript: Callable[[ast.Subscript], ast.expr], node: ast.stmt
) -> ast.stmt:
    """Rewrites a subscript in a statement."""

    class RewriteSubscript(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            new_subscript = get_subscript(node)
            if new_subscript is not None:
                return new_subscript
            return node

    return RewriteSubscript().visit(copy.deepcopy(node))


def swap_prefix(s: str, old: str, new: str):
    """Swaps the prefix of s from old to new."""
    assert s.startswith(old), f"{s} does not start with {old}"
    return new + s[len(old) :]


def check_is_indexed_loop(loop_code: ast.For) -> bool:
    match loop_code:
        case ast.For(iter=ast.Call(func=ast.Name(id="range")), target=decl, body=body):
            decl_vars = get_variables_in_scope(decl)
            iter_vars = get_variables_in_scope(loop_code.iter)
            for node in itertools.chain.from_iterable(ast.walk(n) for n in body):
                match node:
                    case ast.Subscript(value=ast.Name(id=name), slice=ast.Name(id=idx)):
                        if idx in decl_vars and name in iter_vars:
                            return True

            return False
        case _:
            return False


def check_is_foreach_loop(loop_code: ast.For) -> bool:
    match loop_code:
        case ast.For(iter=ast.Name(id=VAR_NAME.INPUT_STREAM)):
            return True
        case _:
            return False


def check_if_new_decl(stmt: ast.stmt) -> Optional[bool]:
    """Checks if the declaration is a new declaration."""
    match stmt:
        case ast.Assign(targets=targets, value=value):
            target_vars = list(
                itertools.chain.from_iterable(extract_top_var_decls(t) for t in targets)
            )
            value_vars = get_variables_in_scope(value)
            assert target_vars, "No target variables in an assignment"
            return not any(v in value_vars for v in target_vars)
        case ast.AugAssign():
            return False

    return None


def filter_statements_by_vars(
    stmts: list[ast.stmt], variables: set[str]
) -> list[ast.stmt]:
    """Filters statements by variables."""
    return [stmt for stmt in stmts if get_variables_in_scope(stmt) & variables]


def find_division_terms(stmt: ast.stmt) -> list[tuple[ast.expr, ast.expr]]:
    terms = []
    for node in ast.walk(stmt):
        match node:
            case ast.BinOp(left=left, right=right, op=ast.Div()):
                terms.append((left, right))
    return terms


K, V = TypeVar("K"), TypeVar("V")


def subdict(d: dict[K, V], keys: Sequence[K]) -> dict[K, V]:
    return {k: d[k] for k in keys}


class timeout:
    def __init__(self, seconds: float = 1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def enumerate_bitstrings(n: int) -> Generator[tuple[bool, ...], None, None]:
    for i in range(n + 1):
        for bits in itertools.combinations(range(n), i):
            new_bits = [False for _ in range(n)]
            for bit in bits:
                new_bits[bit] = True
            yield tuple(new_bits)


def find_function_by_name(module: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    """
    Find a function by its name.
    """
    for node in ast.walk(module):
        match node:
            case ast.FunctionDef() if node.name == name:
                return node
    return None


def find_func_call_that_used_args(
    func: ast.FunctionDef, arg_strs: set[str]
) -> list[ast.Call]:
    """
    Find all function calls that used the given arguments.
    """
    nodes = []
    for node in ast.walk(func):
        match node:
            case ast.Call(func=ast.Name(), args=args):
                if set(ast.unparse(arg) for arg in args) & arg_strs:
                    nodes.append(node)
    return nodes


def format_infix(op: str, operands: Sequence[str]) -> str:
    """
    Formats an infix expression.
    """
    if len(operands) == 1:
        return f"{op}{operands[0]}"

    use_prefix = re.match(r"^[a-zA-Z0-9_]+$", op) is not None
    if use_prefix:
        return f"{op}({', '.join(operands)})"

    operands = list(operands)

    if op in "*/" and ("+" in operands[0] or "-" in operands[0]):
        operands[0] = f"({operands[0]})"

    if op == "/" and any(x in "+-*/" for x in operands[1]):
        operands[1] = f"({operands[1]})"

    if op in "*-" and ("+" in operands[1] or "-" in operands[1]):
        operands[1] = f"({operands[1]})"

    if op == "**":
        operands = [f"({x})" if not x.isdigit() else x for x in operands]

    if op == "/":
        return "(" + f" {op} ".join(operands) + ")"

    return f" {op} ".join(operands)
