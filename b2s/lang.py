from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from numbers import Number
from typing import Any, Callable, Self, Sequence, TypeAlias, TypeVar

from frozendict import frozendict


class Expr:
    pass


class Value:
    pass


Id: TypeAlias = str
Env: TypeAlias = frozendict[Id, Value]
RefEnv: TypeAlias = dict[Id, Value]

r"""
Grammar for the intermediate language:

Expr        ::= Variable | Constant | BinOp | App | Lambda | Let | ITE
Variable    ::= [a-zA-Z_][a-zA-Z0-9_]*
Constant    ::= [0-9]+ | ".*" | true | false | Nil
BinOp       ::= Expr + Expr | Expr - Expr | Expr * Expr | Expr / Expr | Cons Expr Expr
App         ::= Expr Expr
Lambda      ::= \Variable -> Expr
Let         ::= let Variable = Expr in Expr
ITE         ::= if Expr then Expr else Expr


Library functions:
- trace :: String -> a -> a
    The trace function outputs the trace message given as its first
    argument, before returning the second argument as its result.

    For example, this returns the value of f x and outputs the
    message to stderr. Depending on your terminal (settings), they may or
    may not be mixed.

- foldl :: Foldable t => (b -> a -> b) -> b -> t a -> b

"""


class BinOpKinds(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EQ = "=="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    AND = "&&"
    OR = "||"
    POW = "^"
    CONS = "::"


@dataclass(frozen=True, slots=True, eq=True)
class EVar(Expr):
    name: Id


@dataclass(frozen=True, slots=True, eq=True)
class EStream(Expr):
    name: Id


@dataclass(frozen=True, slots=True, eq=True)
class ENumber(Expr):
    value: int | float


@dataclass(frozen=True, slots=True, eq=True)
class EBool(Expr):
    value: bool


@dataclass(frozen=True, slots=True, eq=True)
class EString(Expr):
    value: str


@dataclass(frozen=True, slots=True, eq=True)
class EBinOp(Expr):
    left: Expr
    op: BinOpKinds
    right: Expr


@dataclass(frozen=True, slots=True, eq=True)
class ECall(Expr):
    func: Expr
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True, eq=True)
class ELam(Expr):
    params: tuple[Id, ...]
    body: Expr


@dataclass(frozen=True, slots=True, eq=True)
class ELet(Expr):
    name: Id
    expr: Expr
    body: Expr


@dataclass(frozen=True, slots=True, eq=True)
class EIte(Expr):
    cond: Expr
    then: Expr
    els: Expr


@dataclass(frozen=True, slots=True, eq=True)
class ENil(Expr):
    pass


@dataclass(frozen=True, slots=True, eq=True)
class EPair(Expr):
    elts: tuple[Expr, ...]

    @classmethod
    def create(cls, *elts: Expr) -> Self:
        return cls(elts)


@dataclass(frozen=True, slots=True, eq=True)
class EMapUpdate(Expr):
    expr: Expr
    updates: frozendict[Id, Expr]

    @classmethod
    def create(cls, expr: Expr, **kwargs: Expr) -> Self:
        return cls(expr, frozendict(kwargs))


@dataclass(frozen=True, slots=True, eq=True)
class EMapNil(Expr):
    pass


@dataclass(frozen=True, slots=True, eq=True)
class EMapGet(Expr):
    expr: Expr
    key: Id

    @classmethod
    def create(cls, expr: Expr, key: Id) -> Self:
        return cls(expr, key)


@dataclass(frozen=True, slots=True, eq=True)
class EUnknown(Expr):
    unk_id: str


@dataclass(frozen=True, slots=True, eq=True)
class EPythonExpr(Expr):
    expr: str


@dataclass(frozen=True, slots=True, eq=True)
class VNumber(Value):
    value: int | float


@dataclass(frozen=True, slots=True, eq=True)
class VBool(Value):
    value: bool


@dataclass(frozen=True, slots=True, eq=True)
class VString(Value):
    value: str


@dataclass(frozen=True, slots=True, eq=True)
class VPair(Value):
    elts: tuple[Value, ...]


@dataclass(frozen=True, slots=True, eq=True)
class VMap(Value):
    value: frozendict[str, Value]


@dataclass(frozen=True, slots=True, eq=True)
class VNil(Value):
    pass


@dataclass(frozen=True, slots=True, eq=True)
class VCons(Value):
    head: Value
    tail: Value


@dataclass(frozen=True, slots=True, eq=True)
class VClosure(Value):
    params: tuple[Id, ...]
    body: Expr
    env: Env


@dataclass(frozen=True, slots=True, eq=True)
class VError(Value):
    msg: str


@dataclass(frozen=True, slots=True, eq=True)
class VPreludeFunc(Value):
    func: Callable[..., Value]


@dataclass(frozen=True, slots=True, eq=True)
class VForeignValue(Value):
    value: Any


T = TypeVar("T")


def fold_expr(
    expr: Expr,
    f_var: Callable[[EVar], T],
    f_stream: Callable[[EStream], T],
    f_int: Callable[[ENumber], T],
    f_bool: Callable[[EBool], T],
    f_str: Callable[[EString], T],
    f_binop: Callable[[T, BinOpKinds, T], T],
    f_call: Callable[[T, tuple[T, ...]], T],
    f_lam: Callable[[tuple[Id, ...], T], T],
    f_let: Callable[[Id, T, T], T],
    f_ite: Callable[[T, T, T], T],
    f_nil: Callable[[], T],
    f_pair: Callable[[tuple[T, ...]], T],
    f_unk: Callable[[Id], T],
    f_map: Callable[[T, frozendict[Id, T]], T],
    f_map_nil: Callable[[], T],
    f_map_get: Callable[[T, Id], T],
    f_python_expr: Callable[[str], T],
) -> T:
    def fold_expr_internal(expr: Expr) -> T:
        return fold_expr(
            expr,
            f_var,
            f_stream,
            f_int,
            f_bool,
            f_str,
            f_binop,
            f_call,
            f_lam,
            f_let,
            f_ite,
            f_nil,
            f_pair,
            f_unk,
            f_map,
            f_map_nil,
            f_map_get,
            f_python_expr,
        )

    match expr:
        case EVar():
            return f_var(expr)
        case EStream():
            return f_stream(expr)
        case ENumber():
            return f_int(expr)
        case EBool():
            return f_bool(expr)
        case EString():
            return f_str(expr)
        case EBinOp(left, op, right):
            return f_binop(fold_expr_internal(left), op, fold_expr_internal(right))
        case ECall(func, args):
            return f_call(
                fold_expr_internal(func), tuple(map(fold_expr_internal, args))
            )
        case ELam(params, body):
            return f_lam(params, fold_expr_internal(body))
        case ELet(name, expr, body):
            return f_let(name, fold_expr_internal(expr), fold_expr_internal(body))
        case EIte(cond, then, els):
            return f_ite(
                fold_expr_internal(cond),
                fold_expr_internal(then),
                fold_expr_internal(els),
            )
        case ENil():
            return f_nil()
        case EPair(elts):
            return f_pair(tuple(map(fold_expr_internal, elts)))
        case EUnknown(unk_id):
            return f_unk(unk_id)
        case EMapUpdate(expr, updates):
            return f_map(
                fold_expr_internal(expr),
                frozendict({k: fold_expr_internal(v) for k, v in updates.items()}),
            )
        case EMapNil():
            return f_map_nil()
        case EMapGet(expr, key):
            return f_map_get(fold_expr_internal(expr), key)
        case EPythonExpr(e):
            return f_python_expr(e)
    raise ValueError(f"Unknown expression type: {expr}")


def fold_expr_rec(
    expr: Expr,
    f_var: Callable[[Callable[[Expr], T], EVar], T],
    f_stream: Callable[[Callable[[Expr], T], EStream], T],
    f_int: Callable[[Callable[[Expr], T], ENumber], T],
    f_bool: Callable[[Callable[[Expr], T], EBool], T],
    f_str: Callable[[Callable[[Expr], T], EString], T],
    f_binop: Callable[[Callable[[Expr], T], T, BinOpKinds, T], T],
    f_call: Callable[[Callable[[Expr], T], T, tuple[T, ...]], T],
    f_lam: Callable[[Callable[[Expr], T], tuple[Id, ...], T], T],
    f_let: Callable[[Callable[[Expr], T], Id, T, T], T],
    f_ite: Callable[[Callable[[Expr], T], T, T, T], T],
    f_nil: Callable[
        [
            Callable[[Expr], T],
        ],
        T,
    ],
    f_pair: Callable[[Callable[[Expr], T], tuple[T, ...]], T],
    f_unk: Callable[[Callable[[Expr], T], Id], T],
    f_map: Callable[[Callable[[Expr], T], T, frozendict[Id, T]], T],
    f_map_nil: Callable[
        [
            Callable[[Expr], T],
        ],
        T,
    ],
    f_map_get: Callable[[Callable[[Expr], T], T, Id], T],
    f_python_expr: Callable[[Callable[[Expr], T], str], T],
) -> T:
    def fold_expr_internal(expr: Expr) -> T:
        return fold_expr_rec(
            expr,
            f_var,
            f_stream,
            f_int,
            f_bool,
            f_str,
            f_binop,
            f_call,
            f_lam,
            f_let,
            f_ite,
            f_nil,
            f_pair,
            f_unk,
            f_map,
            f_map_nil,
            f_map_get,
            f_python_expr,
        )

    match expr:
        case EVar():
            return f_var(fold_expr_internal, expr)
        case EStream():
            return f_stream(fold_expr_internal, expr)
        case ENumber():
            return f_int(fold_expr_internal, expr)
        case EBool():
            return f_bool(fold_expr_internal, expr)
        case EString():
            return f_str(fold_expr_internal, expr)
        case EBinOp(left, op, right):
            return f_binop(
                fold_expr_internal,
                fold_expr_internal(left),
                op,
                fold_expr_internal(right),
            )
        case ECall(func, args):
            return f_call(
                fold_expr_internal,
                fold_expr_internal(func),
                tuple(map(fold_expr_internal, args)),
            )
        case ELam(params, body):
            return f_lam(fold_expr_internal, params, fold_expr_internal(body))
        case ELet(name, expr, body):
            return f_let(
                fold_expr_internal,
                name,
                fold_expr_internal(expr),
                fold_expr_internal(body),
            )
        case EIte(cond, then, els):
            return f_ite(
                fold_expr_internal,
                fold_expr_internal(cond),
                fold_expr_internal(then),
                fold_expr_internal(els),
            )
        case ENil():
            return f_nil(
                fold_expr_internal,
            )
        case EPair(elts):
            return f_pair(fold_expr_internal, tuple(map(fold_expr_internal, elts)))
        case EUnknown(unk_id):
            return f_unk(fold_expr_internal, unk_id)
        case EMapUpdate(expr, updates):
            return f_map(
                fold_expr_internal,
                fold_expr_internal(expr),
                frozendict({k: fold_expr_internal(v) for k, v in updates.items()}),
            )
        case EMapNil():
            return f_map_nil(
                fold_expr_internal,
            )
        case EMapGet(expr, key):
            return f_map_get(fold_expr_internal, fold_expr_internal(expr), key)
        case EPythonExpr(e):
            return f_python_expr(fold_expr_internal, e)
    raise ValueError(f"Unknown expression type: {expr}")


def replace_unknown(expr: Expr, unk_id: str, new_expr: Expr) -> Expr:
    return fold_expr(
        expr,
        f_var=lambda e: e,
        f_stream=lambda e: e,
        f_int=lambda e: e,
        f_bool=lambda e: e,
        f_str=lambda e: e,
        f_binop=lambda left, op, right: EBinOp(left, op, right),
        f_call=lambda func, args: ECall(func, args),
        f_lam=lambda params, body: ELam(params, body),
        f_let=lambda name, expr, body: ELet(name, expr, body),
        f_ite=lambda cond, then, els: EIte(cond, then, els),
        f_nil=lambda: ENil(),
        f_pair=lambda elts: EPair(elts),
        f_unk=lambda unk_id_: new_expr if unk_id_ == unk_id else EUnknown(unk_id_),
        f_map=lambda e, updates: EMapUpdate(e, updates),
        f_map_nil=lambda: EMapNil(),
        f_map_get=lambda e, key: EMapGet(e, key),
        f_python_expr=lambda e: EPythonExpr(e),
    )


def replace_variable(expr: Expr, var_id: str, new_expr: Expr) -> Expr:
    return fold_expr(
        expr,
        f_var=lambda e: new_expr if e.name == var_id else e,
        f_stream=lambda e: new_expr if e.name == var_id else e,
        f_int=lambda e: e,
        f_bool=lambda e: e,
        f_str=lambda e: e,
        f_binop=lambda left, op, right: EBinOp(left, op, right),
        f_call=lambda func, args: ECall(func, args),
        f_lam=lambda params, body: ELam(params, body),
        f_let=lambda name, expr, body: ELet(name, expr, body),
        f_ite=lambda cond, then, els: EIte(cond, then, els),
        f_nil=lambda: ENil(),
        f_pair=lambda elts: EPair(elts),
        f_unk=lambda unk_id_: EUnknown(unk_id_),
        f_map=lambda e, updates: EMapUpdate(e, updates),
        f_map_nil=lambda: EMapNil(),
        f_map_get=lambda e, key: EMapGet(e, key),
        f_python_expr=lambda e: EPythonExpr(e),
    )


def expr_is_complete(expr: Expr) -> bool:
    return fold_expr(
        expr,
        f_var=lambda _: True,
        f_stream=lambda _: True,
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
        f_unk=lambda _: False,
        f_map=lambda e, updates: e and all(updates.values()),
        f_map_nil=lambda: True,
        f_map_get=lambda e, key: e,
        f_python_expr=lambda e: True,
    )


def to_value(val: Any) -> Value:
    match val:
        case bool(v):
            return VBool(v)
        case n if isinstance(n, Number):
            return VNumber(n)
        case v if isinstance(v, Value):
            return v
        case str(v):
            return VString(v)
        case tuple(v):
            return VPair(tuple(map(to_value, v)))
        case defaultdict():
            return VForeignValue(val)
        case dict(v):
            return VMap(frozendict({k: to_value(v) for k, v in v.items()}))
        case list(v):
            expr = VNil()
            for x in reversed(v):
                expr = VCons(to_value(x), expr)
            return expr
        case v:
            return VForeignValue(v)


def to_const_expr(val: Any) -> Expr:
    match val:
        case v if isinstance(v, Number):
            return ENumber(v)
        case bool(v):
            return EBool(v)
        case str(v):
            return EString(v)
        case tuple(v):
            return EPair(tuple(map(to_const_expr, v)))
        case _:
            raise ValueError(f"Unknown value type: {val}")


def to_value_list(xs: Sequence[Any]) -> Value:
    return reduce(lambda acc, x: VCons(to_value(x), acc), reversed(xs), VNil())  # type: ignore


def to_expr_list(xs: Sequence[Any]) -> Expr:
    return reduce(lambda acc, x: EBinOp(acc, BinOpKinds.CONS, x), reversed(xs), ENil())  # type: ignore


def to_value_map(xs: dict[str, Any]) -> Value:
    return VMap(frozendict({k: to_value(v) for k, v in xs.items()}))


def to_expr_map(xs: dict[str, Any]) -> Expr:
    return EMapUpdate.create(EMapNil(), **{k: to_const_expr(v) for k, v in xs.items()})


def to_python_value(val: Value) -> Any:
    match val:
        case VNumber(v) | VBool(v) | VString(v):
            return v
        case VPair(elts):
            return tuple(map(to_python_value, elts))
        case VMap(value):
            return {k: to_python_value(v) for k, v in value.items()}
        case VNil():
            return ()
        case VCons(head, tail):
            return (to_python_value(head), *to_python_value(tail))
        case VForeignValue(v):
            return v

    raise ValueError(f"Unknown value type: {val}")


def pprint(expr: Expr) -> str:
    match expr:
        case EVar(name) | EStream(name):
            return name
        case ENumber(value) | EBool(value) | EString(value):
            return str(value)
        case EBinOp(left, op, right):
            return f"({pprint(left)} {op.value} {pprint(right)})"
        case ECall(func, args):
            return f"{pprint(func)}({', '.join(map(pprint, args))})"
        case ELam(params, body):
            return f"\\{' '.join(params)} -> {pprint(body)}"
        case ELet(name, expr, body):
            return f"let {name} = {pprint(expr)} in {pprint(body)}"
        case EIte(cond, then, els):
            return f"if {pprint(cond)} then {pprint(then)} else {pprint(els)}"
        case ENil():
            return "[]"
        case EPair(elts):
            return f"({', '.join(map(pprint, elts))})"
        case EUnknown(name):
            return f"??{name}"
        case EMapUpdate(expr, updates):
            return f"{pprint(expr)}{{{'; '.join(f'{k} = {pprint(v)}' for k, v in updates.items())}}}"
        case EMapNil():
            return "{}"
        case EMapGet(expr, key):
            return f"{pprint(expr)}[{key}]"
        case EPythonExpr(e):
            return f"python({e})"
    raise ValueError(f"Unknown expression type: {expr}")
