from __future__ import annotations

import ast
import operator
import re
from enum import Enum
from functools import singledispatch
from typing import Any

import sympy as sp
import z3
from attrs import define
from attrs import field as attrs_field

import b2s.lang as ir
from b2s.utils import format_infix


class FunctionKinds(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    NEG = "^-"
    POS = "^+"
    POW = "**"
    MOD = "%"


class PredicateKinds(Enum):
    EQ = "="
    NEQ = "<>"
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="


AST_TO_FUNCTION_KINDS = {
    ast.Add: FunctionKinds.ADD,
    ast.Sub: FunctionKinds.SUB,
    ast.Mult: FunctionKinds.MUL,
    ast.Div: FunctionKinds.DIV,
    ast.USub: FunctionKinds.NEG,
    ast.UAdd: FunctionKinds.POS,
    ast.Pow: FunctionKinds.POW,
    ast.Mod: FunctionKinds.MOD,
    ast.operator: None,
    ast.unaryop: None,
}

AST_TO_PREDICATE_KINDS = {
    ast.Eq: PredicateKinds.EQ,
    ast.NotEq: PredicateKinds.NEQ,
    ast.Lt: PredicateKinds.LT,
    ast.Gt: PredicateKinds.GT,
    ast.LtE: PredicateKinds.LTE,
    ast.GtE: PredicateKinds.GTE,
    ast.cmpop: None,
}

IR_TO_FUNCTION_KINDS = {
    ir.BinOpKinds.ADD: FunctionKinds.ADD,
    ir.BinOpKinds.SUB: FunctionKinds.SUB,
    ir.BinOpKinds.MUL: FunctionKinds.MUL,
    ir.BinOpKinds.DIV: FunctionKinds.DIV,
    ir.BinOpKinds.POW: FunctionKinds.POW,
}

IR_TO_PREDICATE_KINDS = {
    ir.BinOpKinds.EQ: PredicateKinds.EQ,
    ir.BinOpKinds.NEQ: PredicateKinds.NEQ,
    ir.BinOpKinds.LT: PredicateKinds.LT,
    ir.BinOpKinds.GT: PredicateKinds.GT,
    ir.BinOpKinds.LTE: PredicateKinds.LTE,
    ir.BinOpKinds.GTE: PredicateKinds.GTE,
}

FUNCTION_KINDS_TO_OPERATOR = {
    FunctionKinds.ADD: operator.add,
    FunctionKinds.SUB: operator.sub,
    FunctionKinds.MUL: operator.mul,
    FunctionKinds.DIV: operator.truediv,
    FunctionKinds.NEG: operator.neg,
    FunctionKinds.POS: operator.pos,
    FunctionKinds.POW: operator.pow,
    FunctionKinds.MOD: operator.mod,
}

PREDICATE_KINDS_TO_OPERATOR = {
    PredicateKinds.EQ: operator.eq,
    PredicateKinds.NEQ: operator.ne,
    PredicateKinds.LT: operator.lt,
    PredicateKinds.GT: operator.gt,
    PredicateKinds.LTE: operator.le,
    PredicateKinds.GTE: operator.ge,
}


@define(frozen=True, slots=True, eq=True, hash=True)
class Variable:
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


@define(frozen=True, slots=True, eq=True, hash=True)
class Constant:
    value: Any

    def __str__(self) -> str:
        if isinstance(self.value, float) and self.value.is_integer():
            return str(int(self.value))
        return str(self.value)

    def __repr__(self) -> str:
        return str(self)


@define(frozen=True, slots=True, eq=True, hash=True)
class FunctionApp:
    func: FunctionKinds | Variable
    args: tuple[Expression, ...] = attrs_field()

    def __str__(self) -> str:
        func_str = (
            str(self.func) if isinstance(self.func, Variable) else self.func.value
        )

        if isinstance(self.func, Variable):
            return f"{func_str}({', '.join(map(str, self.args))})"

        return format_infix(func_str, tuple(map(str, self.args)))

    def __repr__(self) -> str:
        return str(self)

    @args.validator
    def _validate_args(self, _: Any, args: tuple[Expression, ...]) -> None:
        str_args = tuple(map(str, args))
        for arg in str_args:
            if "RootObject" in arg:
                raise ValueError(f"Invalid argument {arg}")
        if self.func == FunctionKinds.DIV and args[1] == Constant(0):
            raise ZeroDivisionError(f"Invalid argument {args[1]} for function {self.func}")
    
   
    @classmethod
    def from_many_args(cls, func: FunctionKinds, args: tuple[Expression, ...]) -> Expression:
        if len(args) == 1 and func == FunctionKinds.SUB:
            return FunctionApp(FunctionKinds.NEG, (args[0],))
        if len(args) < 2:
            raise ValueError(f"Function {func} requires at least 2 arguments")
        
        base = cls(func, (args[0], args[1]))
        for arg in args[2:]:
            base = cls(func, (base, arg))
        return base


@define(frozen=True, slots=True, eq=True, hash=True)
class ValueTop:
    def __str__(self) -> str:
        return "\u22a4"

    def __repr__(self) -> str:
        return str(self)


@define(frozen=True, slots=True, eq=True, hash=True)
class ValueBottom:
    def __str__(self) -> str:
        return "\u22a5"

    def __repr__(self) -> str:
        return str(self)


@define(frozen=True, slots=True, eq=True, hash=True)
class PredicateApp:
    pred: PredicateKinds
    args: tuple[Expression, ...]

    def __str__(self) -> str:
        return format_infix(self.pred.value, tuple(map(str, self.args)))

    def __repr__(self) -> str:
        return str(self)


@define(frozen=True, slots=True, eq=True, hash=True)
class Negation:
    expr: Formula

    def __str__(self) -> str:
        return f"\u00ac{self.expr}"

    def __repr__(self) -> str:
        return f"Negation({str(self)})"


@define(frozen=True, slots=True, eq=True, hash=True)
class Conjunction:
    exprs: tuple[Formula, ...]

    def __str__(self) -> str:
        conj_char = " \u2227 "
        return f"({conj_char.join(map(str, self.exprs))})"

    def __repr__(self) -> str:
        return f"Conjunction({str(self)})"

    @classmethod
    def create(cls, *exprs: Formula) -> Formula:
        assert all(isinstance(expr, Formula) for expr in exprs)
        exprs = tuple(filter(lambda expr: not isinstance(expr, ValueTop), exprs))
        if not exprs:
            return ValueTop()
        if any(isinstance(expr, ValueBottom) for expr in exprs):
            return ValueBottom()
        if len(exprs) == 1:
            return exprs[0]
        if all(isinstance(expr, Conjunction) for expr in exprs):
            return cls(tuple(expr for conj in exprs for expr in conj.exprs))  # type: ignore
        return cls(exprs)


@define(frozen=True, slots=True, eq=True, hash=True)
class Disjunction:
    exprs: tuple[Formula, ...]

    def __str__(self) -> str:
        disj_char = " \u2228 "
        return f"({disj_char.join(map(str, self.exprs))})"

    def __repr__(self) -> str:
        return f"Disjunction({str(self)})"

    @classmethod
    def create(cls, *exprs: Formula) -> Formula:
        assert all(isinstance(expr, Formula) for expr in exprs)
        exprs = tuple(filter(lambda expr: not isinstance(expr, ValueBottom), exprs))
        if not exprs:
            return ValueBottom()
        if any(isinstance(expr, ValueTop) for expr in exprs):
            return ValueTop()
        if len(exprs) == 1:
            return exprs[0]
        if all(isinstance(expr, Disjunction) for expr in exprs):
            return cls(tuple(expr for disj in exprs for expr in disj.exprs))  # type: ignore
        return cls(exprs)


@define(frozen=True, slots=True, eq=True, hash=True)
class Implication:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} \u2192 {self.right})"

    def __repr__(self) -> str:
        return f"Implication({str(self)})"


Expression = Variable | FunctionApp | Constant
Formula = (
    PredicateApp
    | Negation
    | Conjunction
    | Disjunction
    | Implication
    | ValueTop
    | ValueBottom
)

@singledispatch
def extract_variables(_: Any) -> set[str]:
    raise NotImplementedError

@extract_variables.register
def extract_variables_from_expression(ex: Expression) -> set[str]:
    match ex:
        case Variable(name):
            return {name}
        case Constant():
            return set()
        case FunctionApp(_, args):
            return set().union(*(map(extract_variables_from_expression, args)))
        
@extract_variables.register
def extract_variables_from_formula(f: Formula) -> set[str]:
    match f:
        case ValueTop():
            return set()
        case ValueBottom():
            return set()
        case Implication(left, right):
            return extract_variables_from_formula(left) | extract_variables_from_formula(right)
        case Conjunction(exprs):
            return set().union(*(map(extract_variables_from_formula, exprs)))
        case Disjunction(exprs):
            return set().union(*(map(extract_variables_from_formula, exprs)))
        case Negation(expr):
            return extract_variables_from_formula(expr)
        case PredicateApp(_, args):
            return set().union(*(map(extract_variables_from_expression, args)))

@singledispatch
def to_sympy(_: Any) -> Any:
    raise NotImplementedError

@to_sympy.register
def to_sympy_expr(expr: Expression):
    match expr:
        case Variable(name):
            return sp.Symbol(name)
        case Constant(value):
            return sp.Number(value)
        case FunctionApp(func, args):
            assert isinstance(func, FunctionKinds)
            return FUNCTION_KINDS_TO_OPERATOR[func](*map(to_sympy_expr, args))


@singledispatch
def to_z3(_: Any) -> Any:
    raise NotImplementedError


@to_z3.register
def to_z3_formula(f: Formula) -> Any:
    match f:
        case ValueTop():
            return z3.BoolVal(True)
        case ValueBottom():
            return z3.BoolVal(False)
        case Implication(left, right):
            return z3.Implies(to_z3(left), to_z3(right))
        case Conjunction(exprs):
            return z3.And(*map(to_z3, exprs))
        case Disjunction(exprs):
            return z3.Or(*map(to_z3, exprs))
        case Negation(expr):
            return z3.Not(to_z3(expr))
        case PredicateApp(pred, args):
            return PREDICATE_KINDS_TO_OPERATOR[pred](*map(to_z3, args))


@to_z3.register
def to_z3_expr(ex: Expression) -> Any:
    match ex:
        case Variable(name):
            return z3.Real(name)
        case Constant(value):
            return z3.RealVal(value)
        case FunctionApp(func, args):
            assert isinstance(func, FunctionKinds)
            return FUNCTION_KINDS_TO_OPERATOR[func](*map(to_z3_expr, args))


Z3_DECL_STR_TO_PRED = {
    "<=": PredicateKinds.LTE,
    ">=": PredicateKinds.GTE,
    "<": PredicateKinds.LT,
    ">": PredicateKinds.GT,
    "=": PredicateKinds.EQ,
    "!=": PredicateKinds.NEQ,
}

Z3_DECL_STR_TO_FUNC = {
    "+": FunctionKinds.ADD,
    "-": FunctionKinds.SUB,
    "*": FunctionKinds.MUL,
    "/": FunctionKinds.DIV,
    "^": FunctionKinds.POW,
}


def z3_ref_to_formula(ex):
    match ex:
        case z3.BoolRef():
            match ex.decl().name():
                case "true":
                    return ValueTop()
                case "false":
                    return ValueBottom()
                case "and":
                    return Conjunction.create(*(map(z3_ref_to_formula, ex.children())))  # type: ignore
                case "or":
                    return Disjunction.create(*(map(z3_ref_to_formula, ex.children())))
                case "not":
                    return Negation(z3_ref_to_formula(ex.children()[0]))
                case "implies":
                    return Implication(
                        z3_ref_to_formula(ex.children()[0]),
                        z3_ref_to_formula(ex.children()[1]),
                    )
                case op if op in Z3_DECL_STR_TO_PRED:
                    try:
                        return PredicateApp(
                            Z3_DECL_STR_TO_PRED[op],
                            tuple(map(z3_ref_to_formula, ex.children())),
                        )
                    except ZeroDivisionError:
                        return ValueBottom()

                case _:
                    raise NotImplementedError

        case z3.RatNumRef() if ex.is_int_value():
            return Constant(ex.as_long())
        case z3.RatNumRef() if ex.is_real():
            frac = ex.as_fraction()
            return FunctionApp(
                FunctionKinds.DIV,
                (Constant(frac.numerator), Constant(frac.denominator)),
            )
        case z3.ArithRef():
            match ex.decl().name():
                case op if op in Z3_DECL_STR_TO_FUNC:
                    return FunctionApp.from_many_args(
                        Z3_DECL_STR_TO_FUNC[op],
                        tuple(map(z3_ref_to_formula, ex.children())),
                    )
                
                case "RootObject":
                    return Constant(float(str(ex).removesuffix("?")))

                case var_name if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
                    return Variable(var_name)

                case _:
                    raise NotImplementedError

        case int() | float():
            return Constant(ex)

    raise NotImplementedError

def sympy_expr_to_expression(expr: sp.Basic) -> Expression:
    symbols = {
        sym.name: z3.Real(sym.name)
        for sym in expr.free_symbols
        if isinstance(sym, sp.Symbol)
    }

    z3_expr = eval(str(expr), symbols)
    result = z3_ref_to_formula(z3_expr)
    assert isinstance(result, Expression)
    return result


def check_sat(f: Formula) -> bool:
    match f:
        case ValueTop():
            return True
        case ValueBottom():
            return False

    solver = z3.Solver()
    solver.add(to_z3(f))
    return solver.check() == z3.sat


def simplify(f: Formula) -> Formula:
    z3_f = to_z3(f)
    simplified = z3.simplify(z3_f)
    return z3_ref_to_formula(simplified)  # type: ignore


@singledispatch
def to_reduce_formula(f: Any) -> str:
    raise NotImplementedError


@to_reduce_formula.register
def _formula_to_reduce(f: Formula) -> str:
    match f:
        case ValueTop():
            return "true"
        case ValueBottom():
            return "false"
        case Implication(left, right):
            return f"(({to_reduce_formula(left)}) impl ({to_reduce_formula(right)}))"
        case Conjunction(exprs):
            return f"({' and '.join(map(to_reduce_formula, exprs))})"
        case Disjunction(exprs):
            return f"({' or '.join(map(to_reduce_formula, exprs))})"
        case Negation(expr):
            return f"(not ({to_reduce_formula(expr)}))"
        case PredicateApp(pred, args):
            return f"({format_infix(pred.value, tuple(map(to_reduce_formula, args)))})"


@to_reduce_formula.register
def _expr_to_reduce(ex: Expression) -> str:
    match ex:
        case Variable(name):
            return name
        case Constant(value):
            return str(value)
        case FunctionApp(func, args):
            if func == FunctionKinds.NEG:
                return f"(-{to_reduce_formula(args[0])})"
            fs = str(func) if isinstance(func, Variable) else func.value
            return format_infix(str(fs), tuple(map(to_reduce_formula, args)))


@singledispatch
def to_python_str(f: Any) -> str:
    raise NotImplementedError


@to_python_str.register
def _expr_to_python_str(ex: Expression) -> str:
    match ex:
        case Variable(name):
            return name
        case Constant(value):
            return str(value)
        case FunctionApp(func, args):
            func = str(func) if isinstance(func, Variable) else to_python_str(func)
            return format_infix(func, tuple(map(to_python_str, args)))


@to_python_str.register
def _func_to_python_str(f: FunctionKinds) -> str:
    match f:
        case FunctionKinds.POS:
            return "+"
        case FunctionKinds.NEG:
            return "-"

    return f.value


@to_python_str.register
def _formula_to_python_str(f: Formula) -> str:
    match f:
        case ValueTop():
            return "True"
        case ValueBottom():
            return "False"
        case Conjunction(exprs):
            return f"({' and '.join(map(to_python_str, exprs))})"
        case Disjunction(exprs):
            return f"({' or '.join(map(to_python_str, exprs))})"
        case Implication(left, right):
            return f"(not ({to_python_str(left)}) or {to_python_str(right)})"
        case Negation(expr):
            return f"(not ({to_python_str(expr)}))"
        case PredicateApp(pred, args):
            return f"({format_infix(pred.value, tuple(map(to_python_str, args)))})"


def rewrite_div_to_mul(f: Formula) -> Formula:
    match f:
        case PredicateApp(PredicateKinds.EQ, (left, right)):
            match right:
                case FunctionApp(FunctionKinds.DIV, (num, denom)):
                    return PredicateApp(
                        PredicateKinds.EQ,
                        (FunctionApp(FunctionKinds.MUL, (left, denom)), num),
                    )
            match left:
                case FunctionApp(FunctionKinds.DIV, (num, denom)):
                    return PredicateApp(
                        PredicateKinds.EQ,
                        (FunctionApp(FunctionKinds.MUL, (right, denom)), num),
                    )
        case Conjunction(exprs):
            return Conjunction(tuple(map(rewrite_div_to_mul, exprs)))
        case Disjunction(exprs):
            return Disjunction(tuple(map(rewrite_div_to_mul, exprs)))
        case Implication(left, right):
            return Implication(rewrite_div_to_mul(left), rewrite_div_to_mul(right))
        case Negation(expr):
            return Negation(rewrite_div_to_mul(expr))

    return f


def contains(f: Formula | Expression, item: Formula | Expression) -> bool:
    match f:
        case _ if f == item:
            return True
        case Implication(left, right):
            return contains(left, item) or contains(right, item)
        case Conjunction(exprs):
            return any(contains(expr, item) for expr in exprs)
        case Disjunction(exprs):
            return any(contains(expr, item) for expr in exprs)
        case Negation(expr):
            return contains(expr, item)
        case PredicateApp(_, args):
            return any(contains(arg, item) for arg in args)
        case FunctionApp(_, args):
            return any(contains(arg, item) for arg in args)

    return False


def substitute_variable_expr(
    expr: Expression, old_expr: Expression, new_expr: Expression
) -> Expression:
    match expr:
        case x if x == old_expr:
            return new_expr

        case FunctionApp(func=func, args=args):
            return FunctionApp(
                func=func,
                args=tuple(
                    substitute_variable_expr(a, old_expr, new_expr) for a in args
                ),
            )

    return expr


def substitute_variable(
    f: Formula, old_expr: Expression, new_expr: Expression
) -> Formula:
    match f:
        case PredicateApp(pred=pred, args=args):
            return PredicateApp(
                pred=pred,
                args=tuple(
                    substitute_variable_expr(a, old_expr, new_expr) for a in args
                ),
            )

        case Negation(expr=expr):
            return Negation(substitute_variable(expr, old_expr, new_expr))

        case Conjunction(exprs=exprs):
            return Conjunction(
                tuple(substitute_variable(e, old_expr, new_expr) for e in exprs)
            )

        case Disjunction(exprs=exprs):
            return Disjunction(
                tuple(substitute_variable(e, old_expr, new_expr) for e in exprs)
            )

        case Implication(left=left, right=right):
            return Implication(
                left=substitute_variable(left, old_expr, new_expr),
                right=substitute_variable(right, old_expr, new_expr),
            )

    return f
