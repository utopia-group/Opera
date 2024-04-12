import re
from numbers import Number

from attrs import define
from frozendict import frozendict
from sympy import Poly, Pow, Symbol, nan
from sympy.core.function import UndefinedFunction
from sympy.parsing.sympy_parser import parse_expr

from b2s.sketch_types import EnvState, IOExamples
from b2s.utils import find_denominator_lcm, sympy_const_to_number


def _make_expression_template(expr_str: str) -> tuple[str, dict[str, Number]]:
    expr_str_replaced = re.sub(r"\bexp\(", "exp_(", expr_str)

    expr = parse_expr(expr_str_replaced)
    if expr == nan:
        return expr_str, {} # when the solver returns invalid result
    if expr.is_number:
        return "{const}", {"const": sympy_const_to_number(expr)}

    poly = Poly(expr)
    common_deno = find_denominator_lcm(poly.coeffs())

    template_str_parts = []
    consts = {}

    for unk_idx, term in enumerate(expr.as_ordered_terms()):
        unk_name = f"c{unk_idx}"
        unk_symb = Symbol(f"{{{unk_name}}}")
        [const] = [t for t in term.args if not t.free_symbols and not isinstance(t.__class__, UndefinedFunction)] or [1]
        if term.func is Pow:
            const = 1 # don't modify the power

        sub_term = term.subs(const, unk_symb)
        if const == 1 and sub_term == term:
            sub_term *= unk_symb

        template_str_parts.append(str(sub_term))

        consts[unk_name] = sympy_const_to_number(const * common_deno)

        if abs(consts[unk_name]) == 1:
            # make the grammar consistent to QE solver result
            # eliminates free variables from solving equations for more examples
            template_str_parts[-1] = str(term.subs(const, consts[unk_name]))
            del consts[unk_name]

    template_str = " + ".join(template_str_parts)
    template_str = re.sub(r"\bexp_\(", "exp(", template_str)
    if common_deno != 1:
        unk_name = f"c{len(template_str_parts)}"
        template_str = f"({template_str}) / {{{unk_name}}}"
        consts[unk_name] = sympy_const_to_number(common_deno)
    
    if not consts:
        # no constants
        # sympy simplication for exp/log
        expr = parse_expr(expr_str)
        expr = expr.simplify()
        template_str = str(expr)

    return template_str, consts # type: ignore


@define(frozen=True)
class ExprTemplate:
    """
    A template for an expression.

    Fields:
        template:   The template string.
        ctx_ios:    maps context variables to a list of (context, constant_value) pairs.
    """

    expr: str
    ctx_ios: frozendict[str, IOExamples]

    @classmethod
    def from_expr(cls, expr: str, ctx: EnvState) -> "ExprTemplate":
        expr_str, consts = _make_expression_template(expr)
        tmp: dict[str, IOExamples] = {k: ((ctx, v),) for k, v in consts.items()}
        return cls(expr=expr_str, ctx_ios=frozendict(tmp))


@define(frozen=True, slots=True)
class ConcreteExprGrammar:
    exprs: tuple[ExprTemplate, ...]
    guards: tuple[str, ...]
