import logging
import math
import random
from itertools import combinations
from numbers import Number
from typing import Generator, Sequence

import numpy as np
from frozendict import frozendict
from sympy import Expr, Poly, Symbol, nsimplify, solve, symbols
from sympy.parsing.sympy_parser import parse_expr
from sympy.polys.polyfuncs import interpolate as sympy_interpolate
from sympy.polys.polytools import degree
from tqdm import tqdm

from b2s.const import VAR_NAME
from b2s.expr_template import ExprTemplate
from b2s.sketch_types import IOExample, IOExamples
from b2s.synthesizers.enum_syn import (
    eval_eq,
    eval_eq_exs,
)
from b2s.utils import powerset, sympy_const_to_number

MATH_ENV = {
    "log": math.log,
    "exp": math.exp,
}


def synthesize_partial(
    io_examples: IOExamples, templates: Sequence[ExprTemplate]
) -> Generator[str, None, None]:
    def _try_eval(expr: str, exs: IOExamples):
        for st, val in exs:
            if eval_eq([st], expr, [val]):
                return True
        return False

    # try template with constants
    for t in templates:
        consts = {k: vs[0][1] for k, vs in t.ctx_ios.items()}
        prog = t.expr.format(**consts)
        if _try_eval(prog, io_examples):
            yield prog

    # synthesize exprs for constants
    for t in templates:
        r = synthesize_from_qe_formula(io_examples, t)
        if r is not None and _try_eval(r, io_examples):
            yield r

    vars_in_scope = io_examples[0][0].keys()
    for v in vars_in_scope:
        if _try_eval(v, io_examples):
            yield v


def synthesize_stun(
    io_examples: IOExamples, guards: Sequence[str], templates: Sequence[ExprTemplate]
) -> str | None:
    partial_programs = set()

    def _try_partial_program(
        p: str, check_heuristic: bool = True
    ) -> tuple[str | None, bool]:
        failed_idx = []
        succeeded_idx = []
        for i, (st, val) in enumerate(io_examples):
            if eval_eq([st], p, [val]):
                succeeded_idx.append(i)
            else:
                failed_idx.append(i)

        if not failed_idx:
            return p, True

        # heuristic to try another program
        # TODO: should be new input
        rand_ex = random.choice(io_examples)
        if check_heuristic and not eval_eq([rand_ex[0]], p, [rand_ex[1]]):
            return None, True

        # try to find a guard that can describe the failed examples
        for gs in powerset(guards, 3):
            g = " and ".join(gs)
            if eval_eq(
                [st for st, _ in io_examples],
                g,
                [i in failed_idx for i in range(len(io_examples))],
            ):
                break
        else:
            # raise NotImplementedError("how to decide guard?")
            return None, False

        print(f"trying guard {g}")
        rec_case = synthesize_stun(
            [(st, val) for i, (st, val) in enumerate(io_examples) if i in failed_idx],
            guards,
            templates,
        )
        if rec_case is None:
            return None, False

        return f"{rec_case} if {g} else {p}", True

    for p in synthesize_partial(io_examples, templates):
        print(f"trying partial program {p}")
        s, passed_any_ex = _try_partial_program(p)
        if s is not None:
            return s

        if passed_any_ex:
            partial_programs.add(p)

    for p in partial_programs:
        s, _ = _try_partial_program(p, check_heuristic=False)
        if s is not None:
            return s

    return None


def interpolate(
    data: Sequence[tuple[Number, Number]],
    x: Symbol,
) -> Expr:
    expr = sympy_interpolate(data, x)
    if expr.is_number:
        return expr

    poly = Poly(expr)
    coeffs_map = {c: round(c, 3) for c in poly.coeffs()}
    for c, v in coeffs_map.items():
        poly = poly.subs(c, v)

    return poly.as_expr()


def simplify_expr_str(expr_str: str) -> str:
    expr = parse_expr(expr_str)
    expr = nsimplify(expr)
    return str(expr)


def synthesize_from_qe_formula(
    exs: IOExamples,
    f: ExprTemplate,
) -> str | None:
    if not f.ctx_ios:
        return None  # no constants

    len_var_name = (
        VAR_NAME.PREVIOUS_LENGTH
        if VAR_NAME.PREVIOUS_LENGTH in next(iter(f.ctx_ios.values()))[0][0]
        else "prev_n"
    )

    exs_by_len = {}
    for ctx, val in exs:
        # TODO: remove this hack
        exs_by_len.setdefault(ctx.get(len_var_name, -1) + 1, []).append((ctx, val))

    f = extend_examples_for_qe_formula(exs_by_len, f, len_var_name)
    if not f.ctx_ios:
        # the formula is likely invalid
        return None

    l = symbols(len_var_name)
    denominators = [
        l - l + 1,  # 1
        l,
        l + 1,
        l - 1,
    ]

    const_expr = {}

    for unk in f.ctx_ios.keys():
        for denominator in denominators:
            mapped_exs = [
                (
                    ctx[len_var_name],
                    round(v * denominator.subs(l, ctx[len_var_name])),
                )
                for ctx, v in f.ctx_ios[unk]
                if len_var_name in ctx
            ]
            if not mapped_exs:
                continue

            expr = interpolate(mapped_exs, l)
            if not expr.is_number and is_const_func_overfit(mapped_exs, expr, l):
                continue
            if (
                expr.is_number
                and len(mapped_exs) == 1
                and mapped_exs[0][1] == expr == mapped_exs[0][0]
                and all(v == expr for _, v in f.ctx_ios[unk])
            ):
                expr = l  # edge case
            if expr.is_number and not all(v == expr for _, v in f.ctx_ios[unk]):
                continue

            expr = expr / denominator if denominator != 1 else expr
            const_expr[unk] = f"({expr})"
            break
        else:
            logging.warn(f"Failed to interpolate {unk} in {f.expr}")
            return None

    expr = f.expr.format_map(const_expr)
    expr = simplify_expr_str(expr)
    expr_str = f"({expr})"
    if not eval_eq_exs(exs, expr_str):
        logging.warn(f"undetected overfitting: {expr_str}")
        return None
    return expr_str


def overapprox_solve_eqn_group(eqns, symbols, pct=0.4):
    solns = solve(eqns, *symbols, dict=True)
    if solns:
        soln = solns[0]
        if any(not v.is_constant() for v in soln.values()):
            return None
        return soln

    # overapproximate
    soln_values = {s: [] for s in symbols}
    num_unks = len(symbols)
    num_eqns_comb = math.comb(len(eqns), num_unks)

    min_samples = min(25, num_eqns_comb)
    num_samples = min(max(min_samples, int(num_eqns_comb * pct)), 50)
    for comb in tqdm(
        random.sample(list(combinations(eqns, num_unks)), num_samples),
        desc="solving eqn groups",
        total=num_samples,
    ):
        solns = solve(comb, *symbols, dict=True)
        if not solns:
            continue

        [soln] = solns

        if any(not v.is_constant() for v in soln.values()):
            continue

        if any(v.free_symbols for v in soln.values()):
            continue

        for s, v in soln.items():
            soln_values[s].append(float(v))

        variances = [np.var(vs) for vs in soln_values.values()]
        if any(v > 1 for v in variances):
            return None

    soln_values = {s: round(np.mean(vs), 3) for s, vs in soln_values.items() if vs}
    if len(soln_values) != num_unks:
        return None
    return soln_values


def extend_examples_for_qe_formula(
    exs_by_len: dict[int, IOExamples],
    f: ExprTemplate,
    len_var_name: str,
) -> ExprTemplate:
    qe_exs: dict[str, list[IOExample]] = {c: list(vs) for c, vs in f.ctx_ios.items()}

    const_symbs = {c: Symbol(c) for c in f.ctx_ios.keys()}

    expr_src = f.expr.format_map({c: c for c in f.ctx_ios.keys()})
    for len_stream, exs in exs_by_len.items():
        if len(exs) < len(const_symbs):
            continue

        eqns = []

        for ctx, val in exs:
            eqn = eval(
                expr_src,
                {
                    **{k: float(v) for k, v in ctx.items() if isinstance(v, Number)},
                    **const_symbs,
                    **MATH_ENV,
                },
            )
            eqns.append(eqn - val)

        soln = overapprox_solve_eqn_group(eqns, const_symbs.values())
        if soln is None:
            # the formula is likely invalid
            return ExprTemplate(f.expr, frozendict())

        for c, v in soln.items():
            qe_exs[c.name].append(
                (frozendict({len_var_name: len_stream - 1}), sympy_const_to_number(v))
            )

    # remove duplicates
    qe_exs = {c: list(set(vs)) for c, vs in qe_exs.items()}

    qe_exs_: dict[str, IOExamples] = qe_exs  # type: ignore
    return ExprTemplate(f.expr, frozendict(qe_exs_))


def is_const_func_overfit(
    exs: IOExamples,
    eqn: Expr | Poly,
    eqn_var: Symbol,
) -> bool:
    # check if degree matches num examples
    assert (isinstance(eqn, Expr) or isinstance(eqn, Poly)) and isinstance(
        eqn_var, Symbol
    )
    deg = degree(eqn, gen=eqn_var)  # type: ignore
    if not deg.is_constant():
        return True
    if int(deg) > 1 and int(deg) + 1 >= len(exs):
        return True

    for ctx, val in exs:
        if eqn.subs(eqn_var, ctx) != val:
            return True

    return False


def synthesize_from_templates(
    exs: IOExamples,
    templates: Sequence[ExprTemplate],
) -> str | None:
    # check if the template is already valid
    for t in templates:
        consts = {k: vs[0][1] for k, vs in t.ctx_ios.items()}
        prog = t.expr.format(**consts)
        if eval_eq_exs(exs, prog):
            return prog

    # check if any trivial solution QE fails to find
    params = exs[0][0].keys()
    for p in params:
        for ctx, val in exs:
            if p not in ctx or ctx[p] != val:
                break
        else:
            return p

    # synthesize exprs for constants
    for t in templates:
        r = synthesize_from_qe_formula(exs, t)
        if r is not None:
            return r

    return None
