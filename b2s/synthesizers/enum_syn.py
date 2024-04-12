import logging
import math
import numbers
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Sequence

from sympy import symbols

import b2s.synthesizers.smart_expr_enumerator as expr_enum
from b2s.sketch_types import EnvState, IOExamples
from b2s.utils import timeout

EVAL_ENV = {"log": math.log, "exp": math.exp}


def eval_no_err(e: str, vals: EnvState) -> Any:
    try:
        with timeout(1):
            return eval(e, {**vals, **EVAL_ENV})
    except ZeroDivisionError:
        return None
    except NameError as ex:
        logging.warning(f"NameError when evaluating expressions: {ex}")
        return None
    except Exception:
        return None


def eval_from_io(valuations: Sequence[EnvState], expression: str) -> list[Any]:
    return [eval_no_err(expression, valuation) for valuation in valuations]


def eval_exs(exs: IOExamples, expr: str) -> list[Any]:
    return eval_from_io([ex[0] for ex in exs], expr)


def eval_eq_exs(exs: IOExamples, expr: str) -> bool:
    return eval_eq([ex[0] for ex in exs], expr, [ex[1] for ex in exs])


def eval_eq(valuations: Sequence[EnvState], expr: str, expected: list[Any]) -> bool:
    for valuation, exp in zip(valuations, expected):
        if not compare_values(eval_no_err(expr, valuation), exp):
            return False
    return True


def compare_values(v1: Any, v2: Any) -> bool:
    if isinstance(v1, numbers.Number) and isinstance(v2, numbers.Number):
        try:
            return v1 == v2 or math.isclose(float(v1), float(v2), rel_tol=1e-6, abs_tol=0)  # type: ignore
        except Exception:
            return False
    elif type(v1) == type(v2):
        return v1 == v2
    else:
        return False


def synthesize_from_io(
    examples: IOExamples,
    *,
    consts: set[str] | None = None,
    allow_trivial: bool = True,
    return_all: bool = False,
    max_num_terms: int | None = None,
    dedup: bool = False,
    timeout: int = 0,
) -> str | list[str]:
    start_time = time.time()

    variables = set(examples[0][0].keys())
    if not variables:
        return []

    consts = consts or {"1", "2", "3"}
    terms = consts.copy()

    for v in variables:
        if isinstance(examples[0][0][v], numbers.Number):
            terms.add(v)
        else:
            if eval_eq_exs(examples, v):
                return v
    
    if not isinstance(examples[0][1], numbers.Number):
        return []


    result = []
    sym_results = []

    stopped = False

    n_checked = 0
    max_num_terms = max_num_terms or len(terms) + 2
    for n_vars in range(1, max_num_terms + 1):
        if stopped:
            break

        boxed_vars = [expr_enum.BoxedString("v") for _ in range(n_vars)]
        exprs = list(expr_enum.enum_with_dedup(expr_enum.smart4, boxed_vars))
        if n_vars == 1:
            exprs = boxed_vars

        for vs in combinations_with_replacement(terms, n_vars):
            if stopped:
                break

            if (
                any(v > 2 for v in Counter(vs).values())
                or sum(v == 2 for v in Counter(vs).values()) > 1
            ):
                continue

            for boxed, v in zip(boxed_vars, vs):
                boxed.value = v

            for expr in exprs:
                if timeout > 0 and time.time() - start_time > timeout:
                    stopped = True
                    break

                expr_str = str(expr)

                n_checked += 1
                if not eval_eq_exs(examples, expr_str):
                    continue

                if dedup:
                    # TODO: obvisously you need more variables here
                    eval_locals = {"prev_len": symbols("prev_len")}
                    sym_expr = eval(expr_str, eval_locals)

                    if not allow_trivial:
                        if (
                            isinstance(sym_expr, (int, float, complex))
                            or sym_expr.is_Number
                        ):
                            continue

                    if any(sym_expr == s for s in sym_results):
                        continue

                    sym_results.append(sym_expr)

                result.append(f"({expr_str})")
                if not return_all:
                    return result[0]
    return result
