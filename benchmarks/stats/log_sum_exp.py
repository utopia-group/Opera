"""
func = "log_sum_exp"
find_good_input = false
"""

import math
from math import log, exp

# https://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html

def log_sum_exp(xs: list[float]) -> float:
    alpha = 1.0
    for x in xs:
        if x > alpha:
            alpha = x

    s = 0.0
    for x in xs:
        s += exp(x - alpha)

    return alpha + log(s)

def log_sum_exp_online(prev_out, prev_alpha, prev_s, x):
    if x <= prev_alpha:
        s = prev_s + exp(x - prev_alpha)
        alpha = prev_alpha
    else:
        s = prev_s * exp(prev_alpha-x) + 1
        alpha = x
    return alpha + log(s)

def log_sum_exp_stream(xs: list[float]) -> float:
    alpha = -math.inf
    s = 0.0

    for x in xs:
        if x <= alpha:
            s += math.exp(x - alpha)
        else:
            s *= math.exp(alpha - x)
            s += 1.0
            alpha = x
    return alpha + math.log(s)

def log_sum_exp_stream2(xs: list[float]) -> float:
    alpha = -math.inf
    s = 0.0

    for x in xs:
        if x <= alpha:
            s += math.exp(x - alpha)
        else:
            # s = s * exp(alpha-x) + exp(x - x)
            # s = s + exp(-alpha)*exp(x)
            # s = (s + exp(-alpha)*exp(x))*exp(-x) / exp(-alpha)
            # s = (s + exp(-alpha)*exp(x))*exp(-x) / exp(-x)
            s = s * exp(alpha-x) + 1
            alpha = x
    return alpha + math.log(s)


import math
from functools import reduce

from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.floats(min_value=1, max_value=9.223372036854776e18), min_size=1))
def main(xs):
    assert math.isclose(log_sum_exp(xs), log_sum_exp_stream2(xs))


if __name__ == "__main__":
    main()
