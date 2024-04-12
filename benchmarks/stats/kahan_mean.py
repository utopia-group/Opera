"""
func = "kahan_mean"
"""


def kahan_mean(xs: list[float]) -> float:
    mu = 0.0
    c = 0.0
    n = 0
    y = 0.0
    t = 0.0
    for x in xs:
        n += 1
        y = (x - mu) / n - c
        t = mu + y
        c = (t - mu) - y
        mu = t
    return mu


def kahan_mean_online(prev_mu: float, prev_c: float, prev_n: int, x: float):
    n = prev_n + 1
    y = (x - prev_mu) / n - prev_c
    t = prev_mu + y
    c = (t - prev_mu) - y
    mu = t
    return mu, c, n


import math
from functools import reduce

from hypothesis import assume, given
from hypothesis import strategies as st


@given(st.lists(st.floats(min_value=1, max_value=9.223372036854776e18), min_size=1))
def test_mean(xs):
    expected = sum(xs) / len(xs)
    assume(not math.isnan(expected) and not math.isinf(expected))
    assert math.isclose(kahan_mean(xs), expected), (kahan_mean(xs), expected)
    assert math.isclose(
        reduce(lambda acc, x: kahan_mean_online(*acc, x), xs, (0.0, 0.0, 0))[0],
        expected,
    )


if __name__ == "__main__":
    test_mean()
