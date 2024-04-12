"""
func = "kahan_sum"
"""


def kahan_sum(xs: list[float]) -> float:
    s = 0.0
    c = 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def kahan_sum_online(prev_out: float, prev_c: float, x: float) -> tuple[float, float]:
    s = prev_out
    c = prev_c

    y = x - c
    t = s + y
    c = (t - s) - y
    s = t
    return s, c


import math
from functools import reduce

from hypothesis import assume, given
from hypothesis import strategies as st


@given(st.lists(st.floats()))
def test_sum(xs):
    expected = sum(xs)
    assume(not math.isnan(expected) and not math.isinf(expected))
    assert math.isclose(kahan_sum(xs), expected)
    assert math.isclose(
        reduce(lambda acc, x: kahan_sum_online(*acc, x), xs, (0.0, 0.0))[0], expected
    )


if __name__ == "__main__":
    test_sum()
