"""
func = "tmean"

[[args]]
limit_u = 75
limit_l = 25
"""


def tmean(xs: list[float], limit_l: float, limit_u: float) -> float:
    s = 0.0
    n = 0
    for x in xs:
        if x >= limit_l:
            if x <= limit_u:
                s += x
                n += 1
    return s / (n + 1)


def tmean_online(
    prev_out: float, prev_len: int, x: float, limits: tuple[float, float]
) -> tuple[float, int]:
    if not (limits[0] <= x <= limits[1]):
        return prev_out, prev_len
    return (prev_out * prev_len + x) / (prev_len + 1), prev_len + 1

def tmean_opera(
    prev_out: float, prev_s: float, prev_len: int, x: float, limits: tuple[float, float]
) -> tuple[float, int]:
    """
    Compute the trimmed mean incrementally.

    This function finds the arithmetic mean of given values, ignoring values outside the given limits.
    """
    if not (limits[0] <= x <= limits[1]):
        return prev_out, prev_len
    return (prev_s + x) / (prev_len + 1), prev_len + 1


import math
from functools import reduce

from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.stats import tmean as tmean_ref


@given(
    st.lists(st.floats(min_value=0, max_value=10000), min_size=1),
    st.tuples(
        st.floats(min_value=0, max_value=100),
        st.floats(min_value=9000, max_value=10000),
    ),
)
def test_tmean(xs, limits):
    assume(
        limits[0] < limits[1]
        and len(list(filter(lambda x: limits[0] <= x <= limits[1], xs))) > 0
    )
    expected = tmean_ref(xs, limits)
    assert math.isclose(tmean(xs, limits), expected)

    r = reduce(lambda acc, x: tmean_online(*acc, x, limits), xs, (0, 0))[0]
    assert math.isclose(r, expected)


if __name__ == "__main__":
    test_tmean()
