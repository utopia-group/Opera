"""
func = "tvar"

[[args]]
limit_u = 90
"""


def tvar(xs: list[float], limit_u: float) -> float:
    """
    Compute the trimmed variance.

    This function finds the arithmetic mean of given values, ignoring values outside the given limits.
    """
    s = 0.0
    n = 0
    for x in xs:
        if x <= limit_u:
            s += x
            n += 1
    avg = s / n

    sq_s = 0.0
    for x in xs:
        if x <= limit_u:
            sq_s += (x - avg) ** 2
    return sq_s / n


def tvar_offline(xs: list[float], limit_u: float) -> float:
    s = 0.0
    n = 0
    for x in xs:
        if x <= limit_u:
            s += x
            n += 1
    avg = s / n

    sq_s = 0.0
    for x in xs:
        if x <= limit_u:
            sq_s += (x - avg) ** 2
    return sq_s / n


def tvar_online(
    prev_out: float,
    prev_n: int,
    prev_s: float,
    prev_sq_s: float,
    x: float,
    limit_u: float,
) -> tuple[float, int, float, float]:
    """
    Compute the trimmed mean incrementally.

    This function finds the arithmetic mean of given values, ignoring values outside the given limits.
    """
    if x > limit_u:
        return prev_out, prev_n, prev_s, prev_sq_s
    n = prev_n + 1
    s = prev_s + x
    avg = s / n
    sq_s = prev_sq_s + (x - avg) * (x - prev_s / prev_n) if prev_n else 0
    return sq_s / n, n, s, sq_s


def tvar_opera(
    prev_out: float,
    prev_n: int,
    prev_s: float,
    prev_sq_s: float,
    x: float,
    limit_u: float,
) -> tuple[float, int, float, float]:
    """
    Compute the trimmed mean incrementally.

    This function finds the arithmetic mean of given values, ignoring values outside the given limits.
    """
    if x > limit_u:
        return prev_out, prev_n, prev_s, prev_sq_s
    n = prev_n + 1
    s = prev_s + x
    sq_s = prev_sq_s + ((prev_n**2*x**2 - 2*prev_n*prev_s*x + prev_s**2 + prev_sq_s*(prev_n**2 + prev_n))/(prev_n**2 + prev_n)) if prev_n else 0
    return sq_s / n, n, s, sq_s


import math
from functools import reduce

from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.stats import tvar as tvar_ref


@given(
    st.lists(st.floats(min_value=0, max_value=10000), min_size=1),
    st.tuples(
        st.floats(min_value=0, max_value=100),
        st.floats(min_value=9000, max_value=10000),
    ),
)
def main(xs, limits):
    assume(
        limits[0] < limits[1]
        and len(list(filter(lambda x: limits[0] <= x <= limits[1], xs))) > 0
    )
    expected = tvar_ref(xs, (None, limits[1]), ddof=0)
    assert math.isclose(tvar(xs, limits[1]), expected)

    r = reduce(lambda acc, x: tvar_online(*acc, x, limits[1]), xs, (0, 0, 0, 0))[0]
    assert math.isclose(r, expected)


if __name__ == "__main__":
    main()
