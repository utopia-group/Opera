"""
func = "coefficient_of_variation"
"""


def coefficient_of_variation(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs) / avg

def cv_offline(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs) / avg

def cv_opera(prev_out, prev_s, prev_sq_s, prev_len, x):
    l = prev_len + 1
    s = prev_s + x
    avg = s / l
    sq_s = (prev_len**2*x**2 - 2*prev_len*prev_s*x + prev_s**2 + prev_sq_s*(prev_len**2 + prev_len))/(prev_len**2 + prev_len)
    return (sq_s / l / avg), s, sq_s, l

def cv_online(prev_out, prev_s, prev_sq_s, prev_len, x):
    l = prev_len + 1
    s = prev_s + x
    avg = s / l
    sq_s = prev_sq_s + (x - avg) * (x - prev_s / prev_len)
    return (sq_s / l / avg), s, sq_s, l



import math

from hypothesis import assume, given
from hypothesis import strategies as st
from numpy import mean, std


@given(st.lists(st.floats(min_value=-10e5, max_value=10e5), min_size=2))
def test_var(xs):
    assume(mean(xs) != 0)
    expected_var = std(xs) / mean(xs)
    assert math.isclose(coefficient_of_variation(xs), expected_var)


if __name__ == "__main__":
    test_var()
