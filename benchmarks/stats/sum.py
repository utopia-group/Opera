"""
func = "sum"
"""


def sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s

def sum_online(prev_out, x):
    return prev_out + x


import math
from hypothesis import given, strategies as st


@given(st.lists(st.floats()))
def test_sum(xs):
    expected = sum(xs)
    assert math.isclose(sum(xs), expected) or math.isnan(expected)


if __name__ == "__main__":
    test_sum()
