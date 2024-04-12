"""
func = "geometric_mean"
"""

import math

def geometric_mean(xs):
    p = 1
    for x in xs:
        p *= x
    return p ** (1 / len(xs))


def geometric_mean_online(prev_out, prev_len, x):
    prev_p = prev_out**prev_len
    l = prev_len + 1
    p = prev_p * x
    return p ** (1 / l), l


def geometric_mean_log(xs):
    s = 0
    for x in xs:
        s += math.log(x)
    return math.exp(s / len(xs))


def geometric_mean_log_online(prev_out, prev_len, x):
    prev_s = math.log(prev_out) * prev_len
    l = prev_len + 1
    s = prev_s + math.log(x)
    return math.exp(s / l), l


from functools import reduce
from statistics import geometric_mean

from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.floats(min_value=1, max_value=10000), min_size=1))
def test_geometric_mean(xs):
    expected_gm = geometric_mean(xs)
    assert math.isclose(geometric_mean_product(xs), expected_gm)
    assert math.isclose(geometric_mean_log(xs), expected_gm)
    assert math.isclose(
        reduce(lambda acc, x: geometric_mean_product_online(*acc, x), xs, (0, 0))[0],
        expected_gm,
    )

    init_state = (geometric_mean(xs[:1]), 1)
    assert math.isclose(
        reduce(lambda acc, x: geometric_mean_log_online(*acc, x), xs[1:], init_state)[
            0
        ],
        expected_gm,
    )


if __name__ == "__main__":
    test_geometric_mean()
