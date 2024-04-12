"""
func = "geometric_standard_deviation_twopass"
find_good_input = false
"""
import math
from math import exp, log

import numpy as np

def geometric_standard_deviation_onepass(xs):
    s = 0
    sq_s = 0
    for x in xs:
        s += math.log(x)
        sq_s += math.log(x) ** 2

    var = sq_s / len(xs) - (s / len(xs)) ** 2
    return math.exp(np.abs(var**0.5))


def geometric_standard_deviation_twopass(xs):
    s = 0
    for x in xs:
        s += log(x)
    avg = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (log(x) - avg) ** 2
    return exp(abs(sq_s / len(xs)) ** 0.5)

def geometric_standard_deviation_offline(xs):
    s = 0
    for x in xs:
        s += log(x)
    avg = s / len(xs)
    sq_s = 0
    for x in xs:
        sq_s += (log(x) - avg) ** 2
    return exp(abs(sq_s / len(xs)) ** 0.5)

def geometric_standard_deviation_online(prev_out, prev_s, prev_sq_s, prev_len, x):
    l = prev_len + 1
    s = prev_s + log(x)
    sq_s = prev_sq_s + (log(x) - prev_s / prev_len) * (log(x) - s / l)
    return exp(abs(sq_s / l) ** 0.5), s, sq_s, l


def geometric_standard_deviation_welford_onepass(xs):
    n = 0
    mean = 0
    M2 = 0
    for x in xs:
        n += 1
        delta = math.log(x) - mean
        mean += delta / n
        M2 += delta * (math.log(x) - mean)
    return math.exp(np.abs(M2 / n) ** 0.5)


from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import gstd


@given(st.lists(st.floats(min_value=1, max_value=10000), min_size=2))
def test_geometric_std(xs):
    expected_gstd = gstd(xs, ddof=0)

    v = np.exp(np.log(np.array(xs)).std())
    assert np.isclose(v, expected_gstd)
    assert np.isclose(geometric_standard_deviation_onepass(xs), expected_gstd)
    assert np.isclose(geometric_standard_deviation_twopass(xs), expected_gstd)
    assert np.isclose(geometric_standard_deviation_welford_onepass(xs), expected_gstd)


if __name__ == "__main__":
    test_geometric_std()
