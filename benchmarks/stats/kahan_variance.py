"""
func = "kahan_variance"
find_good_input = false
"""


def kahan_variance(xs: list[float]) -> float:
    n = 0
    mu = 0.0
    cmu = 0.0
    csigma_sq = 0.0
    sigma_sq = 0.0

    for x in xs:
        n += 1
        prev_mu = mu

        y = (x - prev_mu) / n - cmu
        t = mu + y
        cmu = (t - mu) - y
        mu = t

        y = (x - prev_mu) * (x - mu) - csigma_sq
        t = sigma_sq + y
        csigma_sq = (t - sigma_sq) - y
        sigma_sq = t

    return sigma_sq / n

def kahan_variance_online(prev_out: float, prev_n, prev_mu, prev_cmu, prev_sigma_sq, prev_csigma_sq, x):
    n = prev_n + 1
    prev_mu = prev_mu
    prev_cmu = prev_cmu
    prev_sigma_sq = prev_sigma_sq
    prev_csigma_sq = prev_csigma_sq

    y = (x - prev_mu) / n - prev_cmu
    t = prev_mu + y
    prev_cmu = (t - prev_mu) - y
    prev_mu = t

    y = (x - prev_mu) * (x - prev_mu) - prev_csigma_sq
    t = prev_sigma_sq + y
    prev_csigma_sq = (t - prev_sigma_sq) - y
    prev_sigma_sq = t

    return prev_sigma_sq / n, n, prev_mu, prev_cmu, prev_sigma_sq, prev_csigma_sq


import math
from functools import reduce

from hypothesis import given
from hypothesis import strategies as st
from numpy import abs, var


@given(st.lists(st.floats(min_value=-10e5, max_value=10e5), min_size=2))
def test_var(xs):
    expected = var(xs)
    assert math.isclose(kahan_variance(xs), expected)


if __name__ == "__main__":
    test_var()
