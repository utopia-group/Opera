"""
func = "skewness_definition"
find_good_input = false
unroll_depth = 4
"""

import numpy as np


def skewness_alternative(xs):
    s = 0
    cube_s = 0
    for x in xs:
        s += x
        cube_s += x**3
    mean = s / len(xs)
    mean3 = cube_s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - mean) ** 2
    sigma = (sq_s / len(xs)) ** 0.5
    return (mean3 - 3 * mean * sigma**2 - mean**3) / sigma**3

def skewness_online(prev_out, prev_s, prev_sq_s, prev_m3, prev_len, x):
    new_len = prev_len + 1
    new_s = prev_s + x
    
    avg = new_s / new_len
    delta = x - prev_s / prev_len
    delta_n = delta / new_len
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * prev_len

    new_m3 = prev_m3 + term1 * delta_n * (new_len - 2) - 3 * delta_n * prev_sq_s
    new_sq_s = prev_sq_s + term1

    new_sigma = (new_sq_s / new_s) ** 0.5
    return (new_m3 / new_len / (new_sigma ** 3)), new_s, new_m3, new_len, new_m3


def skewness_opera(prev_out, prev_s, prev_sq_s, prev_m3, prev_len, x__):
    new_len = prev_len + 1
    new_s = prev_s + x__
    new_avg = new_s / new_len

    new_m3 = ((prev_len - 1)*(prev_len**3*x__**3 - 3*prev_len**2*prev_s*x__**2 + 3*prev_len*prev_s**2*x__ + prev_m3*(prev_len**4 + 2*prev_len**3 + prev_len**2)/(prev_len - 1) - prev_s**3 + prev_s*prev_sq_s*(3*prev_len**2 + 3*prev_len)/(prev_len - 1) + prev_sq_s*x__*(-3*prev_len**3 - 3*prev_len**2)/(prev_len - 1))/(prev_len**4 + 2*prev_len**3 + prev_len**2))
    new_sq_s = prev_sq_s + (x__ - prev_s / prev_len) * (x__ - new_avg)

    new_sigma = (new_sq_s / new_s) ** 0.5
    return (new_m3 / new_len / (new_sigma ** 3)), new_s, new_m3, new_len, new_m3

def skewness_offline(xs):
    s = 0
    for x in xs:
        s += x
    mean = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - mean) ** 2
    sigma = (sq_s / len(xs)) ** 0.5

    m3 = 0
    for x in xs:
        m3 += (x - mean) ** 3
    return (m3 / len(xs)) / sigma**3


def skewness_definition(xs):
    s = 0
    for x in xs:
        s += x
    mean = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - mean) ** 2
    sigma = (sq_s / len(xs)) ** 0.5

    m3 = 0
    for x in xs:
        m3 += (x - mean) ** 3
    return (m3 / len(xs)) / sigma**3

def skewness_vars(xs):
    s = 0
    for x in xs:
        s += x
    mean = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - mean) ** 2

    m3 = 0
    for x in xs:
        m3 += (x - mean) ** 3
    return locals()

# (18*prev_sq_s*prev_s - 54*prev_sq_s*x__ + 72*prev_out - prev_s**3 + 9*prev_s**2*x__ - 27*prev_s*x__**2 + 27*x__**3)/72
TRUTH = r"({c1}*prev_sq_s*prev_s - {c2}*prev_sq_s*x__ + {c3}*prev_out - prev_s**3 + {c4}*prev_s**2*x__ - {c5}*prev_s*x__**2 + {c6}*x__**3)/{c7}".format(
    c1="(3*prev_len*(prev_len+1) / (prev_len-1))",
    c2="(3*prev_len**2*(prev_len+1) / (prev_len-1))",
    c3="(prev_len**2 * (prev_len+1)**2 / (prev_len-1))",
    c4="3*prev_len",
    c5="3*prev_len**2",
    c6="prev_len**3",
    c7="(prev_len**2 * (prev_len+1)**2 / (prev_len-1))",
)
STMT = TRUTH

def skewness_test(xs):
    init = xs[:2]
    rest = xs[2:]

    state = dict(prev_len=2)
    for x in rest:
        state.update(skewness_vars(init))

        new_state = {
            "prev_len": state["prev_len"],
            "prev_sq_s": state["sq_s"],
            "prev_s": state["s"],
            "prev_m3": state["m3"],
            "prev_out": state["m3"],
            "x__": x
        }
        new_m3 = eval(STMT, new_state)
        
        init += [x]
        expected_m3 = skewness_definition(init)
        assert np.isclose(expected_m3, new_m3, atol=0.01), f"{expected_m3} != {new_m3} with len_stream={state['prev_len'] + 1}"
        state["prev_len"] += 1


def skewness_welford_alternative(xs):
    n = 0
    mean = 0
    cube_mean = 0
    m2 = 0
    for x in xs:
        n += 1
        old_mean = mean
        mean += (x - old_mean) / n
        cube_mean += (x**3 - cube_mean) / n
        m2 += (x - old_mean) * (x - mean)

    sigma = (m2 / n) ** 0.5
    return (cube_mean - 3 * mean * sigma**2 - mean**3) / (sigma**3)


import math

from hypothesis import given, assume
from hypothesis import strategies as st
from scipy.stats import skew as sk


@given(
    st.lists(
        st.floats(min_value=-1e5, max_value=1e5),
        min_size=3,
        max_size=100,
        unique=True,
    )
)
def test_moments(xs):
    assume(sk(xs) != 0 and not np.isclose(np.std(xs), 0) and not any(np.isnan(xs)))
    # assert np.isclose(skewness_definition(xs), sk(xs))
    # assert np.isclose(skewness_alternative(xs), sk(xs))
    # assert np.isclose(skewness_welford_alternative(xs), sk(xs))
    # assert np.isclose(skewness_terriberry(xs), sk(xs))
    skewness_test(xs)


if __name__ == "__main__":
    test_moments()
