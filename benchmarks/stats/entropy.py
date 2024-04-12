"""
func = "entropy"
find_good_input = false
"""

r"""
prev_s = x0 + x1 + x2;
prev_ent = -(x0 / prev_s) * log(x0 / prev_s) - (x1 / prev_s) * log(x1 / prev_s) - (x2 / prev_s) * log(x2 / prev_s);
prev_ent = -(x0 / prev_s) * (log(x0) - log(prev_s)) - (x1 / prev_s) * (log(x1) - log(prev_s)) - (x2 / prev_s) * (log(x2) - log(prev_s));

s = x0 + x1 + x2 + x3;
ent = -(x0 / s) * log(x0 / s) - (x1 / s) * log(x1 / s) - (x2 / s) * log(x2 / s) - (x3 / s) * log(x3 / s);
ent = -(x0 / s) * (log(x0) - log(s)) - (x1 / s) * (log(x1) - log(s)) - (x2 / s) * (log(x2) - log(s)) - (x3 / s) * (log(x3) - log(s));

t0 = log(x0);
t1 = log(x1);
t2 = log(x2);
t3 = log(x3);
tps = log(prev_s);
ts = log(s);

phi_1 := (prev_s = x0 + x1 + x2) and (prev_ent * prev_s = -(x0) * (t0 - tps) - (x1) * (t1 - tps) - (x2) * (t2 - tps));
phi_2 := (s = x0 + x1 + x2 + x3) and (ent * s = -(x0) * (t0 - ts) - (x1) * (t1 - ts) - (x2) * (t2 - ts) - (x3) * (t3 - ts));
phi := ex({x0, x1, x2, t0, t1, t2}, phi_1 and phi_2);

ent = prev_ent * prev_sum / new_sum + entr(p / new_sum) + entr(prev_sum / new_sum)

QE Result:
- s = prev_s + x3
- ent = prev_ent*prev_s/s + (-s*tps + s*ts - t3*x3 + tps*x3)/s

- ent = prev_ent*prev_s/s + (-s*log(prev_s) + s*log(s) - log(x3)*x3 + log(prev_s)*x3)/s
- ent = prev_ent*prev_s/s + (-prev_s*log(prev_s) + s*log(s) - log(x3)*x3)/s
"""

import math
from math import log
from typing import *

import numpy as np


def entr(x):
    if x > 0:
        return -x * log(x)
    elif x == 0:
        return 0
    else:
        return np.nan


def rel_entr(x, y):
    if x > 0 and y > 0:
        return x * log(x / y)
    elif x == 0 and y >= 0:
        return 0
    else:
        return np.nan


def entropy_local(pk, qk=None, base: Optional[float] = None):
    s = 0.0
    for p in pk:
        s += p

    ent_s = 0.0
    if qk is None:
        for p in pk:
            ent_s += entr(p / s)
    else:
        qs = sum(qk)
        for p, q in zip(pk, qk):
            ent_s += rel_entr(p / s, q / qs)

    if base is not None:
        ent_s /= math.log(base)

    return ent_s


def entropy(xs):
    s = 0.0
    for p in xs:
        s += p

    ent_s = 0.0
    for p in xs:
        ent_s += -(p / s) * log(p / s)
    return ent_s

def entropy_offline(xs):
    s = 0.0
    for p in xs:
        s += p

    ent_s = 0.0
    for p in xs:
        ent_s += -(p / s) * log(p / s)
    return ent_s

def entropy_online(prev_out, prev_s, prev_ent_s, x__):
    s = prev_s + x__
    ent_s = (prev_ent_s*prev_s - prev_s*log(prev_s) + prev_s*log(prev_s + x__) - x__*log(x__) + x__*log(prev_s + x__))/(prev_s + x__)
    return ent_s, s, ent_s

def log(x):
    if x == 0:
        return 0
    else:
        return log(x)

def incr_entropy(prev_ent, prev_sum, p):
    new_sum = prev_sum + p
    x__ = p
    return (
        prev_ent * prev_sum / new_sum + (-prev_sum*log(prev_sum) + new_sum*log(new_sum) - log(p)*p)/new_sum,
        new_sum,
    )


from functools import reduce

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import entropy as scipy_entropy


@st.composite
def same_len_lists(draw):
    n = draw(st.integers(min_value=3, max_value=50))
    fixed_length_list = st.lists(
        st.floats(min_value=1, max_value=1e16), min_size=n, max_size=n, unique=True
    )

    return (draw(fixed_length_list), draw(fixed_length_list))


@given(same_len_lists(), st.floats(min_value=2, max_value=50))
def test_entropy(list_pair, base):
    xs, ys = list_pair
    assert np.isclose(entropy_local(xs, ys, base), scipy_entropy(xs, ys, base))
    assert np.isclose(entropy(xs), scipy_entropy(xs))

    incr_entr = reduce(lambda acc, x: incr_entropy(*acc, x), xs, (0.0, 0.0))[0]
    expected_entr = entropy(xs)
    assert np.isclose(incr_entr, expected_entr), (incr_entr, expected_entr)


if __name__ == "__main__":
    test_entropy()
