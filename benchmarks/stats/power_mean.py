"""
func = "power_mean"
find_good_input = false

[[args]]
p = 2

[[args]]
p = 3
"""

import math

"""
rlset reals;
off nat;

phi_1 := ret = expt(( expt(xs0, p) + expt(xs1, p) + expt(xs2, p) ) / new_len, 1 / p);
phi_2 := ret = expt(( expt(xs0, p) + expt(xs1, p) ) / prev_len, 1 / p);

phi := ex({xs0, xs1}, phi_1 and phi_2);
"""

"""
rlset reals;
off nat;

phi_0 := xs0_p = expt(xs0, p) and xs1_p = expt(xs1, p) and xs2_p = expt(xs2, p);
phi_1 := ret = expt(ret_p, 1 / p) and ret_p * new_len = ( xs0_p + xs1_p + xs2_p );
phi_2 := prev_ret = expt(prev_ret_p, 1 / p) and prev_ret_p * prev_len = ( xs0_p + xs1_p );
phi := ex({xs0, xs1, ret_p, prev_ret_p}, phi_0 and phi_1 and phi_2);
"""



"""
rlset reals;
off nat;

phi_1 := ret = expt(ret_p, 1 / p) and ret_p * new_len = ( xs0_p + xs1_p + xs2_p );
phi_2 := prev_ret = expt(prev_ret_p, 1 / p) and prev_ret_p * prev_len = ( xs0_p + xs1_p );
psi := prev_ret_p = expt(prev_ret, p) and ret_p = expt(ret, p);
phi := ex({xs0, xs1, xs0_p, xs1_p, xs2_p, prev_ret_p, ret_p}, phi_1 and phi_2 and psi);
"""

"""
psi := ex({ret_p, prev_ret_p}, xs2**p - new_len*ret_p + prev_len*prev_ret_p = 0 and
ret_p**(1/p) - ret = 0 and 
prev_ret_p**(1/p) - prev_ret = 0 and
ret_p = ret**p and prev_ret_p = prev_ret**p);


((prev_ret**p*prev_len + xs2**p) / new_len) ** (1/p) = ret

psi := ex({y}, y = x and z = 2**y);




"""

SHANKARA = """
phi_1 := ret = expt(ret_p, 1 / p) and ret_p * new_len = ( xs0_p + xs1_p + xs2_p );
phi_2 := prev_ret = expt(prev_ret_p, 1 / p) and prev_ret_p * prev_len = ( xs0_p + xs1_p );
phi := ex({xs0_p, xs1_p, xs2_p, ret_p, prev_ret_p}, phi_1 and phi_2);

phi_0 := xs2_p = expt(xs2, p);
phi_1 := ret = expt(ret_p, 1 / p) and ret_p * new_len = ( xs0_p + xs1_p + xs2_p );
phi_2 := prev_ret = expt(prev_ret_p, 1 / p) and prev_ret_p * prev_len = ( xs0_p + xs1_p );
psi := prev_ret_p = expt(prev_ret, p) and ret_p = expt(ret, p);
phi := ex({xs0, xs1, xs0_p, xs1_p, xs2_p, prev_ret_p, ret_p}, phi_0 and phi_1 and phi_2 and psi);
"""


def power_mean(xs, p=2):
    s = 0
    for x in xs:
        s += x**p
    return (s / len(xs)) ** (1 / p)

def power_mean_offline(xs, p=2):
    s = 0
    for x in xs:
        s += x**p
    return (s / len(xs)) ** (1 / p)

def power_mean_online(prev_out, prev_len, x, p=2):
    s = prev_out**p * prev_len
    s += x**p
    return (s / (prev_len + 1)) ** (1 / p), prev_len + 1


def online_power_mean(prev_out, prev_len, x, p=2):
    # s = prev_out**p * prev_len
    # s += x**p

    # return (s / (prev_len + 1)) ** (1 / p), prev_len + 1
    new_prev_len = prev_len + 1

    # Calculate the p-norm
    if prev_len > 0:
        new_prev_out = ((prev_out**p * prev_len + x**p) / new_prev_len) ** (1 / p)
    else:
        new_prev_out = x

    return new_prev_out, new_prev_len


from hypothesis import given, strategies as st
from scipy.stats import pmean
from functools import reduce


@given(st.lists(st.floats(min_value=2, max_value=10000), min_size=1))
def test_pmean(xs):
    assert math.isclose(pmean(xs, 2), power_mean(xs, 2))

    init_state = (0, 0)
    assert math.isclose(
        pmean(xs, 2),
        reduce(lambda acc, x: online_power_mean(*acc, x, 2), xs, init_state)[0],
    )


if __name__ == "__main__":
    test_pmean()
