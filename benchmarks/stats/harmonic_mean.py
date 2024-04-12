"""
func = "harmonic_mean_offline"
find_good_input = false
"""

"""
rlset reals;
off nat;

p := s * (xs1 * xs2 * x) = (xs1 * xs2 + xs1 * x + xs2 * x);
v := prev_s * (xs1 * xs2) = (xs1 + xs2);
phi := ex({xs1, xs2}, p and v);

p := s = 1/xs1 + 1/xs2 + 1/x;
v := prev_s = 1/xs1 + 1/xs2;
phi := ex({xs1, xs2}, p and v);

f := part(rlatl(rlqe phi), 1);
solve(f, r);
"""

def harmonic_mean_offline(xs):
    s = 0
    for x in xs:
        s += 1 / x
    return len(xs) / s


def harmonic_mean_online(prev_out, prev_len, x):
    prev_s = prev_len / prev_out
    l = prev_len + 1
    s = prev_s + 1 / x
    return (l * prev_out * x) / (prev_len * x + prev_out), l


import math
import statistics
from functools import reduce

from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.floats(min_value=1, max_value=10000), min_size=1))
def test_harmonic_mean(xs):
    expected_hm = statistics.harmonic_mean(xs)
    assert math.isclose(harmonic_mean(xs), expected_hm)
    assert math.isclose(
        reduce(lambda acc, x: harmonic_mean_online(*acc, x), xs[1:], (xs[0], 1))[0],
        expected_hm,
    )


if __name__ == "__main__":
    test_harmonic_mean()
