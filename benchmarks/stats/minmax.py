"""
func = "minmax"

"""


from sys import maxsize

"""
rlset reals;
off nat;

p := ((xs0 < mn) => (mn = xs0)) and ((xs1 < mn) => (mn = xs1)) and ((x < mn) => (mn = x));
v := ((xs0 < prev_mn) => (prev_mn = xs0)) and ((xs1 < prev_mn) => (prev_mn = xs1));
phi := ex({xs0, xs1}, p and v);

f := part(rlatl(rlqe phi), 1);
solve(f, r);
"""

"""

min xs = foldl (\acc x -> if x < acc then x else acc) maxsize xs

min [x0, x1] =
    (x0 < maxsize) => ((x1 < x0) => prev_out = x1 and (x1 >= x0) => prev_out = x0) and
    (x0 >= maxsize) => ((x1 < maxsize) => prev_out = x1 and (x1 >= maxsize) => prev_out = maxsize)

min [x0, x1, x2] =
    (x0 < maxsize) => ((x1 < x0) => ((x2 < x1) => prev_out = x2 and (x2 >= x1) => prev_out = x1) and (x1 >= x0) => ((x2 < x0) => prev_out = x2 and (x2 >= x0) => prev_out = x0)) and
    (x0 >= maxsize) => ((x1 < maxsize) => ((x2 < x1) => prev_out = x2 and (x2 >= x1) => prev_out = x1) and (x1 >= maxsize) => ((x2 < maxsize) => prev_out = x2 and (x2 >= maxsize) => prev_out = maxsize))
"""

def min_(xs):
    mn = 10000
    for x in xs:
        if x < mn:
            mn = x
    return mn


def minmax(xs):
    mn = 1000
    mx = 0
    for n in xs:
        if n < mn:
            mn = n
        if n > mx:
            mx = n
    return mx - mn

def minmax_online(prev_out, prev_mn, prev_mx, x):
    if x < prev_mn:
        prev_mn = x
    if x > prev_mx:
        prev_mx = x
    return prev_mx - prev_mn, prev_mn, prev_mx




r"""
minmax xs =
    let S = foldl (\S x -> S[mn = if n < mn then n else mn; l = l + 1]) {mn=1000; l=0} xs in (S[l], S[mn])




"""


from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.integers(min_value=-maxsize, max_value=maxsize), min_size=1))
def test_minmax(xs):
    assert minmax(xs) == (min(xs), max(xs))


if __name__ == "__main__":
    test_minmax()
