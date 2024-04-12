"""
func = "mean"
"""


def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)

"""
mean xs = foldl (+) 0 xs / length xs
length xs = foldl (\acc _ -> acc + 1) 0 xs

def mean_1(xs):
    s = 0
    n = 0
    for i in xs:
        s += i
        n += 1
    return s / n

mean_1 xs =
    let loop_1 (s, n) i = (s + i, n + 1) in
    let (s, n) = foldl loop_1 (0, 0) xs in
    s / n

mean_2 xs =
    let loop_1 M i = M[s = s + i; n = n + 1] in
    let M = foldl loop_1 M[s = 0; n = 0] xs in
    M[s] / M[n]
"""

r"""
\xs -> let S = {} in let S = foldl(xs, S{s = 0}, \S n -> S{s = (s + n)}) in (S[s] / len(xs))

xs = [x0, x1, x2, x3]
S = {s = 0}
prev_s = 0 + x0 + x1 + x2 + x3
prev_out = prev_s / 4
"""


import math


def mean_offline(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)

def mean_online(prev_out, prev_len, x):
    l = prev_len + 1
    r = prev_out + (x - prev_out) / l
    return r, l

def mean_opera(prev_out, prev_s, prev_len, x):
    s = prev_s + x
    l = prev_len + 1
    return s / l, s, l


import math
from functools import reduce

from hypothesis import given
from hypothesis import strategies as st
from numpy import mean as np_mean


@given(st.lists(st.integers(), min_size=2))
def test_sample_mean(xs):
    expected_sample_mean = np_mean(xs)
    assert math.isclose(
        reduce(lambda acc, x: mean_online_smart(*acc, x), xs, (0, 0))[0],
        expected_sample_mean,
    )


if __name__ == "__main__":
    test_sample_mean()
