"""
func = "variance_twopass"
"""


import math

"""
rlset reals;
off nat;

eqn1 := prev_out * prev_len = (xs0 - prev_avg)^2 + (xs1 - prev_avg)^2;
eqn2 := sq_s = (xs0 - avg)^2 + (xs1 - avg)^2 + (x__ - avg)^2;
avg_defn := avg * (prev_len+1) = (xs0 + xs1 + x__) and prev_avg * prev_len = (xs0 + xs1);
phi := ex({xs0, xs1}, eqn1 and eqn2 and avg_defn);

r := part(rlatl(rlqe phi), 4);
solve(r, sq_s);

3*avg**2 - 2*avg*prev_avg*prev_len - 2*avg*x__ + 2*prev_avg**2*prev_len - 2*prev_avg**2 + prev_len*prev_out + x__**2

f := (((prev_out = (0 + (xs0 - ((0 + xs0 + xs1) / 2)) ** 2 + (xs1 - ((0 + xs0 + xs1) / 2)) ** 2) / 2) and (prev_s = 0 + xs0 + xs1) and (prev_sq_s = 0 + (xs0 - ((0 + xs0 + xs1) / 2)) ** 2 + (xs1 - ((0 + xs0 + xs1) / 2)) ** 2) and (prev_avg = (0 + xs0 + xs1) / 2) and (prev_len = 2)) and (__unk_1_s = 0 + xs0 + xs1 + x__) and (__unk_2 = 3) and (sq_s = 0 + (xs0 - ((0 + xs0 + xs1 + x__) / 3)) ** 2 + (xs1 - ((0 + xs0 + xs1 + x__) / 3)) ** 2 + (x__ - ((0 + xs0 + xs1 + x__) / 3)) ** 2) and (unk_4__ = 3));
f := (((prev_out = (0 + (xs0 - ((0 + xs0 + xs1) / 2)) ** 2 + (xs1 - ((0 + xs0 + xs1) / 2)) ** 2) / 2) and (prev_s = 0 + xs0 + xs1) and (prev_sq_s = 0 + (xs0 - ((0 + xs0 + xs1) / 2)) ** 2 + (xs1 - ((0 + xs0 + xs1) / 2)) ** 2) and (prev_avg = (0 + xs0 + xs1) / 2) and (prev_len = 2)) and (__unk_2_s = 0 + xs0 + xs1 + x__) and (unk_2__ = 3) and (sq_s = 0 + (xs0 - ((0 + xs0 + xs1 + x__) / 3)) ** 2 + (xs1 - ((0 + xs0 + xs1 + x__) / 3)) ** 2 + (x__ - ((0 + xs0 + xs1 + x__) / 3)) ** 2) and (unk_4__ = 3));
phi := ex({xs0, xs1}, f);

sq_s = (prev_s**2 - 4*prev_s*x__ + 6*prev_sq_s + 4*x__**2)/6
"""

NEW_EQNS = r"""

(prev_s ** 2 - 4 * prev_s * x__ + 6 * prev_sq_s + 4 * x__ ** 2) / 6     // where prev_len = 2
(prev_s ** 2 - 6 * prev_s * x__ + 12 * prev_sq_s + 9 * x__ ** 2) / 12   // where prev_len = 3

new_sq_s = (prev_s ** 2 - f(prev_len) * prev_s * x__ + g(prev_len) * prev_sq_s + h(prev_len) * x__ ** 2) / k(prev_len)

adding test IOs:

10 = (11 ** 2 - f(4) * 11 * 4 + g(4) * 8.75 + h(4) * 4 ** 2) / k(4)
40 = (19 ** 2 - f(4) * 19 * 1 + g(4) * 28.75 + h(4) * 1 ** 2) / k(4)
35.2 = (22 ** 2 - f(4) * 22 * 1 + g(4) * 19 + h(4) * 1 ** 2) / k(4)
53.2 = (24 ** 2 - f(4) * 24 * 8 + g(4) * 50 + h(4) * 8 ** 2) / k(4)
21.388 = (11.6 ** 2 - f(4) * 11.6 * 6.5 + g(4) * 11.02 + h(4) * 6.5 ** 2) / k(4)


x = f(4)
y = g(4)
z = h(4)
w = k(4)

0 = (11 ** 2 - x * 11 * 4 + y * 8.75 + z * 4 ** 2) / w - 10
40 = (19 ** 2 - x * 19 * x__ + y * 28.75 +z * 1 ** 2) / w - 40
35.2 = (22 ** 2 - x * 22 * x__ + y * 19 +z * 1 ** 2) / w - 35.2
53.2 = (24 ** 2 - x * 24 * x__ + y * 50 +z * 8 ** 2) / w - 53.2
21.388 = (11.6 ** 2 - x * 11.6 * x__ + y * 11.02 +z * 6.5 ** 2) / w - 21.388



def solve_var():
    from sympy import solve
    from sympy.abc import x, y, z, w
    eqns = [
        (11 ** 2 - x * 11 * 4 + y * 8.75 + z * 4 ** 2) / w - 10,
        (19 ** 2 - x * 19 * 1 + y * 28.75 +z * 1 ** 2) / w - 40,
        (22 ** 2 - x * 22 * 1 + y * 19 +z * 1 ** 2) / w - 35.2,
        (24 ** 2 - x * 24 * 8 + y * 50 +z * 8 ** 2) / w - 53.2,
        (11.6 ** 2 - x * 11.6 * 6.5 + y * 11.02 +z * 6.5 ** 2) / w - 21.388
    ]
    result = solve(eqns, x, y, z, w, dict=True)
    print(result)

solve_var()
"""


def variance_onepass(xs):
    n = 0
    s = 0
    sq_s = 0
    for x in xs:
        n += 1
        s += x
        sq_s += x * x

    return sq_s / n - (s / n) ** 2


def variance_twopass(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)

    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs)

def variance_offline(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)
    sq_s = 0
    for x in xs:
        sq_s += (x - avg) ** 2
    return sq_s / len(xs)

def variance_opera(prev_out, prev_s, prev_sq_s, prev_len, x):
    l = prev_len + 1
    s = prev_s + x
    sq_s = (prev_len**2*x**2 - 2*prev_len*prev_s*x + prev_s**2 + prev_sq_s*(prev_len**2 + prev_len))/(prev_len**2 + prev_len)
    return sq_s / l, s, sq_s, l

def variance_online(prev_out, prev_s, prev_sq_s, prev_len, x):
    l = prev_len + 1
    s = prev_s + x
    avg = s / l
    sq_s = prev_sq_s + (x - avg) * (x - prev_s / prev_len)
    return sq_s / l, s, sq_s, l


r"""
var [] = 0
var xs =
    let s = foldl (+) 0 xs
        avg = s / (length xs)
        f acc x = acc + (x - avg)^2
    in foldl f 0 xs / (length xs)

variance xs =
    let s xs = foldl (+) 0 xs
        avg xs = s / (length xs)
        f xs acc x = acc + (x - (avg xs))^2
    in foldl (f xs) 0 xs / (length xs)

EApp div (EApp (EApp (EApp (EVar "f") EStream) 0) EStream) (EApp (EVar "length") xs)

f xs = 
    let loop_1 s x = s + x in
    let s = foldl loop_1 0 xs in
    let avg = s / (length xs) in
    let loop_2 sq_s x = sq_s + (x - avg)^2 in
    let sq_s = foldl loop_2 0 xs in
        sq_s / (length xs)

var_2 xs =
    let M = foldl (\M x -> M[s += x]) [][s = 0] xs in
    let M = M[avg = s / (length xs)] in
    let M = foldl (\M x -> M[sq_s = sq_s + (x - avg)^2]) M[sq_s = 0] xs in
        M[sq_s] / (length xs)


"""


def variance_twopass_template(xs):
    s = 0
    len_xs = len(xs)

    before_loop1 = dict(vars())
    for x in xs[:-1]:
        s += x
    after_loop1_unroll0 = dict(vars())
    s += xs[-1]
    after_loop1_unroll1 = dict(vars())
    avg = s / len_xs
    sq_s = 0

    before_loop2 = dict(vars())
    for x in xs[:-1]:
        sq_s += (x - avg) ** 2
    after_loop2_unroll0 = dict(vars())
    sq_s += (xs[-1] - avg) ** 2
    after_loop2_unroll1 = dict(vars())

    return (
        before_loop1,
        after_loop1_unroll0,
        after_loop1_unroll1,
        before_loop2,
        after_loop2_unroll0,
        after_loop2_unroll1,
        sq_s / len_xs,
    )


def variance_sketch_online(prev_out, prev_avg, prev_len, x):
    s = 0

    # add aux variables
    len_xs = prev_len + 1

    # first loop
    s = prev_avg * prev_len + x

    # post first loop
    avg = s / len_xs
    sq_s = 0

    # second loop
    sq_s = prev_out * prev_len + (x - prev_avg) * (x - avg)

    # post second loop
    return sq_s / len_xs, avg, len_xs, sq_s


def variance_welford(xs):
    mean = 0
    m2 = 0
    count = 0
    for n in xs:
        count += 1
        old_mean = mean
        mean += (n - mean) / count
        m2 += (n - old_mean) * (n - mean)
    return m2 / count


import math
from functools import reduce

from hypothesis import given
from hypothesis import strategies as st
from numpy import abs, var


@given(st.lists(st.floats(min_value=-10e5, max_value=10e5), min_size=2))
def test_var(xs):
    expected_var = var(xs)
    # assert math.isclose(variance_onepass(xs), expected_var)
    assert math.isclose(variance_twopass(xs), expected_var)
    assert math.isclose(variance_welford(xs), expected_var)
    init_state = (var(xs[:1]), xs[0], 1)
    # assert math.isclose(
    #     reduce(lambda acc, x: variance_onepass_online(*acc, x), xs[1:], init_state)[0],
    #     expected_var,
    # )

    assert math.isclose(
        abs(
            reduce(lambda acc, x: variance_twopass_online(*acc, x), xs[1:], init_state)[
                0
            ]
        ),
        expected_var,
    ), f"{reduce(lambda acc, x: variance_twopass_online(*acc, x), xs[1:], init_state)[0]}, {expected_var}"


if __name__ == "__main__":
    test_var()
    # from pprint import pprint

    # xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # pprint(variance_twopass_sketch(xs))

    # xs = [11, 12, 13, 14, 15]
    # pprint(variance_twopass_sketch(xs))
