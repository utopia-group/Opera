"""
func = "kstatvars_offline"
"""

import numpy as np


def kstat_1(xs):
    s1 = 0.0
    for x in xs:
        s1 += x

    return s1 / len(xs)


def kstat_2(xs):
    s1 = 0.0
    s2 = 0.0
    for x in xs:
        s1 += x
        s2 += x**2

    return (len(xs) * s2 - s1**2) / (len(xs) * (len(xs) - 1))


def kstat_3(xs):
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    for x in xs:
        s1 += x
        s2 += x**2
        s3 += x**3
    return (2 * s1**3 - 3 * len(xs) * s1 * s2 + len(xs) ** 2 * s3) / (
        len(xs) * (len(xs) - 1) * (len(xs) - 2)
    )


def kstat_4(xs):
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for x in xs:
        s1 += x
        s2 += x**2
        s3 += x**3
        s4 += x**4
    return (
        -6 * s1**4
        + 12 * len(xs) * s1**2 * s2
        - 3 * len(xs) * (len(xs) - 1) * s2**2
        - 4 * len(xs) * (len(xs) + 1) * s1 * s3
        + len(xs) ** 2 * (len(xs) + 1) * s4
    ) / (len(xs) * (len(xs) - 1) * (len(xs) - 2) * (len(xs) - 3))


def kstat_local(xs, n):
    match n:
        case 1:
            return kstat_1(xs)
        case 2:
            return kstat_2(xs)
        case 3:
            return kstat_3(xs)
        case 4:
            return kstat_4(xs)
        case _:
            raise ValueError("Only n=1,2,3,4 supported.")


def kstatvars_offline(xs):
    s1 = 0.0
    s2 = 0.0
    for x in xs:
        s1 += x
        s2 += x**2

    return ((len(xs) * s2 - s1**2) / (len(xs) * (len(xs) - 1))) / len(xs)

def kstatvars_online(prev_out, prev_s1, prev_s2, prev_len, x):
    l = prev_len + 1
    s1 = prev_s1 + x
    s2 = prev_s2 + x**2
    return ((l * s2 - s1**2) / (l * (l - 1))) / l, l, s1, s2


def kstatvar_2(xs):
    s1 = 0.0
    s2 = 0.0
    for x in xs:
        s1 += x
        s2 += x**2
    k2 = (len(xs) * s2 - s1**2) / (len(xs) * (len(xs) - 1))

    s3 = 0.0
    s4 = 0.0
    for x in xs:
        s3 += x**3
        s4 += x**4
    k4 = (
        -6 * s1**4
        + 12 * len(xs) * s1**2 * s2
        - 3 * len(xs) * (len(xs) - 1) * s2**2
        - 4 * len(xs) * (len(xs) + 1) * s1 * s3
        + len(xs) ** 2 * (len(xs) + 1) * s4
    ) / (len(xs) * (len(xs) - 1) * (len(xs) - 2) * (len(xs) - 3))
    return (2 * len(xs) * k2**2 + (len(xs) - 1) * k4) / (len(xs) * (len(xs) + 1))


def kstatvar_local(xs, n):
    match n:
        case 1:
            return kstatvar_1(xs)
        case 2:
            return kstatvar_2(xs)
        case _:
            raise ValueError("Only n=1,2 supported.")


from functools import reduce
from hypothesis import given, strategies as st
from scipy.stats import kstat, kstatvar
import numpy as np


@given(st.lists(st.floats(min_value=2, max_value=10000), min_size=5, unique=True))
def test_kstat(xs):
    for n in range(1, 5):
        assert np.isclose(kstat_local(xs, n), kstat(xs, n))

    for n in range(1, 3):
        assert np.isclose(kstatvar_local(xs, n), kstatvar(xs, n))


if __name__ == "__main__":
    test_kstat()
