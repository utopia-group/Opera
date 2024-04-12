"""
func = "mode_offline"
"""


def mode_offline(xs):
    counts = {}
    for x in xs:
        counts = {**counts, x: counts.get(x, 0) + 1}

    k = 0
    v = 0
    for xc in list(counts.items()):
        if xc[1] > v:
            k = xc[0]
            v = xc[1]
    return k


def mode_online(prev_out, counts, x):
    counts[x] = counts.get(x, 0) + 1

    k = 0
    v = 0
    for xc in list(counts.items()):
        if xc[1] > v:
            k = xc[0]
            v = xc[1]
    return k


from hypothesis import given, strategies as st
from statistics import mode as mode_ref
from functools import reduce


@given(st.lists(st.floats(), min_size=2))
def test_mode(xs):
    assert mode(xs) == mode_ref(xs)
    assert reduce(lambda acc, x: mode_online(*acc, x), xs[1:], (xs[0], {}))[
        0
    ] == mode_ref(xs)


if __name__ == "__main__":
    test_mode()
