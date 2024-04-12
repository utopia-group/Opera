"""
func = "nobs"
"""


def nobs(xs):
    n = 0
    for _ in xs:
        n += 1
    return n

def nobs_online(prev_out):
    return prev_out + 1


from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.floats()))
def test_nobs(xs):
    assert nobs(xs) == len(xs)


if __name__ == "__main__":
    test_nobs()
