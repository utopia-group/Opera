"""
func = "sem"
find_good_input = false
"""


def sem(xs):
    s = 0
    for x in xs:
        s += x
    mean = s / len(xs)

    m2 = 0
    for x in xs:
        m2 += (x - mean) ** 2
    variance = m2 / (len(xs) - 1)

    std = variance**0.5

    return std / len(xs) ** 0.5

def sem_online(prev_out, prev_s, prev_m2, prev_len, x):
    l = prev_len + 1
    s = prev_s + x
    m2 = prev_m2 + (x - prev_s / prev_len) * (x - s / l)
    variance = m2 / (l - 1)
    std = variance**0.5
    return std / l ** 0.5, s, m2, l


def sem_welford_onepass(xs):
    mean = 0
    m2 = 0
    count = 0
    for x in xs:
        count += 1
        old_mean = mean
        mean += (x - mean) / count
        m2 += (x - old_mean) * (x - mean)
    variance = m2 / (count - 1)
    std = variance**0.5
    return std / count**0.5


from hypothesis import given
from hypothesis import strategies as st
from numpy import isclose
from scipy import stats


@given(st.lists(st.floats(min_value=1, max_value=1e16), min_size=2, unique=True))
def test_sem(xs):
    expected = stats.sem(xs)
    assert isclose(sem(xs), expected)
    assert isclose(sem_welford_onepass(xs), expected)


if __name__ == "__main__":
    test_sem()
