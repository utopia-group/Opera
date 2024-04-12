"""
func = "kurtosis"
"""

import numpy as np

"""


((36 * (prev_m2) ** 2 + 12 * prev_m2 * (prev_s) ** 2 - 48 * prev_m2 * prev_s * x__ + 48 * prev_m2 * (x__) ** 2 + (prev_s) ** 4 - 8 * (prev_s) ** 3 * x__ + 24 * (prev_s) ** 2 * (x__) ** 2 - 32 * prev_s * (x__) ** 3 + 16 * (x__) ** 4) / 72)


(1728 * new_m4 - 7 * (prev_s) ** 4 + 84 * (prev_s) ** 3 * x__ - 378 * (prev_s) ** 2 * (x__) ** 2 + 756 * prev_s * (x__) ** 3 - 567 * (x__) ** 4 = 0)
(576 * new_m4 + 19 * (prev_s) ** 4 - 36 * (prev_s) ** 3 * x__ - 126 * (prev_s) ** 2 * (x__) ** 2 + 252 * prev_s * (x__) ** 3 - 189 * (x__) ** 4 = 0)
(192 * new_m3 * prev_s - 576 * new_m3 * x__ - 576 * new_m4 + 5 * (prev_s) ** 4 - 60 * (prev_s) ** 3 * x__ + 270 * (prev_s) ** 2 * (x__) ** 2 - 540 * prev_s * (x__) ** 3 + 405 * (x__) ** 4 = 0)
(2985984 * (new_m4) ** 2 - 2985984 * new_m4 * (prev_m2) ** 2 - 248832 * new_m4 * prev_m2 * (prev_s) ** 2 + 1492992 * new_m4 * prev_m2 * prev_s * x__ - 2239488 * new_m4 * prev_m2 * (x__) ** 2 - 24192 * new_m4 * (prev_s) ** 4 + 290304 * new_m4 * (prev_s) ** 3 * x__ - 1306368 * new_m4 * (prev_s) ** 2 * (x__) ** 2 + 2612736 * new_m4 * prev_s * (x__) ** 3 - 1959552 * new_m4 * (x__) ** 4 + 746496 * (prev_m2) ** 4 + 69120 * (prev_m2) ** 3 * (prev_s) ** 2 - 414720 * (prev_m2) ** 3 * prev_s * x__ + 622080 * (prev_m2) ** 3 * (x__) ** 2 + 17280 * (prev_m2) ** 2 * (prev_s) ** 4 - 207360 * (prev_m2) ** 2 * (prev_s) ** 3 * x__ + 933120 * (prev_m2) ** 2 * (prev_s) ** 2 * (x__) ** 2 - 1866240 * (prev_m2) ** 2 * prev_s * (x__) ** 3 + 1399680 * (prev_m2) ** 2 * (x__) ** 4 + 1008 * prev_m2 * (prev_s) ** 6 - 18144 * prev_m2 * (prev_s) ** 5 * x__ + 136080 * prev_m2 * (prev_s) ** 4 * (x__) ** 2 - 544320 * prev_m2 * (prev_s) ** 3 * (x__) ** 3 + 1224720 * prev_m2 * (prev_s) ** 2 * (x__) ** 4 - 1469664 * prev_m2 * prev_s * (x__) ** 5 + 734832 * prev_m2 * (x__) ** 6 + 49 * (prev_s) ** 8 - 1176 * (prev_s) ** 7 * x__ + 12348 * (prev_s) ** 6 * (x__) ** 2 - 74088 * (prev_s) ** 5 * (x__) ** 3 + 277830 * (prev_s) ** 4 * (x__) ** 4 - 666792 * (prev_s) ** 3 * (x__) ** 5 + 1000188 * (prev_s) ** 2 * (x__) ** 6 - 857304 * prev_s * (x__) ** 7 + 321489 * (x__) ** 8 = 0)
(1728 * new_m4 - 864 * (prev_m2) ** 2 - 72 * prev_m2 * (prev_s) ** 2 + 432 * prev_m2 * prev_s * x__ - 648 * prev_m2 * (x__) ** 2 - 576 * prev_m3 * prev_s + 1728 * prev_m3 * x__ - 7 * (prev_s) ** 4 + 84 * (prev_s) ** 3 * x__ - 378 * (prev_s) ** 2 * (x__) ** 2 + 756 * prev_s * (x__) ** 3 - 567 * (x__) ** 4 = 0)
(192 * new_m3 * prev_s - 576 * new_m3 * x__ - 576 * new_m4 + 288 * (prev_m2) ** 2 - 24 * prev_m2 * (prev_s) ** 2 + 144 * prev_m2 * prev_s * x__ - 216 * prev_m2 * (x__) ** 2 + 5 * (prev_s) ** 4 - 60 * (prev_s) ** 3 * x__ + 270 * (prev_s) ** 2 * (x__) ** 2 - 540 * prev_s * (x__) ** 3 + 405 * (x__) ** 4 = 0)
(576 * new_m4 - 288 * (prev_m2) ** 2 - 120 * prev_m2 * (prev_s) ** 2 + 432 * prev_m2 * prev_s * x__ - 216 * prev_m2 * (x__) ** 2 + 19 * (prev_s) ** 4 - 36 * (prev_s) ** 3 * x__ - 126 * (prev_s) ** 2 * (x__) ** 2 + 252 * prev_s * (x__) ** 3 - 189 * (x__) ** 4 = 0)

# most likely one:

(1728 * new_m4 - 864 * (prev_m2) ** 2 - 72 * prev_m2 * (prev_s) ** 2 + 432 * prev_m2 * prev_s * x__ - 648 * prev_m2 * (x__) ** 2 - 576 * prev_m3 * prev_s + 1728 * prev_m3 * x__ - 7 * (prev_s) ** 4 + 84 * (prev_s) ** 3 * x__ - 378 * (prev_s) ** 2 * (x__) ** 2 + 756 * prev_s * (x__) ** 3 - 567 * (x__) ** 4 = 0)

solved:
new_m4 = (864*prev_m2**2 + 72*prev_m2*prev_s**2 - 432*prev_m2*prev_s*x__ + 648*prev_m2*x__**2 + 576*prev_m3*prev_s - 1728*prev_m3*x__ + 7*prev_s**4 - 84*prev_s**3*x__ + 378*prev_s**2*x__**2 - 756*prev_s*x__**3 + 567*x__**4)/1728
"""


def kurtosis(xs):
    s = 0
    for x in xs:
        s += x
    avg = s / len(xs)

    m2 = 0
    for x in xs:
        m2 += (x - avg) ** 2
    sigma = (m2 / len(xs)) ** 0.5

    m4 = 0
    for x in xs:
        m4 += (x - avg) ** 4
    return (m4 / len(xs)) / (sigma**4) - 3


def kurtosis_online(v, m4, m3, m2, s, n, x):
    new_s = s + x
    new_n = n + 1
    delta = x - (s / n)
    delta_n = delta / n
    term1 = delta * delta_n * (n - 1)
    new_m4 = m4 + (
        term1 * (delta_n**2) * (n**2 - 3 * n + 3)
        + 6 * delta_n**2 * m2
        - 4 * delta_n * m3
    )
    new_m3 = m3 + term1 * delta_n * (n - 2) - 3 * delta_n * m2
    new_m2 = m2 + term1
    sigma = (m2 / n) ** 0.5
    return (new_m4 / n) / (sigma**4) - 3, new_m4, new_m3, new_m2, new_s, new_n


def kurtosis_alternative(xs):
    n = len(xs)

    s = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for x in xs:
        s += x
        s2 += x**2
        s3 += x**3
        s4 += x**4
    mean = s / n
    mean2 = s2 / n
    mean3 = s3 / n
    mean4 = s4 / n

    sq_s = 0
    for x in xs:
        sq_s += (x - mean) ** 2
    sigma = (sq_s / n) ** 0.5

    return (
        mean4 - 4 * mean3 * mean + 6 * mean2 * mean**2 - 3 * mean**4
    ) / sigma**4 - 3


def kurtosis_terriberry(xs):
    mean = 0
    n = 0
    m2 = 0
    m3 = 0
    m4 = 0

    for x in xs:
        prev_mean = mean
        prev_m2 = m2
        prev_m3 = m3
        prev_m4 = m4
        prev_len = n

        n += 1
        delta = x - mean
        mean += delta / n
        prev_m2 = m2

        m2 += (delta**2) * (n - 1) / n
        m3 += (delta**3) * (n - 1) * (n - 2) / (n**2) - 3 * delta * prev_m2 / n
        # m4 += (
        #     (delta**4) * (n - 1) * (n**2 - 3 * n + 3) / (n**3)
        #     + 6 * (delta**2) * prev_m2 / (n**2)
        #     - 4 * delta * prev_m3 / n
        # )

        m4 += (
            ((x - prev_mean) ** 4)
            * prev_len
            * ((prev_len + 1) ** 2 - 3 * (prev_len + 1) + 3)
            / ((prev_len + 1) ** 3)
            + 6 * ((x - prev_mean) ** 2) * prev_m2 / ((prev_len + 1) ** 2)
            - 4 * (x - prev_mean) * prev_m3 / (prev_len + 1)
        )
    return (n * m4) / (m2**2) - 3


def kurtosis_welford_onepass(xs):
    n = 0
    mean = 0
    m2 = 0
    m3 = 0
    m4 = 0
    for x in xs:
        n += 1
        delta = x - mean
        delta_n = delta / n
        term1 = delta * delta_n * (n - 1)
        mean += delta_n
        m4 += (
            term1 * (delta_n**2) * (n**2 - 3 * n + 3)
            + 6 * delta_n**2 * m2
            - 4 * delta_n * m3
        )
        m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2
        m2 += term1
    sigma = (m2 / n) ** 0.5
    return (m4 / n) / (sigma**4) - 3


from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import kurtosis as ks


@given(
    st.lists(
        st.floats(min_value=1, max_value=9.223372036854776e18), min_size=5, unique=True
    )
)
def test_moments(xs):
    assert np.isclose(kurtosis_definition(xs), ks(xs))
    assert np.isclose(kurtosis_alternative(xs), ks(xs))
    assert np.isclose(kurtosis_welford_onepass(xs), ks(xs))
    assert np.isclose(kurtosis_terriberry(xs), ks(xs))


if __name__ == "__main__":
    test_moments()

RLQE_INPUT = r"""
phi_temp := ex({x0, x1, x2, x3}, ((prev_s = x0 + x1 + x2 + x3) and (prev_m2 = ((3 / 4) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x2 + (((-1) / 4)) * x3) ** 2 + ((3 / 4) * x1 + (((-1) / 4)) * x0 + (((-1) / 4)) * x2 + (((-1) / 4)) * x3) ** 2 + ((3 / 4) * x2 + (((-1) / 4)) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x3) ** 2 + ((3 / 4) * x3 + (((-1) / 4)) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x2) ** 2) and (4 = prev_len) and (prev_m3 = ((3 / 4) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x2 + (((-1) / 4)) * x3) ** 3 + ((3 / 4) * x1 + (((-1) / 4)) * x0 + (((-1) / 4)) * x2 + (((-1) / 4)) * x3) ** 3 + ((3 / 4) * x2 + (((-1) / 4)) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x3) ** 3 + ((3 / 4) * x3 + (((-1) / 4)) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x2) ** 3) and (prev_m4 = ((3 / 4) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x2 + (((-1) / 4)) * x3) ** 4 + ((3 / 4) * x1 + (((-1) / 4)) * x0 + (((-1) / 4)) * x2 + (((-1) / 4)) * x3) ** 4 + ((3 / 4) * x2 + (((-1) / 4)) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x3) ** 4 + ((3 / 4) * x3 + (((-1) / 4)) * x0 + (((-1) / 4)) * x1 + (((-1) / 4)) * x2) ** 4) and (5 = new_len) and (new_m2 = ((4 / 5) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 2 + ((4 / 5) * x1 + (((-1) / 5)) * x0 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 2 + ((4 / 5) * x2 + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 2 + ((4 / 5) * x3 + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x__) ** 2 + ((4 / 5) * x__ + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3) ** 2) and (new_m3 = ((4 / 5) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 3 + ((4 / 5) * x1 + (((-1) / 5)) * x0 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 3 + ((4 / 5) * x2 + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 3 + ((4 / 5) * x3 + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x__) ** 3 + ((4 / 5) * x__ + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3) ** 3) and (new_m4 = ((4 / 5) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 4 + ((4 / 5) * x1 + (((-1) / 5)) * x0 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 4 + ((4 / 5) * x2 + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x3 + (((-1) / 5)) * x__) ** 4 + ((4 / 5) * x3 + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x__) ** 4 + ((4 / 5) * x__ + (((-1) / 5)) * x0 + (((-1) / 5)) * x1 + (((-1) / 5)) * x2 + (((-1) / 5)) * x3) ** 4) and (new_s = x0 + x1 + x2 + x3 + x__)));

"""

RLQE_RAW_OUTPUT = r"""
{prev_out = 0,
prev_m4 = 0,
prev_m3 = 0,
prev_m2 = 0,
prev_m2 > 0,
prev_m2**2 - 2*prev_out = 0,
prev_m2**2 - 2*prev_m4 = 0,
prev_m2**3 - 6*prev_m3**2 = 0,
prev_m2**3 - 6*prev_m3**2 > 0,
prev_len - 3 = 0,
new_s - prev_s - x__ = 0,
576*new_m4 + 19*prev_s**4 - 36*prev_s**3*x__ - 126*prev_s**2*x__**2 + 252*prev_s
*x__**3 - 189*x__**4 = 0,
576*new_m4 + 19*prev_s**4 - 36*prev_s**3*x__ - 126*prev_s**2*x__**2 + 252*prev_s
*x__**3 - 189*x__**4 <> 0,
576*new_m4 - 288*prev_m2**2 - 120*prev_m2*prev_s**2 + 432*prev_m2*prev_s*x__ - 
216*prev_m2*x__**2 + 19*prev_s**4 - 36*prev_s**3*x__ - 126*prev_s**2*x__**2 + 
252*prev_s*x__**3 - 189*x__**4 = 0,
576*new_m4 - 288*prev_m2**2 - 120*prev_m2*prev_s**2 + 432*prev_m2*prev_s*x__ - 
216*prev_m2*x__**2 + 19*prev_s**4 - 36*prev_s**3*x__ - 126*prev_s**2*x__**2 + 
252*prev_s*x__**3 - 189*x__**4 <> 0,
1728*new_m4 - 7*prev_s**4 + 84*prev_s**3*x__ - 378*prev_s**2*x__**2 + 756*prev_s
*x__**3 - 567*x__**4 = 0,
1728*new_m4 - 864*prev_m2**2 - 72*prev_m2*prev_s**2 + 432*prev_m2*prev_s*x__ - 
648*prev_m2*x__**2 - 576*prev_m3*prev_s + 1728*prev_m3*x__ - 7*prev_s**4 + 84*
prev_s**3*x__ - 378*prev_s**2*x__**2 + 756*prev_s*x__**3 - 567*x__**4 = 0,
2985984*new_m4**2 - 2985984*new_m4*prev_m2**2 - 248832*new_m4*prev_m2*prev_s**2 
+ 1492992*new_m4*prev_m2*prev_s*x__ - 2239488*new_m4*prev_m2*x__**2 - 24192*
new_m4*prev_s**4 + 290304*new_m4*prev_s**3*x__ - 1306368*new_m4*prev_s**2*x__**2
 + 2612736*new_m4*prev_s*x__**3 - 1959552*new_m4*x__**4 + 746496*prev_m2**4 + 
69120*prev_m2**3*prev_s**2 - 414720*prev_m2**3*prev_s*x__ + 622080*prev_m2**3*
x__**2 + 17280*prev_m2**2*prev_s**4 - 207360*prev_m2**2*prev_s**3*x__ + 933120*
prev_m2**2*prev_s**2*x__**2 - 1866240*prev_m2**2*prev_s*x__**3 + 1399680*prev_m2
**2*x__**4 + 1008*prev_m2*prev_s**6 - 18144*prev_m2*prev_s**5*x__ + 136080*
prev_m2*prev_s**4*x__**2 - 544320*prev_m2*prev_s**3*x__**3 + 1224720*prev_m2*
prev_s**2*x__**4 - 1469664*prev_m2*prev_s*x__**5 + 734832*prev_m2*x__**6 + 49*
prev_s**8 - 1176*prev_s**7*x__ + 12348*prev_s**6*x__**2 - 74088*prev_s**5*x__**3
 + 277830*prev_s**4*x__**4 - 666792*prev_s**3*x__**5 + 1000188*prev_s**2*x__**6 
- 857304*prev_s*x__**7 + 321489*x__**8 = 0,
2985984*new_m4**2 - 2985984*new_m4*prev_m2**2 - 248832*new_m4*prev_m2*prev_s**2 
+ 1492992*new_m4*prev_m2*prev_s*x__ - 2239488*new_m4*prev_m2*x__**2 - 24192*
new_m4*prev_s**4 + 290304*new_m4*prev_s**3*x__ - 1306368*new_m4*prev_s**2*x__**2
 + 2612736*new_m4*prev_s*x__**3 - 1959552*new_m4*x__**4 + 746496*prev_m2**4 + 
69120*prev_m2**3*prev_s**2 - 414720*prev_m2**3*prev_s*x__ + 622080*prev_m2**3*
x__**2 + 17280*prev_m2**2*prev_s**4 - 207360*prev_m2**2*prev_s**3*x__ + 933120*
prev_m2**2*prev_s**2*x__**2 - 1866240*prev_m2**2*prev_s*x__**3 + 1399680*prev_m2
**2*x__**4 + 1008*prev_m2*prev_s**6 - 18144*prev_m2*prev_s**5*x__ + 136080*
prev_m2*prev_s**4*x__**2 - 544320*prev_m2*prev_s**3*x__**3 + 1224720*prev_m2*
prev_s**2*x__**4 - 1469664*prev_m2*prev_s*x__**5 + 734832*prev_m2*x__**6 + 49*
prev_s**8 - 1176*prev_s**7*x__ + 12348*prev_s**6*x__**2 - 74088*prev_s**5*x__**3
 + 277830*prev_s**4*x__**4 - 666792*prev_s**3*x__**5 + 1000188*prev_s**2*x__**6 
- 857304*prev_s*x__**7 + 321489*x__**8 < 0,
8*new_m3 + prev_s**3 - prev_s**2*x__ + 3*prev_s*x__**2 - 3*x__**3 <> 0,
8*new_m3 - 6*prev_m2*prev_s + 6*prev_m2*x__ + prev_s**3 - prev_s**2*x__ + 3*
prev_s*x__**2 - 3*x__**3 = 0,
8*new_m3 - 6*prev_m2*prev_s + 6*prev_m2*x__ + prev_s**3 - prev_s**2*x__ + 3*
prev_s*x__**2 - 3*x__**3 <> 0,
72*new_m3 + prev_s**3 - 9*prev_s**2*x__ + 27*prev_s*x__**2 - 27*x__**3 = 0,
72*new_m3 - 18*prev_m2*prev_s + 54*prev_m2*x__ - 72*prev_m3 + prev_s**3 - 9*
prev_s**2*x__ + 27*prev_s*x__**2 - 27*x__**3 = 0,
192*new_m3*prev_s - 576*new_m3*x__ - 576*new_m4 + 5*prev_s**4 - 60*prev_s**3*x__
 + 270*prev_s**2*x__**2 - 540*prev_s*x__**3 + 405*x__**4 = 0,
192*new_m3*prev_s - 576*new_m3*x__ - 576*new_m4 + 288*prev_m2**2 - 24*prev_m2*
prev_s**2 + 144*prev_m2*prev_s*x__ - 216*prev_m2*x__**2 + 5*prev_s**4 - 60*
prev_s**3*x__ + 270*prev_s**2*x__**2 - 540*prev_s*x__**3 + 405*x__**4 = 0,
5184*new_m3**2 - 2592*new_m3*prev_m2*prev_s + 7776*new_m3*prev_m2*x__ + 144*
new_m3*prev_s**3 - 1296*new_m3*prev_s**2*x__ + 3888*new_m3*prev_s*x__**2 - 3888*
new_m3*x__**3 - 864*prev_m2**3 + 324*prev_m2**2*prev_s**2 - 1944*prev_m2**2*
prev_s*x__ + 2916*prev_m2**2*x__**2 - 36*prev_m2*prev_s**4 + 432*prev_m2*prev_s
**3*x__ - 1944*prev_m2*prev_s**2*x__**2 + 3888*prev_m2*prev_s*x__**3 - 2916*
prev_m2*x__**4 + prev_s**6 - 18*prev_s**5*x__ + 135*prev_s**4*x__**2 - 540*
prev_s**3*x__**3 + 1215*prev_s**2*x__**4 - 1458*prev_s*x__**5 + 729*x__**6 = 0,
5184*new_m3**2 - 2592*new_m3*prev_m2*prev_s + 7776*new_m3*prev_m2*x__ + 144*
new_m3*prev_s**3 - 1296*new_m3*prev_s**2*x__ + 3888*new_m3*prev_s*x__**2 - 3888*
new_m3*x__**3 - 864*prev_m2**3 + 324*prev_m2**2*prev_s**2 - 1944*prev_m2**2*
prev_s*x__ + 2916*prev_m2**2*x__**2 - 36*prev_m2*prev_s**4 + 432*prev_m2*prev_s
**3*x__ - 1944*prev_m2*prev_s**2*x__**2 + 3888*prev_m2*prev_s*x__**3 - 2916*
prev_m2*x__**4 + prev_s**6 - 18*prev_s**5*x__ + 135*prev_s**4*x__**2 - 540*
prev_s**3*x__**3 + 1215*prev_s**2*x__**4 - 1458*prev_s*x__**5 + 729*x__**6 < 0,
12*new_m2 - prev_s**2 + 6*prev_s*x__ - 9*x__**2 = 0,
12*new_m2 - 12*prev_m2 - prev_s**2 + 6*prev_s*x__ - 9*x__**2 = 0,
new_len - 4 = 0}$
"""