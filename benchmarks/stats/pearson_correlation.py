"""
func = "pearson_correlation_"
stream_type = "(float, float)"
find_good_input = false
unroll_depth = 4
"""

def pearson_correlation_(xs):
    s_x = 0
    s_y = 0
    for p in xs:
        s_x += p[0]
        s_y += p[1]
    
    mean_x = s_x / len(xs)
    mean_y = s_y / len(xs)

    sq_s_x = 0
    sq_s_y = 0
    sq_r = 0
    for p in xs:
        sq_s_x += (p[0] - mean_x) ** 2
        sq_s_y += (p[1] - mean_y) ** 2
        sq_r += (p[0] - mean_x) * (p[1] - mean_y)
    sigma_x = (sq_s_x / len(xs)) ** 0.5
    sigma_y = (sq_s_y / len(xs)) ** 0.5
    return sq_r / (len(xs) * sigma_x * sigma_y)

def pearson_correlation(xs, ys):
    s_x = 0
    s_y = 0
    for x, y in zip(xs, ys):
        s_x += x
        s_y += y
    
    mean_x = s_x / len(xs)
    mean_y = s_y / len(xs)

    sq_s_x = 0
    sq_s_y = 0
    sq_r = 0
    for x, y in zip(xs, ys):
        sq_s_x += (x - mean_x) ** 2
        sq_s_y += (y - mean_y) ** 2
        sq_r += (x - mean_x) * (y - mean_y)
    sigma_x = (sq_s_x / len(xs)) ** 0.5
    sigma_y = (sq_s_y / len(xs)) ** 0.5
    return sq_r / (len(xs) * sigma_x * sigma_y)


def pearson_correlation_online(prev_out, prev_s_x, prev_s_y, prev_sq_s_x, prev_sq_s_y, prev_sq_r, prev_len, x, y):
    s_x = prev_s_x + x
    s_y = prev_s_y + y
    mean_x = s_x / (prev_len + 1)
    mean_y = s_y / (prev_len + 1)

    sq_s_x = prev_sq_s_x + (x - mean_x) * (x - prev_s_x / prev_len)
    sq_s_y = prev_sq_s_y + (y - mean_y) * (y - prev_s_y / prev_len)
    sq_r = prev_sq_r + (x - prev_s_x / prev_len) * (y - mean_y)
    sigma_x = (sq_s_x / (prev_len + 1)) ** 0.5
    sigma_y = (sq_s_y / (prev_len + 1)) ** 0.5
    return sq_r / ((prev_len + 1) * sigma_x * sigma_y)

def pearson_correlation_opera(prev_out, prev_s_x, prev_s_y, prev_sq_s_x, prev_sq_s_y, prev_sq_r, prev_len, x, y):
    s_x = prev_s_x + x
    mean_x = s_x / (prev_len + 1)

    sq_s_x = prev_sq_s_x + (x - mean_x) * (x - prev_s_x / prev_len)
    sq_s_x = (prev_len**2*x**2 - 2*prev_len*prev_s_x*x + prev_s_x**2 + prev_sq_s_x*(prev_len**2 + prev_len))/(prev_len**2 + prev_len)
    sq_s_y = ((prev_len**2*y**2 - 2*prev_len*prev_s_y*y + prev_s_y**2 + prev_sq_s_y*(prev_len**2 + prev_len))/(prev_len**2 + prev_len))
    sq_r = (prev_len**2*x*y - prev_len*prev_s_x*y - prev_len*prev_s_y*x + prev_s_x*prev_s_y + prev_sq_r*(prev_len**2 + prev_len))/(prev_len**2 + prev_len)
    sigma_x = (sq_s_x / (prev_len + 1)) ** 0.5
    sigma_y = (sq_s_y / (prev_len + 1)) ** 0.5
    return sq_r / ((prev_len + 1) * sigma_x * sigma_y)


def pearson_correlation2(xs, ys):
    n = len(xs)

    s_x = 0
    s_y = 0
    for xi, yi in zip(xs, ys):
        s_x += xi
        s_y += yi
    mean_x = s_x / n
    mean_y = s_y / n

    sq_s_x = 0
    sq_s_y = 0
    sq_r = 0
    for xi, yi in zip(xs, ys):
        sq_s_x += (xi - mean_x) ** 2
        sq_s_y += (yi - mean_y) ** 2
        sq_r += (xi - mean_x) * (yi - mean_y)
    sigma_x = (sq_s_x / n) ** 0.5
    sigma_y = (sq_s_y / n) ** 0.5
    return sq_r / (n * sigma_x * sigma_y)


def pearson_correlation_welford_onepass(xs, ys):
    mean_x = 0
    mean_y = 0
    m2_x = 0
    m2_y = 0
    m2_xy = 0
    count = 0
    for x, y in zip(xs, ys):
        count += 1
        old_mean_x = mean_x
        old_mean_y = mean_y
        mean_x += (x - mean_x) / count
        mean_y += (y - mean_y) / count
        m2_x += (x - old_mean_x) * (x - mean_x)
        m2_y += (y - old_mean_y) * (y - mean_y)
        m2_xy += (x - old_mean_x) * (y - mean_y)
    sigma_x = (m2_x / count) ** 0.5
    sigma_y = (m2_y / count) ** 0.5
    return m2_xy / (count * sigma_x * sigma_y)


from hypothesis import given
from numpy import corrcoef, isclose

from utils import same_len_lists


@given(same_len_lists())
def test_pearson(list_pair):
    xs, ys = list_pair
    expected = corrcoef(xs, ys)[0, 1]
    assert isclose(pearson_correlation(xs, ys), expected)
    assert isclose(pearson_correlation_welford_onepass(xs, ys), expected)


if __name__ == "__main__":
    test_pearson()
