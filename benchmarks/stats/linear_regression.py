"""
func = "linreg"
stream_type = "(float, float)"
find_good_input = false
"""

def linreg(xs: list[tuple[float, float]]) -> float:
    x_sum = 0.0
    y_sum = 0.0
    for p in xs:
        x_sum += p[0]
        y_sum += p[1]
    
    x_mean = x_sum / len(xs)
    y_mean = y_sum / len(xs)

    num = 0.0
    den = 0.0
    for p in xs:
        num += (p[0] - x_mean) * (p[1] - y_mean)
        den += (p[0] - x_mean) ** 2
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope

def linear_regression(xs: list[tuple[float, float]]) -> float:
    x_sum = 0.0
    y_sum = 0.0
    for p in xs:
        x_sum += p[0]
        y_sum += p[1]
    
    x_mean = x_sum / len(xs)
    y_mean = y_sum / len(xs)

    num = 0.0
    den = 0.0
    for p in xs:
        num += (p[0] - x_mean) * (p[1] - y_mean)
        den += (p[0] - x_mean) ** 2
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope


def linear_regression_online(
    prev_slope: float,
    prev_intercept: float,
    prev_x_mean: float,
    prev_y_mean: float,
    prev_num: float,
    prev_den: float,
    n: int,
    x: float,
    y: float,
) -> tuple[float, float, float, float, float, float, int]:
    x_mean = prev_x_mean + (x - prev_x_mean) / n
    y_mean = prev_y_mean + (y - prev_y_mean) / n

    num = prev_num + (x - prev_x_mean) * (y - y_mean)
    den = prev_den + (x - prev_x_mean) * (x - x_mean)

    if den == 0:
        slope = prev_slope
    else:
        slope = num / den

    intercept = y_mean - slope * x_mean
    return slope, intercept, x_mean, y_mean, num, den, n + 1


from hypothesis import given
from numpy import isclose
from scipy.stats import linregress

from utils import same_len_lists


@given(same_len_lists())
def main(list_pair):
    xs, ys = list_pair

    expected_slope, expected_intercept, _, _, _ = linregress(xs, ys)
    slope, intercept = linreg(list(zip(xs, ys)))
    assert isclose(slope, expected_slope)
    assert isclose(intercept, expected_intercept)

    state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
    for x, y in zip(xs, ys):
        state = linear_regression_online(*state, x, y)
    slope, intercept, _, _, _, _, _ = state
    assert isclose(slope, expected_slope)
    assert isclose(intercept, expected_intercept)


if __name__ == "__main__":
    main()
