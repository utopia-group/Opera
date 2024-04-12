"""
func = "covariance"
stream_type = "(float, float)"
find_good_input = false
"""


def covariance(xs):
    x_sum = 0
    y_sum = 0
    for p in xs:
        x_sum += p[0]
        y_sum += p[1]
    x_mean = x_sum / len(xs)
    y_mean = y_sum / len(xs)

    cov = 0
    for p in xs:
        cov += (p[0] - x_mean) * (p[1] - y_mean)

    return cov / (len(xs) - 1)

def cov_offline(xs):
    x_sum = 0
    y_sum = 0
    for p in xs:
        x_sum += p[0]
        y_sum += p[1]
    x_mean = x_sum / len(xs)
    y_mean = y_sum / len(xs)

    cov = 0
    for p in xs:
        cov += (p[0] - x_mean) * (p[1] - y_mean)

    return cov / (len(xs) - 1)


"""
phi_0 := ret = (xs0 - x_avg) * (ys0 - y_avg) + (xs1 - x_avg) * (ys1 - y_avg) + (xs2 - x_avg) * (ys2 - y_avg) and
  x_avg = (xs0 + xs1 + xs2) / 3 and y_avg = (ys0 + ys1 + ys2) / 3;

phi_1 := ret = (xs0 - prev_x_avg) * (ys0 - prev_y_avg) + (xs1 - prev_x_avg) * (ys1 - prev_y_avg) and
  prev_x_avg = (xs0 + xs1) / 2 and prev_y_avg = (ys0 + ys1) / 2;

phi := ex({xs0, xs1, ys0, ys1}, phi_0 and phi_1);

2*ret - 3*x_avg*y_avg + 3*x_avg*ys2 + 3*xs2*y_avg - 3*xs2*ys2 = 0 and 2*
prev_y_avg - 3*y_avg + ys2 = 0 and 2*prev_x_avg - 3*x_avg + xs2 = 0 and 4*
prev_x_avg*prev_y_avg - 6*prev_x_avg*y_avg + 2*prev_x_avg*ys2 - 6*prev_y_avg*
x_avg + 2*prev_y_avg*xs2 - 2*ret + 9*x_avg*y_avg - 3*x_avg*ys2 - 3*xs2*y_avg + 
xs2*ys2 = 0


or 2*prev_y_avg - 3*y_avg + ys2 = 0 and 2*prev_x_avg - 3*x_avg + xs2
 = 0 and 2*prev_x_avg*prev_y_avg - 3*prev_x_avg*y_avg + prev_x_avg*ys2 - 3*
prev_y_avg*x_avg + prev_y_avg*xs2 + 3*x_avg*y_avg - xs2*ys2 = 0$

"""

def cov_online(prev_out, prev_x_sum, prev_y_sum, prev_cov, prev_len, x, y):
    x_sum = prev_x_sum + x
    y_sum = prev_y_sum + y
    y_mean = y_sum / (prev_len + 1)
    cov = prev_cov + (x - prev_x_sum / prev_len) * (y - y_mean)
    return cov, x_sum, y_sum, cov, prev_len + 1

def cov_opera(prev_out, prev_x_sum, prev_y_sum, prev_cov, prev_len, x, y):
    x_sum = prev_x_sum + x
    y_sum = prev_y_sum + y
    y_mean = y_sum / (prev_len + 1)
    cov = ((prev_len**2*x*y - prev_len*prev_x_sum*x - prev_len*prev_y_sum*y + prev_out*(prev_len**3 - prev_len) + prev_x_sum*prev_y_sum)/(prev_len**2 + prev_len)) if prev_len > 0 else 0
    return cov, x_sum, y_sum, cov, prev_len + 1


def covariance_welford_onepass(xs, ys):
    x_sum = 0
    y_sum = 0
    x_mean = 0
    y_mean = 0
    xy_m2 = 0
    count = 0
    for x, y in zip(xs, ys):
        prev_len = count
        x__1 = x
        x__2 = y
        prev_x_sum = x_sum
        prev_y_sum = y_sum
        prev_out = 0 if count < 2 else (xy_m2 / (count - 1))

        x_sum += x
        y_sum += y
        count += 1
        x_old_mean = x_mean
        x_mean += (x - x_mean) / count
        
        y_mean += (y - y_mean) / count
        # xy_m2 += (x - x_old_mean) * (y - y_mean)

        xy_m2 = ((prev_len**2*x__1*x__2 - prev_len*prev_x_sum*x__2 - prev_len*prev_y_sum*x__1 + prev_out*(prev_len**3 - prev_len) + prev_x_sum*prev_y_sum)/(prev_len**2 + prev_len)) if prev_len > 0 else 0
    return xy_m2 / (count - 1)


from hypothesis import given
from numpy import cov, isclose

from utils import same_len_lists


@given(same_len_lists())
def main(list_pair):
    xs, ys = list_pair
    expected = cov(xs, ys)[0, 1]
    assert isclose(covariance(list(zip(xs, ys))), expected)
    assert isclose(covariance_welford_onepass(xs, ys), expected)


if __name__ == "__main__":
    main()
