"""
func = "tstd_offline"
find_good_input = false


[[args]]
limit_u = 90
"""

def tstd_offline(xs: list[float], limit_u: float) -> float:
    s = 0.0
    n = 0
    for x in xs:
        if x <= limit_u:
            s += x
            n += 1
    avg = s / n

    sq_s = 0.0
    for x in xs:
        if x <= limit_u:
            sq_s += (x - avg) ** 2
    return sq_s ** 0.5 / n


def tstd_online(
    prev_out: float,
    prev_n: int,
    prev_s: float,
    prev_sq_s: float,
    x: float,
    limit_u: float,
) -> tuple[float, int, float, float]:
    """
    Compute the trimmed mean incrementally.

    This function finds the arithmetic mean of given values, ignoring values outside the given limits.
    """
    if x > limit_u:
        return prev_out, prev_n, prev_s, prev_sq_s
    n = prev_n + 1
    s = prev_s + x
    avg = s / n
    sq_s = prev_sq_s + (x - avg) * (x - prev_s / prev_n) if prev_n else 0
    return (sq_s ** 0.5 / n), n, s, sq_s


def tstd_opera(
    prev_out: float,
    prev_n: int,
    prev_s: float,
    prev_sq_s: float,
    x: float,
    limit_u: float,
) -> tuple[float, int, float, float]:
    """
    Compute the trimmed mean incrementally.

    This function finds the arithmetic mean of given values, ignoring values outside the given limits.
    """
    if x > limit_u:
        return prev_out, prev_n, prev_s, prev_sq_s
    n = prev_n + 1
    s = prev_s + x
    sq_s = prev_sq_s + ((prev_n**2*x**2 - 2*prev_n*prev_s*x + prev_s**2 + prev_sq_s*(prev_n**2 + prev_n))/(prev_n**2 + prev_n)) if prev_n else 0
    return (sq_s ** 0.5 / n), n, s, sq_s
