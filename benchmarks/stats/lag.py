"""
func = "lag"

[[args]]
b = 3
"""


def lag(xs: list[float], b: int) -> list[float]:
    r = [0.0 for _ in range(b)]
    for x in xs:
        r = [*r[1:], x]
    return r


def lag_online(r: list[float], x: float) -> list[float]:
    r.pop(0)
    r.append(x)
    return r
