"""
func = "diff"
"""


def diff(xs: list[float]) -> float:
    last = 0.0
    d = 0.0
    for x in xs:
        d = x - last
        last = x
    return d

def diff_online(prev_out, prev_last, x):
    d = x - prev_last
    return d, x
