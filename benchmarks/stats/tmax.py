"""
func = "tmax_offline"

[[args]]
limit_l = 10
"""

def tmax_offline(xs: list[float], limit_l: float) -> float:
    mn = 10000.0
    for x in xs:
        if limit_l <= x:
            if x < mn:
                mn = x
    return mn

def tmax_online(prev_out, prev_mn, x, limit_l):
    if limit_l <= x:
            if x < prev_mn:
                return (x, x)
    return (prev_mn, x)
    
