"""
func = "query"
"""


def query(xs):
    new_bid = []
    for x in xs:
        new_bid = [*new_bid, x]
    return new_bid


def query_online(prev_out, x):
    return [*prev_out, x]
