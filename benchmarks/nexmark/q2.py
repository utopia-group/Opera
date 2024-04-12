# -- -------------------------------------------------------------------------------------------------
# -- Query2: Selection
# -- -------------------------------------------------------------------------------------------------
# -- Find bids with specific auction ids and show their bid price.
# --
# -- In original Nexmark queries, Query2 is as following (in CQL syntax):
# --
# --   SELECT Rstream(auction, price)
# --   FROM Bid [NOW]
# --   WHERE auction = 1007 OR auction = 1020 OR auction = 2001 OR auction = 2019 OR auction = 2087;
# --
# -- However, that query will only yield a few hundred results over event streams of arbitrary size.
# -- To make it more interesting we instead choose bids for every 123'th auction.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    new_bid = []
    for x in xs:
        if x["auction"] > 3:
            new_bid = [*new_bid, x]
    return new_bid

def query_online(prev_out, x):
    if x["auction"] > 3:
        return [*prev_out, x]
    else:
        return prev_out
