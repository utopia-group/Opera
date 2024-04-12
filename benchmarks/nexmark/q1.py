"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    new_bid = []
    for x in xs:
        new_bid = [
            *new_bid,
            {
                "auction": x["auction"],
                "bidder": x["bidder"],
                "price": 2 * x["price"],
                "dateTime": x["dateTime"],
                "extra": x["extra"],
            },
        ]
    return new_bid

def query_online(prev_out, x):
    return [
            *prev_out,
            {
                "auction": x["auction"],
                "bidder": x["bidder"],
                "price": 2 * x["price"],
                "dateTime": x["dateTime"],
                "extra": x["extra"],
            },
        ]
