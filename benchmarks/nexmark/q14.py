# -- -------------------------------------------------------------------------------------------------
# -- Query 14: Calculation (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- Convert bid timestamp into types and find bids with specific price.
# -- Illustrates duplicate expressions and usage of user-defined-functions.
# -- -------------------------------------------------------------------------------------------------

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
                "price": x["price"] * 0.908,
                "bidTimeType": "dayTime"
                if 6 < x["dateTime"].hour <= 18
                else "nightTime"
                if x["dateTime"].hour <= 6 or x["dateTime"].hour >= 20
                else "otherTime",
                "dateTime": x["dateTime"],
                "extra": x["extra"],
                "c_counts": str(x["extra"]).count("1"),
            },
        ]

    return new_bid

def query_online(prev_out, x):
    new_bid = prev_out
    new_bid = [
        *new_bid,
        {
            "auction": x["auction"],
            "bidder": x["bidder"],
            "price": x["price"] * 0.908,
            "bidTimeType": "dayTime"
            if 6 < x["dateTime"].hour <= 18
            else "nightTime"
            if x["dateTime"].hour <= 6 or x["dateTime"].hour >= 20
            else "otherTime",
            "dateTime": x["dateTime"],
            "extra": x["extra"],
            "c_counts": str(x["extra"]).count("1"),
        },
    ]

    return new_bid
