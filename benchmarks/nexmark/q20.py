# -- -------------------------------------------------------------------------------------------------
# -- Query 20: Expand bid with auction (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- Get bids with the corresponding auction information where category is 10.
# -- Illustrates a filter join.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"

[[args]]
auction = "%auction"
"""


def query(xs, auction):
    new_bid = []
    for x in xs:
        for y in auction:
            if x["auction"] == y["id"] and y["category"] > 2:
                new_bid = [
                    *new_bid,
                    {
                        "auction": x["auction"],
                        "bidder": x["bidder"],
                        "price": x["price"],
                        "channel": x["channel"],
                        "url": x["url"],
                        "bid_dateTime": x["dateTime"],
                        "bid_extra": x["extra"],
                        "itemName": y["itemName"],
                        "description": y["description"],
                        "initialBid": y["initialBid"],
                        "reserve": y["reserve"],
                        "auction_dateTime": y["dateTime"],
                        "expires": y["expires"],
                        "seller": y["seller"],
                        "category": y["category"],
                        "auction_extra": y["extra"],
                    },
                ]
    return new_bid


def query_online(prev_out, auction, x):
    new_bid = prev_out
    for y in auction:
        if x["auction"] == y["id"] and y["category"] > 2:
            new_bid = [
                *new_bid,
                {
                    "auction": x["auction"],
                    "bidder": x["bidder"],
                    "price": x["price"],
                    "channel": x["channel"],
                    "url": x["url"],
                    "bid_dateTime": x["dateTime"],
                    "bid_extra": x["extra"],
                    "itemName": y["itemName"],
                    "description": y["description"],
                    "initialBid": y["initialBid"],
                    "reserve": y["reserve"],
                    "auction_dateTime": y["dateTime"],
                    "expires": y["expires"],
                    "seller": y["seller"],
                    "category": y["category"],
                    "auction_extra": y["extra"],
                },
            ]
    return new_bid
