# -- -------------------------------------------------------------------------------------------------
# -- Query 7: Highest Bid
# -- -------------------------------------------------------------------------------------------------
# -- What are the highest bids per period?
# -- Deliberately implemented using a side input to illustrate fanout.
# --
# -- The original Nexmark Query7 calculate the highest bids in the last minute.
# -- We will use a shorter window (10 seconds) to help make testing easier.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    window_size = 10  # Size of window in seconds
    highest_bids = {}
    # Initialising dictionary to store the max bid for each window period
    for b in xs:
        i = int(b["dateTime"].timestamp() // window_size)  # Identifying window period
        if i not in highest_bids or b["price"] > highest_bids[i]["price"]:
            # If new window period or this bid has higher price
            highest_bids = {
                **highest_bids,
                i: {
                    "auction": b["auction"],
                    "bidder": b["bidder"],
                    "price": b["price"],
                    "dateTime": b["dateTime"],
                    "extra": b["extra"],
                },
            }
    return highest_bids.values()

def query_online(prev_out, highest_bids, b):
    window_size = 10  # Size of window in seconds
    highest_bids = {}
    # Initialising dictionary to store the max bid for each window period
    i = int(b["dateTime"].timestamp() // window_size)  # Identifying window period
    if i not in highest_bids or b["price"] > highest_bids[i]["price"]:
        # If new window period or this bid has higher price
        highest_bids = {
            **highest_bids,
            i: {
                "auction": b["auction"],
                "bidder": b["bidder"],
                "price": b["price"],
                "dateTime": b["dateTime"],
                "extra": b["extra"],
            },
        }
    return highest_bids.values()
