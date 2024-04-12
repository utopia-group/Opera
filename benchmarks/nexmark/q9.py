# -- -------------------------------------------------------------------------------------------------
# -- Query 9: Winning Bids (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- Find the winning bid for each auction.
# -- -------------------------------------------------------------------------------------------------
#
# INSERT INTO discard_sink
# SELECT
#     id, itemName, description, initialBid, reserve, dateTime, expires, seller, category, extra,
#     auction, bidder, price, bid_dateTime, bid_extra
# FROM (
#    SELECT A.*, B.auction, B.bidder, B.price, B.dateTime AS bid_dateTime, B.extra AS bid_extra,
#      ROW_NUMBER() OVER (PARTITION BY A.id ORDER BY B.price DESC, B.dateTime ASC) AS rownum
#    FROM auction A, bid B
#    WHERE A.id = B.auction AND B.dateTime BETWEEN A.dateTime AND A.expires
# )
# WHERE rownum <= 1;

"""
func = "query"
stream_type = "auction"

[[args]]
bids = "%bid"
"""


def query(xs, bids):
    result = {}
    for auction in xs:
        for bid in bids:
            if (
                auction["id"] == bid["auction"]
            ):
                if auction["id"] not in result:
                    result = {
                        **result,
                        auction["id"]: bid
                    }
                else:
                    if result[auction["id"]]["price"] < bid["price"]:
                        result = {
                            **result,
                            auction["id"]: bid
                        }

    return result


def query_online(prev_out, bids, auction):
    result = prev_out
    for bid in bids:
        if (
            auction["id"] == bid["auction"]
        ):
            if auction["id"] not in result:
                result = {
                    **result,
                    auction["id"]: bid
                }
            else:
                if result[auction["id"]]["price"] < bid["price"]:
                    result = {
                        **result,
                        auction["id"]: bid
                    }

    return result
