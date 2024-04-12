# -- -------------------------------------------------------------------------------------------------
# -- Query 4: Average Price for a Category
# -- -------------------------------------------------------------------------------------------------
# -- Select the average of the wining bid prices for all auctions in each category.
# -- Illustrates complex join and aggregation.
# -- -------------------------------------------------------------------------------------------------
#
# INSERT INTO discard_sink
# SELECT
#     Q.category,
#     AVG(Q.final)
# FROM (
#     SELECT MAX(B.price) AS final, A.category
#     FROM auction A, bid B
#     WHERE A.id = B.auction AND B.dateTime BETWEEN A.dateTime AND A.expires
#     GROUP BY A.id, A.category
# ) Q
# GROUP BY Q.category;

"""
func = "query"
stream_type = "auction"

[[args]]
bid = "%bid"
"""


def query(xs, bid):
    result = (
        {}
    )  # A dictionary where the keys are categories and the values are pairs of (sum, count)

    # First loop for each auction entry
    for a in xs:
        max_price = 0

        # Second loop to find bids that match this auction
        for b in bid:
            if a["id"] == b["auction"] and a["dateTime"] != b["dateTime"]:
                if b["price"] > max_price:
                    max_price = b["price"]

        # Update the sum and count for the category of the auction, if there was a matching bid
        if max_price > 0:
            if a["category"] in result:
                result = {**result, a["category"]: (result[a["category"]][0] + max_price, result[a["category"]][1] + 1)}
            else:
                result = {**result, a["category"]: (max_price, 1)}

    return {category: s / c for category, (s, c) in result.items()}

def query_online(prev_out, prev_result, bid, a):
    max_price = 0
    result = prev_result

    # Second loop to find bids that match this auction
    for b in bid:
        if a["id"] == b["auction"] and a["dateTime"] != b["dateTime"]:
            if b["price"] > max_price:
                max_price = b["price"]

    # Update the sum and count for the category of the auction, if there was a matching bid
    if max_price > 0:
        if a["category"] in result:
            result = {**result, a["category"]: (result[a["category"]][0] + max_price, result[a["category"]][1] + 1)}
        else:
            result = {**result, a["category"]: (max_price, 1)}

    return {category: s / c for category, (s, c) in result.items()}
