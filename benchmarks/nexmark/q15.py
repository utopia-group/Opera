# -- -------------------------------------------------------------------------------------------------
# -- Query 15: Bidding Statistics Report (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- How many distinct users join the bidding for different level of price?
# -- Illustrates multiple distinct aggregations with filters.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    # Initialize results dictionary
    results = {}

    create_dict = lambda: {
        "total_bids": 0,
        "rank1_bids": 0,
        "rank2_bids": 0,
        "rank3_bids": 0,
        "total_bidders": set(),
        "rank1_bidders": set(),
        "rank2_bidders": set(),
        "rank3_bidders": set(),
        "total_auctions": set(),
        "rank1_auctions": set(),
        "rank2_auctions": set(),
        "rank3_auctions": set(),
    }
    default_dict = create_dict()

    for bid in xs:
        # Format date
        day = (bid["dateTime"].day, bid["dateTime"].month, bid["dateTime"].year)

        results = {
            **results,
            day: {
                "total_bids": results.get(day, default_dict)["total_bids"] + 1,
                "rank1_bids": results.get(day, default_dict)["rank1_bids"] + 1 if bid["price"] < 10000 else results.get(day, default_dict)["rank1_bids"],
                "rank2_bids": results.get(day, default_dict)["rank2_bids"] + 1 if bid["price"] >= 10000 and bid["price"] < 1000000 else results.get(day, default_dict)["rank2_bids"],
                "rank3_bids": results.get(day, default_dict)["rank3_bids"] + 1 if bid["price"] >= 1000000 else results.get(day, default_dict)["rank3_bids"],
                "total_bidders": results.get(day, default_dict)["total_bidders"].union({bid["bidder"]}),
                "rank1_bidders": results.get(day, default_dict)["rank1_bidders"].union({bid["bidder"]}) if bid["price"] < 10000 else results.get(day, default_dict)["rank1_bidders"],
                "rank2_bidders": results.get(day, default_dict)["rank2_bidders"].union({bid["bidder"]}) if bid["price"] >= 10000 and bid["price"] < 1000000 else results.get(day, default_dict)["rank2_bidders"],
                "rank3_bidders": results.get(day, default_dict)["rank3_bidders"].union({bid["bidder"]}) if bid["price"] >= 1000000 else results.get(day, default_dict)["rank3_bidders"],
                "total_auctions": results.get(day, default_dict)["total_auctions"].union({bid["auction"]}),
                "rank1_auctions": results.get(day, default_dict)["rank1_auctions"].union({bid["auction"]}) if bid["price"] < 10000 else results.get(day, default_dict)["rank1_auctions"],
                "rank2_auctions": results.get(day, default_dict)["rank2_auctions"].union({bid["auction"]}) if bid["price"] >= 10000 and bid["price"] < 1000000 else results.get(day, default_dict)["rank2_auctions"],
                "rank3_auctions": results.get(day, default_dict)["rank3_auctions"].union({bid["auction"]}) if bid["price"] >= 1000000 else results.get(day, default_dict)["rank3_auctions"],
            },
        }

    # Convert sets to counts
    return {
        day: {
            "total_bids": dicts["total_bids"],
            "rank1_bids": dicts["rank1_bids"],
            "rank2_bids": dicts["rank2_bids"],
            "rank3_bids": dicts["rank3_bids"],
            "total_bidders": len(dicts["total_bidders"]),
            "rank1_bidders": len(dicts["rank1_bidders"]),
            "rank2_bidders": len(dicts["rank2_bidders"]),
            "rank3_bidders": len(dicts["rank3_bidders"]),
            "total_auctions": len(dicts["total_auctions"]),
            "rank1_auctions": len(dicts["rank1_auctions"]),
            "rank2_auctions": len(dicts["rank2_auctions"]),
            "rank3_auctions": len(dicts["rank3_auctions"]),
        }
        for day, dicts in results.items()
    }


def query_online(prev_out, prev_results, bid):
    # Initialize results dictionary
    results = prev_results

    create_dict = lambda: {
        "total_bids": 0,
        "rank1_bids": 0,
        "rank2_bids": 0,
        "rank3_bids": 0,
        "total_bidders": set(),
        "rank1_bidders": set(),
        "rank2_bidders": set(),
        "rank3_bidders": set(),
        "total_auctions": set(),
        "rank1_auctions": set(),
        "rank2_auctions": set(),
        "rank3_auctions": set(),
    }
    default_dict = create_dict()

    day = (bid["dateTime"].day, bid["dateTime"].month, bid["dateTime"].year)

    results = {
        **results,
        day: {
            "total_bids": results.get(day, default_dict)["total_bids"] + 1,
            "rank1_bids": results.get(day, default_dict)["rank1_bids"] + 1 if bid["price"] < 10000 else results.get(day, default_dict)["rank1_bids"],
            "rank2_bids": results.get(day, default_dict)["rank2_bids"] + 1 if bid["price"] >= 10000 and bid["price"] < 1000000 else results.get(day, default_dict)["rank2_bids"],
            "rank3_bids": results.get(day, default_dict)["rank3_bids"] + 1 if bid["price"] >= 1000000 else results.get(day, default_dict)["rank3_bids"],
            "total_bidders": results.get(day, default_dict)["total_bidders"].union({bid["bidder"]}),
            "rank1_bidders": results.get(day, default_dict)["rank1_bidders"].union({bid["bidder"]}) if bid["price"] < 10000 else results.get(day, default_dict)["rank1_bidders"],
            "rank2_bidders": results.get(day, default_dict)["rank2_bidders"].union({bid["bidder"]}) if bid["price"] >= 10000 and bid["price"] < 1000000 else results.get(day, default_dict)["rank2_bidders"],
            "rank3_bidders": results.get(day, default_dict)["rank3_bidders"].union({bid["bidder"]}) if bid["price"] >= 1000000 else results.get(day, default_dict)["rank3_bidders"],
            "total_auctions": results.get(day, default_dict)["total_auctions"].union({bid["auction"]}),
            "rank1_auctions": results.get(day, default_dict)["rank1_auctions"].union({bid["auction"]}) if bid["price"] < 10000 else results.get(day, default_dict)["rank1_auctions"],
            "rank2_auctions": results.get(day, default_dict)["rank2_auctions"].union({bid["auction"]}) if bid["price"] >= 10000 and bid["price"] < 1000000 else results.get(day, default_dict)["rank2_auctions"],
            "rank3_auctions": results.get(day, default_dict)["rank3_auctions"].union({bid["auction"]}) if bid["price"] >= 1000000 else results.get(day, default_dict)["rank3_auctions"],
        },
    }

    # Convert sets to counts
    return {
        day: {
            "total_bids": dicts["total_bids"],
            "rank1_bids": dicts["rank1_bids"],
            "rank2_bids": dicts["rank2_bids"],
            "rank3_bids": dicts["rank3_bids"],
            "total_bidders": len(dicts["total_bidders"]),
            "rank1_bidders": len(dicts["rank1_bidders"]),
            "rank2_bidders": len(dicts["rank2_bidders"]),
            "rank3_bidders": len(dicts["rank3_bidders"]),
            "total_auctions": len(dicts["total_auctions"]),
            "rank1_auctions": len(dicts["rank1_auctions"]),
            "rank2_auctions": len(dicts["rank2_auctions"]),
            "rank3_auctions": len(dicts["rank3_auctions"]),
        }
        for day, dicts in results.items()
    }
