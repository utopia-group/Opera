"""
func = "query"
stream_type = "bid"
"""
from collections import defaultdict


def query(xs):
    create_dict = lambda iter: defaultdict(
        lambda: defaultdict(
            lambda: dict(
                {
                    "total_bids": 0,
                    "total_bidders": [],
                    "total_auctions": [],
                    "rank1_bids": 0,
                    "rank1_bidders": [],
                    "rank1_auctions": [],
                    "rank2_bids": 0,
                    "rank2_bidders": [],
                    "rank2_auctions": [],
                    "rank3_bids": 0,
                    "rank3_bidders": [],
                    "rank3_auctions": [],
                }
            )
        ),
        iter,
    )
    channel_stats = create_dict({})
    for row in xs:
        day = (row["dateTime"].year, row["dateTime"].month, row["dateTime"].day)

        channel_stats = create_dict(
            {
                **channel_stats,
                row["channel"]: defaultdict(
                    lambda: dict(
                        {
                            "total_bids": 0,
                            "total_bidders": [],
                            "total_auctions": [],
                            "rank1_bids": 0,
                            "rank1_bidders": [],
                            "rank1_auctions": [],
                            "rank2_bids": 0,
                            "rank2_bidders": [],
                            "rank2_auctions": [],
                            "rank3_bids": 0,
                            "rank3_bidders": [],
                            "rank3_auctions": [],
                        }
                    ),
                    {
                        **channel_stats[row["channel"]],
                        day: {
                            **channel_stats[row["channel"]][day],
                            "total_bids": channel_stats[row["channel"]][day][
                                "total_bids"
                            ]
                            + 1,
                            "total_bidders": channel_stats[row["channel"]][day][
                                "total_bidders"
                            ]
                            + [row["bidder"]],
                            "total_auctions": channel_stats[row["channel"]][day][
                                "total_auctions"
                            ]
                            + [row["auction"]],
                            "rank1_bids": channel_stats[row["channel"]][day][
                                "rank1_bids"
                            ]
                            if row["price"] >= 10000
                            else channel_stats[row["channel"]][day]["rank1_bids"] + 1,
                            "rank1_bidders": channel_stats[row["channel"]][day][
                                "rank1_bidders"
                            ]
                            if row["price"] >= 10000
                            else channel_stats[row["channel"]][day]["rank1_bidders"]
                            + [row["bidder"]],
                            "rank1_auctions": channel_stats[row["channel"]][day][
                                "rank1_auctions"
                            ]
                            if row["price"] >= 10000
                            else channel_stats[row["channel"]][day]["rank1_auctions"]
                            + [row["auction"]],
                            "rank2_bids": channel_stats[row["channel"]][day][
                                "rank2_bids"
                            ]
                            if 10000 <= row["price"] < 1000000
                            else channel_stats[row["channel"]][day]["rank2_bids"] + 1,
                            "rank2_bidders": channel_stats[row["channel"]][day][
                                "rank2_bidders"
                            ]
                            if 10000 <= row["price"] < 1000000
                            else channel_stats[row["channel"]][day]["rank2_bidders"]
                            + [row["bidder"]],
                            "rank2_auctions": channel_stats[row["channel"]][day][
                                "rank2_auctions"
                            ]
                            if 10000 <= row["price"] < 1000000
                            else channel_stats[row["channel"]][day]["rank2_auctions"]
                            + [row["auction"]],
                            "rank3_bids": channel_stats[row["channel"]][day][
                                "rank3_bids"
                            ]
                            if row["price"] >= 1000000
                            else channel_stats[row["channel"]][day]["rank3_bids"] + 1,
                            "rank3_bidders": channel_stats[row["channel"]][day][
                                "rank3_bidders"
                            ]
                            if row["price"] >= 1000000
                            else channel_stats[row["channel"]][day]["rank3_bidders"]
                            + [row["bidder"]],
                            "rank3_auctions": channel_stats[row["channel"]][day][
                                "rank3_auctions"
                            ]
                            if row["price"] >= 1000000
                            else channel_stats[row["channel"]][day]["rank3_auctions"]
                            + [row["auction"]],
                        },
                    },
                ),
            }
        )

    return {
        channel: {
            day: {
                key: len(value) if isinstance(value, list) else value
                for key, value in stats.items()
            }
            for day, stats in day_stats.items()
        }
        for channel, day_stats in channel_stats.items()
    }

def query_online(prev_out, prev_stats, row):
    create_dict = lambda iter: defaultdict(
        lambda: defaultdict(
            lambda: dict(
                {
                    "total_bids": 0,
                    "total_bidders": [],
                    "total_auctions": [],
                    "rank1_bids": 0,
                    "rank1_bidders": [],
                    "rank1_auctions": [],
                    "rank2_bids": 0,
                    "rank2_bidders": [],
                    "rank2_auctions": [],
                    "rank3_bids": 0,
                    "rank3_bidders": [],
                    "rank3_auctions": [],
                }
            )
        ),
        iter,
    )
    channel_stats = prev_stats
    day = (row["dateTime"].year, row["dateTime"].month, row["dateTime"].day)

    channel_stats = create_dict(
        {
            **channel_stats,
            row["channel"]: defaultdict(
                lambda: dict(
                    {
                        "total_bids": 0,
                        "total_bidders": [],
                        "total_auctions": [],
                        "rank1_bids": 0,
                        "rank1_bidders": [],
                        "rank1_auctions": [],
                        "rank2_bids": 0,
                        "rank2_bidders": [],
                        "rank2_auctions": [],
                        "rank3_bids": 0,
                        "rank3_bidders": [],
                        "rank3_auctions": [],
                    }
                ),
                {
                    **channel_stats[row["channel"]],
                    day: {
                        **channel_stats[row["channel"]][day],
                        "total_bids": channel_stats[row["channel"]][day][
                            "total_bids"
                        ]
                        + 1,
                        "total_bidders": channel_stats[row["channel"]][day][
                            "total_bidders"
                        ]
                        + [row["bidder"]],
                        "total_auctions": channel_stats[row["channel"]][day][
                            "total_auctions"
                        ]
                        + [row["auction"]],
                        "rank1_bids": channel_stats[row["channel"]][day][
                            "rank1_bids"
                        ]
                        if row["price"] >= 10000
                        else channel_stats[row["channel"]][day]["rank1_bids"] + 1,
                        "rank1_bidders": channel_stats[row["channel"]][day][
                            "rank1_bidders"
                        ]
                        if row["price"] >= 10000
                        else channel_stats[row["channel"]][day]["rank1_bidders"]
                        + [row["bidder"]],
                        "rank1_auctions": channel_stats[row["channel"]][day][
                            "rank1_auctions"
                        ]
                        if row["price"] >= 10000
                        else channel_stats[row["channel"]][day]["rank1_auctions"]
                        + [row["auction"]],
                        "rank2_bids": channel_stats[row["channel"]][day][
                            "rank2_bids"
                        ]
                        if 10000 <= row["price"] < 1000000
                        else channel_stats[row["channel"]][day]["rank2_bids"] + 1,
                        "rank2_bidders": channel_stats[row["channel"]][day][
                            "rank2_bidders"
                        ]
                        if 10000 <= row["price"] < 1000000
                        else channel_stats[row["channel"]][day]["rank2_bidders"]
                        + [row["bidder"]],
                        "rank2_auctions": channel_stats[row["channel"]][day][
                            "rank2_auctions"
                        ]
                        if 10000 <= row["price"] < 1000000
                        else channel_stats[row["channel"]][day]["rank2_auctions"]
                        + [row["auction"]],
                        "rank3_bids": channel_stats[row["channel"]][day][
                            "rank3_bids"
                        ]
                        if row["price"] >= 1000000
                        else channel_stats[row["channel"]][day]["rank3_bids"] + 1,
                        "rank3_bidders": channel_stats[row["channel"]][day][
                            "rank3_bidders"
                        ]
                        if row["price"] >= 1000000
                        else channel_stats[row["channel"]][day]["rank3_bidders"]
                        + [row["bidder"]],
                        "rank3_auctions": channel_stats[row["channel"]][day][
                            "rank3_auctions"
                        ]
                        if row["price"] >= 1000000
                        else channel_stats[row["channel"]][day]["rank3_auctions"]
                        + [row["auction"]],
                    },
                },
            ),
        }
    )

    return {
        channel: {
            day: {
                key: len(value) if isinstance(value, list) else value
                for key, value in stats.items()
            }
            for day, stats in day_stats.items()
        }
        for channel, day_stats in channel_stats.items()
    }