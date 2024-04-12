# -- -------------------------------------------------------------------------------------------------
# -- Query 12: Processing Time Windows (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- How many bids does a user make within a fixed processing time limit?
# -- Illustrates working in processing time window.
# --
# -- Group bids by the same user into processing time windows of 10 seconds.
# -- Emit the count of bids per window.
# -- -------------------------------------------------------------------------------------------------
#
# INSERT INTO discard_sink
# SELECT
#     B.bidder,
#     count(*) as bid_count,
#     TUMBLE_START(B.p_time, INTERVAL '10' SECOND) as starttime,
#     TUMBLE_END(B.p_time, INTERVAL '10' SECOND) as endtime
# FROM (SELECT *, PROCTIME() as p_time FROM bid) B
# GROUP BY B.bidder, TUMBLE(B.p_time, INTERVAL '10' SECOND);

"""
func = "query"
stream_type = "bid"

[[args]]
tx = 100000
"""

import time


def query(xs, tx):
    # Hold the bid times for each user:
    bid_times = {}

    # resulting data
    result = []

    for bid in xs:
        # Remove all timestamps older than 10 seconds:
        bid_times = {
            **bid_times,
            bid["bidder"]: [
                t for t in bid_times.setdefault(bid["bidder"], []) if tx - t <= 10
            ]
            + [tx],
        }

        result = [
            *result,
            {
                "bidder": bid["bidder"],
                "bid_count": len(bid_times[bid["bidder"]]),
                "starttime": tx- 10,  # Approximate start
                "endtime": tx,  # Approximate end
            },
        ]

    return result

def query_online(prev_out, prev_bid_times, tx, bid):
    bid_times = prev_bid_times
    result = prev_out

    bid_times = {
        **bid_times,
        bid["bidder"]: [
            t for t in bid_times.setdefault(bid["bidder"], []) if tx - t <= 10
        ]
        + [tx],
    }

    result = [
        *result,
        {
            "bidder": bid["bidder"],
            "bid_count": len(bid_times[bid["bidder"]]),
            "starttime": tx- 10,  # Approximate start
            "endtime": tx,  # Approximate end
        },
    ]

    return result
