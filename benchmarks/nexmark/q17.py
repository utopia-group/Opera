# -- -------------------------------------------------------------------------------------------------
# -- Query 17: Auction Statistics Report (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- How many bids on an auction made a day and what is the price?
# -- Illustrates an unbounded group aggregation.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""
from collections import defaultdict


def query(xs):
    report = defaultdict(lambda: [0, 0, 0, 0, float("inf"), float("-inf"), 0, 0])
    for x in xs:
        key = (
            x["auction"],
            (x["dateTime"].year, x["dateTime"].month, x["dateTime"].day),
        )

        report = defaultdict(
            lambda: [0, 0, 0, 0, float("inf"), float("-inf"), 0, 0],
            {
                **report,
                key: [
                    report[key][0] + 1,
                    report[key][1] + (1 if x["price"] < 10000 else 0),
                    report[key][2] + (1 if 10000 <= x["price"] < 1000000 else 0),
                    report[key][3] + (1 if x["price"] >= 1000000 else 0),
                    min(report[key][4], x["price"]),
                    max(report[key][5], x["price"]),
                    report[key][6] + x["price"],
                    0,
                ],
            },
        )
    return {
        key: [
            value[0],
            value[1],
            value[2],
            value[3],
            value[4],
            value[5],
            value[6],
            value[6] / value[0],
        ]
        for key, value in report.items()
    }

def query_online(prev_out, prev_report, x):
    report = prev_report
    key = (
        x["auction"],
        (x["dateTime"].year, x["dateTime"].month, x["dateTime"].day),
    )

    report = defaultdict(
        lambda: [0, 0, 0, 0, float("inf"), float("-inf"), 0, 0],
        {
            **report,
            key: [
                report[key][0] + 1,
                report[key][1] + (1 if x["price"] < 10000 else 0),
                report[key][2] + (1 if 10000 <= x["price"] < 1000000 else 0),
                report[key][3] + (1 if x["price"] >= 1000000 else 0),
                min(report[key][4], x["price"]),
                max(report[key][5], x["price"]),
                report[key][6] + x["price"],
                0,
            ],
        },
    )
    return {
        key: [
            value[0],
            value[1],
            value[2],
            value[3],
            value[4],
            value[5],
            value[6],
            value[6] / value[0],
        ]
        for key, value in report.items()
    }