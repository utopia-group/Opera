# -- -------------------------------------------------------------------------------------------------
# -- Query 10: Log to File System (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- Log all events to file system. Illustrates windows streaming data into partitioned file system.
# --
# -- Every minute, save all events from the last period into partitioned log files.
# -- -----
#
# INSERT INTO fs_sink
# SELECT auction, bidder, price, dateTime, extra, DATE_FORMAT(dateTime, 'yyyy-MM-dd'), DATE_FORMAT(dateTime, 'HH:mm')
# FROM bid;

"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    new_bid = []
    for x in xs:
        t = {
            **x,
            "dt": x["dateTime"].strftime("%Y-%m-%d"),
            "hm": x["dateTime"].strftime("%H:%M"),
        }
        new_bid = [*new_bid, t]
    return new_bid

def query_online(prev_out, prev_new_bid, x):
    new_bid = prev_new_bid
    t = {
        **x,
        "dt": x["dateTime"].strftime("%Y-%m-%d"),
        "hm": x["dateTime"].strftime("%H:%M"),
    }
    new_bid = [*new_bid, t]
    return new_bid
