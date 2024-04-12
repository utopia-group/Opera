# -- -------------------------------------------------------------------------------------------------
# -- Query 21: Add channel id (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- Add a channel_id column to the bid table.
# -- Illustrates a 'CASE WHEN' + 'REGEXP_EXTRACT' SQL.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""
import re


def query(xs):
    new_bid = []
    channel_ids = {"apple": "0", "google": "1", "facebook": "2", "baidu": "3"}

    for x in xs:
        if x["channel"].lower() in channel_ids:
            x = {**x, "channel_id": channel_ids[x["channel"].lower()]}
            new_bid = [*new_bid, x]
        else:
            if re.search(r"(&|^)channel_id=([^&]*)", x["url"]):
                x = {
                    **x,
                    "channel_id": re.search(r"(&|^)channel_id=([^&]*)", x["url"]).group(
                        2
                    ),
                }
                new_bid = [*new_bid, x]

    return new_bid

def query_online(prev_out, x):
    new_bid = prev_out
    channel_ids = {"apple": "0", "google": "1", "facebook": "2", "baidu": "3"}
    if x["channel"].lower() in channel_ids:
        x = {**x, "channel_id": channel_ids[x["channel"].lower()]}
        new_bid = [*new_bid, x]
    else:
        if re.search(r"(&|^)channel_id=([^&]*)", x["url"]):
            x = {
                **x,
                "channel_id": re.search(r"(&|^)channel_id=([^&]*)", x["url"]).group(
                    2
                ),
            }
            new_bid = [*new_bid, x]

    return new_bid
