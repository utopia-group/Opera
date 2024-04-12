# -- -------------------------------------------------------------------------------------------------
# -- Query 22: Get URL Directories (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- What is the directory structure of the URL?
# -- Illustrates a SPLIT_INDEX SQL.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    new_bid = []
    for x in xs:
        url_parts = x["url"].split("/")

        new_bid = [
            *new_bid,
            {
                **x,
                "dir1": url_parts[3] if len(url_parts) > 3 else None,
                "dir2": url_parts[4] if len(url_parts) > 4 else None,
                "dir3": url_parts[5] if len(url_parts) > 5 else None,
            },
        ]
    return new_bid

def query_online(prev_out, x):
    new_bid = prev_out
    url_parts = x["url"].split("/")

    new_bid = [
        *new_bid,
        {
            **x,
            "dir1": url_parts[3] if len(url_parts) > 3 else None,
            "dir2": url_parts[4] if len(url_parts) > 4 else None,
            "dir3": url_parts[5] if len(url_parts) > 5 else None,
        },
    ]
    return new_bid
