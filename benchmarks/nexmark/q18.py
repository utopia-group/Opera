# -- -------------------------------------------------------------------------------------------------
# -- Query 18: Find last bid (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- What's a's last bid for bidder to auction?
# -- Illustrates a Deduplicate query.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"
"""


def query(xs):
    tmp = {}

    for row in xs:
        key = (row["bidder"], row["auction"])

        if key not in tmp:
            tmp = {
                **tmp,
                key: row,
            }

    return list(tmp.values())

def query_online(prev_out, prev_tmp, row):
    tmp = prev_tmp

    key = (row["bidder"], row["auction"])

    if key not in tmp:
        tmp = {
            **tmp,
            key: row,
        }

    return list(tmp.values())
