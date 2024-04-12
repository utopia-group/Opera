# -- -------------------------------------------------------------------------------------------------
# -- Query 13: Bounded Side Input Join (Not in original suite)
# -- -------------------------------------------------------------------------------------------------
# -- Joins a stream to a bounded side input, modeling basic stream enrichment.
# -- -------------------------------------------------------------------------------------------------

"""
func = "query"
stream_type = "bid"

[[args]]
side_inputs = "%side_input"
"""


def query(xs, side_inputs):
    results = []
    for b in xs:
        for s in side_inputs:
            if b["auction"] == s["key"]:
                results = [*results, (b, s["value"])]
    return results

def query_online(prev_out, side_inputs, b):
    results = prev_out
    for s in side_inputs:
        if b["auction"] == s["key"]:
            results = [*results, (b, s["value"])]
    return results
