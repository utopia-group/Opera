# # Import the required library
# -- -------------------------------------------------------------------------------------------------
# -- Query 3: Local Item Suggestion
# -- -------------------------------------------------------------------------------------------------
# -- Who is selling in OR, ID or CA in category 10, and for what auction ids?
# -- Illustrates an incremental join (using per-key state and timer) and filter.
# -- -------------------------------------------------------------------------------------------------
#
# INSERT INTO discard_sink
# SELECT
#     P.name, P.city, P.state, A.id
# FROM
#     auction AS A INNER JOIN person AS P on A.seller = P.id
# WHERE
#     A.category = 10 and (P.state = 'OR' OR P.state = 'ID' OR P.state = 'CA');

"""
func = "query"
stream_type = "auction"

[[args]]
person = "%person"
"""


def query(xs, person):
    nexmark_q3 = []
    for A in xs:
        # look up the seller in the person list
        for P in person:
            if A["seller"] == P["id"]:
                # check the necessary conditions
                if A["category"] == 2 and P["state"] != "OR":
                    # append the result to the output list
                    nexmark_q3 = [*nexmark_q3, {"city": P["city"], "state": P["state"], "id": A["id"]}]

    return nexmark_q3

def query_online(prev_out, person, x):
    for P in person:
            if x["seller"] == P["id"]:
                # check the necessary conditions
                if x["category"] == 2 and P["state"] != "OR":
                    # append the result to the output list
                    prev_out = [*prev_out, {"city": P["city"], "state": P["state"], "id": x["id"]}]
    return prev_out
    
