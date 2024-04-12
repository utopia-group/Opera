"""
func = "find_repeats"
"""


def find_repeats(xs):
    counts = {}
    for x in xs:
        counts = {**counts, x: counts.get(x, 0) + 1}

    counts_2 = {}
    for xc in list(counts.items()):
        if xc[1] > 1:
            counts_2 = {**counts_2, xc[0]: xc[1]}

    return counts_2

def find_repeats_online(prev_out, prev_counts, x):
    counts = {**prev_counts, x: prev_counts.get(x, 0) + 1}
    counts_2 = {}
    for xc in list(counts.items()):
        if xc[1] > 1:
            counts_2 = {**counts_2, xc[0]: xc[1]}
    return counts_2, counts