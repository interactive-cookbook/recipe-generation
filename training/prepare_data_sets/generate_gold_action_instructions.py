
# take each recipe
# look at all split AMRs
# if got split then look at action graph to get ordering
# start by choosing all tokens from the original sentence that have an alignment to one of the nodes
# what about unaligned tokens?
# if they were also not part of the original sentence-level AMR then AMR does not represent them -> copy them also
# but not all of them -> need to figure out which ones; probably those that are surrounded by included tokens or
# preceed or follow it
# if they were part of the original sentence-level AMR then the node was removed, e.g. 'and', 'before', 'after' and
# the corresponding token should not get added
# then look at sentences and think about next steps, e.g. what to do with the determiners
