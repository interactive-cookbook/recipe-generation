from graph_processing.graph_pairs import get_graph_pairs
from utils.paths import ARA_DIR, SENT_AMR_DIR, get_new_amr_dir
from amr_processing.helpers import count_aligned_actions
from amr_processing.splitting_preconditions import cluster_action_aligned_amr_nodes


def split_recipe_amrs():

    # read amrs and action graphs
    ara_corpus = ARA_DIR
    amr_corpus = SENT_AMR_DIR
    graph_pairs = get_graph_pairs(ara_corpus, amr_corpus)

    amrs_to_split_before_clustering = 0
    amrs_to_split_after_clustering = 0

    # for recipe amr graph
    for recipe in graph_pairs.keys():
        graph_pairs[recipe]['split_amrs'] = []

        action_graph = graph_pairs[recipe]['action']
        sentence_amrs = graph_pairs[recipe]['amrs']
        action_nodes = list(action_graph.nodes)

        # for each amr graph
        for amr_graph in sentence_amrs:

            # check number of action nodes covered by the AMR
            num_aligned_actions = count_aligned_actions(amr_graph, action_nodes)
            if num_aligned_actions == 0:    # ignore AMRs / sentences without actions
                continue
            elif num_aligned_actions == 1:  # keep AMRs for one action unchanged
                graph_pairs[recipe]['split_amrs'].append(amr_graph)
                continue

            amrs_to_split_before_clustering += 1

            # go on with splitting process / decision if num_aligned_actions > 1
            # decide which action node pairs to keep together, which ones to split
            action_clusters = cluster_action_aligned_amr_nodes(amr_graph, action_nodes)

            # if clustering of amr nodes and action nodes leaves only one cluster then it will not get split
            if len(action_clusters) == 1:
                graph_pairs[recipe]['split_amrs'].append(amr_graph)
                continue

            amrs_to_split_after_clustering += 1

            # create the pairings of the amr nodes to split

            # split

            # post processsing: i.e. variable naming, new sentence ID etc

            # save AMR to file

    print(amrs_to_split_before_clustering)
    print(amrs_to_split_after_clustering)


if __name__=='__main__':
    split_recipe_amrs()
