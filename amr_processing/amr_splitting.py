from utils.graph_pairs import get_graph_pairs
from utils.paths import ARA_DIR, SENT_AMR_DIR, get_new_amr_dir
from utils.helpers import count_aligned_actions
from utils.splitting_preconditions import cluster_action_aligned_amr_nodes


def split_recipe_amrs():

    # read amrs and action graphs
    ara_corpus = ARA_DIR
    amr_corpus = SENT_AMR_DIR
    graph_pairs = get_graph_pairs(ara_corpus, amr_corpus)

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

            # go on with splitting process / decision if num_aligned_actions > 1
            action_clusters = cluster_action_aligned_amr_nodes(amr_graph, action_nodes)


            # decide which ones to keep together, which ones to split

            # create the pairings of the amr nodes to split

            # split

            # post processsing: i.e. variable naming, new sentence ID etc

            # save AMR to file

if __name__=='__main__':
    split_recipe_amrs()