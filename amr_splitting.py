import networkx as nx

from graph_processing.graph_pairs import get_graph_pairs
from utils.paths import ARA_DIR, SENT_AMR_DIR, get_new_dish_dir
from amr_processing.helpers import count_aligned_actions, find_highest_node
from amr_processing.splitting_preconditions import cluster_action_aligned_amr_nodes
from amr_processing.splitting_algorithm import split_amr
from amr_processing.splitting_algorithm2 import split_amr2
from amr_processing.penman_networkx_conversions import networkx2penman
import penman
from collections import defaultdict


def split_recipe_amrs(version=1):

    # read amrs and action graphs
    ara_corpus = ARA_DIR
    amr_corpus = SENT_AMR_DIR
    graph_pairs = get_graph_pairs(ara_corpus, amr_corpus)

    amrs_to_split_before_clustering = 0
    amrs_to_split_after_clustering = 0
    actions_per_amr = defaultdict(int)
    clusters_per_amr = defaultdict(int)

    # for recipe amr graph
    for recipe in graph_pairs.keys():

        graph_pairs[recipe]['split_amrs'] = []

        action_graph = graph_pairs[recipe]['action']
        sentence_amrs = graph_pairs[recipe]['amrs']
        action_graph_nodes = list(action_graph.nodes)

        # for each amr graph
        for amr_graph in sentence_amrs:
            # check number of action nodes covered by the AMR
            assert action_graph_nodes != []
            num_aligned_actions = count_aligned_actions(amr_graph, action_graph_nodes)
            actions_per_amr[num_aligned_actions] += 1

            if num_aligned_actions == 0:    # ignore AMRs / sentences without actions
                continue
            elif num_aligned_actions == 1:  # keep AMRs for one action unchanged
                graph_pairs[recipe]['split_amrs'].append(amr_graph)
                continue

            amrs_to_split_before_clustering += 1

            # go on with splitting process / decision if num_aligned_actions > 1
            # decide which action node pairs to keep together, which ones to split
            action_clusters = cluster_action_aligned_amr_nodes(amr_graph, action_graph_nodes)

            # if clustering of amr nodes and action nodes leaves only one cluster then it will not get split
            if len(action_clusters) == 1:
                clusters_per_amr[len(action_clusters)] += 1
                graph_pairs[recipe]['split_amrs'].append(amr_graph)
                continue

            amrs_to_split_after_clustering += 1
            clusters_per_amr[len(action_clusters)] += 1

            #if amr_graph.name == 'waffles_0_instr5':
                #print("here")

            # split the AMR
            if version == 1:
                separated_amrs = split_amr(amr_graph, action_clusters)
            else:
                separated_amrs = split_amr2(amr_graph, action_clusters)
            #print(len(separated_amrs))

            # post processsing: i.e. new sentence ID and alignment attribute and add a main action node
            for new_id, sep_amr in enumerate(separated_amrs):
                # update graph name / instruction ID
                new_name = sep_amr.graph['id'] + f'_{new_id}'
                #if new_name == 'baked_ziti_0_instr9_1':
                    #print("here")
                sep_amr.graph['id'] = new_name
                sep_amr.name = new_name
                sep_amr.graph['name'] = new_name

                # update action node aligned amr nodes
                new_alignments = []
                for orig_al in amr_graph.graph['alignments'].split(', '):
                    if orig_al in sep_amr.nodes:
                        new_alignments.append(orig_al)
                sep_amr.graph['alignments'] = ','.join(new_alignments)

                # update root node
                if sep_amr.graph['root'] not in sep_amr.nodes:
                    for ac_cluster in action_clusters:
                        action_nodes = []
                        for ac_node, amr_nodes in ac_cluster.items():
                            for amr_n in amr_nodes:
                                if amr_n in sep_amr.nodes:
                                    action_nodes.append(ac_node)
                        action_nodes.sort()

                        # if node cluster is for other separated amr continue
                        if not action_nodes:
                            continue

                        # find the top node
                        highest_action_node = find_highest_node(action_nodes, action_graph)
                        main_amr_nodes = ac_cluster[highest_action_node]
                        sep_amr.graph['root'] = main_amr_nodes[0]
                        break

                graph_pairs[recipe]['split_amrs'].append(sep_amr)

        # save AMR to file
        save_split_amrs(recipe, graph_pairs[recipe]['split_amrs'])

    print(amrs_to_split_before_clustering)
    print(actions_per_amr)
    print(amrs_to_split_after_clustering)
    clusters_per_amr[1] += actions_per_amr[1]
    print(clusters_per_amr)


def save_split_amrs(recipe_name, amr_list):
    dish_name = recipe_name.split('_')[:-1]
    dish_name = '_'.join(dish_name)
    dish_dir = str(get_new_dish_dir(dish_name))

    with open(f'{dish_dir}/{recipe_name}_instructions_amr.txt', 'w', encoding='utf-8') as new_file:
        for instr_amr in amr_list:
            penman_amr = networkx2penman(instr_amr)
            try:
                amr_string = penman.encode(penman_amr)
            except :
                print(f'{recipe_name} possible unconnected')
                return
            new_file.write(f'{amr_string}\n\n')



if __name__=='__main__':
    split_recipe_amrs(2)
