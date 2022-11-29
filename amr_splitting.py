import networkx
import penman
from collections import defaultdict
from typing import List
from copy import deepcopy
from graph_processing.action_amr_graph_mappings import get_graph_pairs
from utils.paths import ARA_DIR, SENT_AMR_DIR, get_new_dish_dir, get_splitting_log_path, get_non_sep_log_path
from amr_processing.helpers import count_aligned_actions, post_process_imperative
from amr_processing.post_processing_splitting import postprocess_split_amrs, update_alignments
from amr_processing.splitting_preconditions import cluster_action_aligned_amr_nodes
from amr_processing.splitting_algorithm import split_amr
from amr_processing.penman_networkx_conversions import networkx2penman


"""
Main script to create the action-level AMR corpus from a sentence-level AMR corpus 
"""


def split_recipe_amrs():
    """
    Runs the AMR splitting algorithm in order to create a corpus of action-level AMR graphs from the sentence-level
    AMR graphs
    The sentence level AMR graphs need to be in the SENT_AMR_DIR
    :return:
    """

    # create the log file for the non-separable amr graphs
    non_sep_log = get_non_sep_log_path()

    # read amrs and action graphs and pair them
    ara_corpus = ARA_DIR
    amr_corpus = SENT_AMR_DIR
    graph_pairs = get_graph_pairs(ara_corpus, amr_corpus)

    # for splitting_log.txt
    number_of_action_nodes = 0
    amrs_to_split_before_clustering = 0
    amrs_to_split_after_clustering = 0
    actions_per_amr = defaultdict(int)
    clusters_per_amr = defaultdict(int)
    total_number_amrs = 0

    # for each recipe
    for recipe in graph_pairs.keys():

        graph_pairs[recipe]['split_amrs'] = []

        action_graph = graph_pairs[recipe]['action']
        number_of_action_nodes += len(list(action_graph.nodes))
        sentence_amrs = graph_pairs[recipe]['amrs']
        action_graph_nodes = list(action_graph.nodes)

        # for each amr graph
        for amr_graph in sentence_amrs:

            # check number of action nodes covered by the AMR
            assert action_graph_nodes != []

            num_aligned_actions = count_aligned_actions(amr_graph, action_graph_nodes)
            actions_per_amr[num_aligned_actions] += 1

            # keep AMRs not aligned to any action in order to be able to decide what to do with them
            # keep AMRs for one action unchanged
            if num_aligned_actions == 0 or num_aligned_actions == 1:
                amr_graph.graph['snt_id'] = amr_graph.graph['id']
                graph_pairs[recipe]['split_amrs'].append(amr_graph)
                continue

            amrs_to_split_before_clustering += 1

            # go on with splitting process / decision if num_aligned_actions > 1
            # decide which action node pairs to keep together, which ones to split
            action_clusters = cluster_action_aligned_amr_nodes(amr_graph, action_graph_nodes)

            # if clustering of amr nodes and action nodes leaves only one cluster then it will not get split
            # but only the main action node should be part of the alignment attribute
            if len(action_clusters) == 1:
                clusters_per_amr[len(action_clusters)] += 1
                amr_graph_cp = deepcopy(amr_graph)
                new_amr_graph = update_alignments(sep_graph=amr_graph_cp,
                                                  orig_graph=amr_graph,
                                                  action_clusters=action_clusters)
                new_amr_graph.graph['snt_id'] = amr_graph.graph['id']
                graph_pairs[recipe]['split_amrs'].append(new_amr_graph)
                continue

            amrs_to_split_after_clustering += 1
            clusters_per_amr[len(action_clusters)] += 1

            # split the AMR
            separated_amrs = split_amr(amr_graph, action_clusters, non_sep_log)

            # post processsing: i.e. new sentence ID and alignment attribute and add a main action node
            if len(separated_amrs) > 1:
                post_processed_amrs = postprocess_split_amrs(separated_amrs, amr_graph, action_graph, action_clusters)
                graph_pairs[recipe]['split_amrs'].extend(post_processed_amrs)
            # if amr could not be split then the only changes from the post processing that are needed
            # are the alignment updates
            else:
                amr_graph = separated_amrs[0]
                amr_graph_cp = deepcopy(amr_graph)
                new_amr_graph = update_alignments(sep_graph=amr_graph_cp,
                                                  orig_graph=amr_graph,
                                                  action_clusters=action_clusters)
                new_amr_graph.graph['snt_id'] = amr_graph.graph['id']
                graph_pairs[recipe]['split_amrs'].append(new_amr_graph)

        # take care of imperative and implicit 'you'
        modified_split_amrs = []
        for sep_amr in graph_pairs[recipe]['split_amrs']:
            modified_split_amrs.append(post_process_imperative(sep_amr))
        graph_pairs[recipe]['split_amrs'] = modified_split_amrs
        # save AMR to file
        save_split_amrs(recipe, graph_pairs[recipe]['split_amrs'])
        total_number_amrs += len(graph_pairs[recipe]['split_amrs'])

    # Statistics
    clusters_per_amr[1] += actions_per_amr[1]
    action_nodes_aligned = 0
    for key, value in actions_per_amr.items():
        action_nodes_aligned += key * value
    number_of_clusters = 0
    for key, value in clusters_per_amr.items():
        number_of_clusters += key * value

    with open(get_splitting_log_path(), 'w', encoding='utf-8') as log_file:
        log_file.write(f'Number of action nodes: {number_of_action_nodes}\n')
        log_file.write(f'Number of action nodes aligned: {action_nodes_aligned}\n')
        log_file.write(f'Number of action clusters: {number_of_clusters}\n')
        log_file.write(f'Action nodes per AMR before clustering: {str(actions_per_amr)}\n')
        log_file.write(f'Action nodes per AMR after clustering: {str(clusters_per_amr)}\n')
        log_file.write(f'AMRs to split before clustering: {str(amrs_to_split_before_clustering)}\n')
        log_file.write(f'AMRs to split after clustering: {str(amrs_to_split_after_clustering)}\n')
        log_file.write(f'Total number of AMRs before splitting: {amrs_to_split_before_clustering + actions_per_amr[1] + actions_per_amr[0]}\n')
        log_file.write(f'Total number of AMRs after splitting: {total_number_amrs}\n')


def save_split_amrs(recipe_name: str, amr_list: List[networkx.Graph]):
    """
    Converts and encodes all networkX AMR graphs from amr_list into the penman string format
    and saves them in a file called [recipe_name]_instructions_amr.txt
    e.g. if recipe_name is baked_ziti_0, then the file baked_ziti_0_instructions_amr.txt will be saved in
    ACTION_AMR_DIR/baked_ziti/
    :param recipe_name: the name of the recipe
    :param amr_list: List of all action-level AMRs for the recipe
    :return:
    """
    dish_name = recipe_name.split('_')[:-1]
    dish_name = '_'.join(dish_name)
    dish_dir = str(get_new_dish_dir(dish_name))

    with open(f'{dish_dir}/{recipe_name}_instructions_amr.txt', 'w', encoding='utf-8') as new_file:
        for instr_amr in amr_list:

            penman_amr = networkx2penman(instr_amr)
            try:
                amr_string = penman.encode(penman_amr)
                new_file.write(f'{amr_string}\n\n')
            except:
                print(f'{instr_amr.name} possibly unconnected. Could not be added to file.')



if __name__=='__main__':
    split_recipe_amrs()
