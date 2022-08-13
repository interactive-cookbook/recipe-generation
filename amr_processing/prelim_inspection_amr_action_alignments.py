import networkx as nx
import penman

from read_graphs import read_aligned_amr_file, read_action_graph
import os
from penman import surface
from collections import defaultdict
from typing import Dict, List


def get_graph_pairs(action_graph_dir, amr_graph_dir) -> Dict[str: Dict]:
    """
    Reads all action graphs and AMR graphs and pairs the corresponding graphs with each other
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return: dictionary with one subdict per recipe, each containing the action graph and the
            sentence-level AMRs for the recipe
            {recipe1: {'action': action_graph, 'amrs': [amr_isntr1, amr_instr2, ...]}, recipe2: {}}
            The action graphs are networkX.Graph objects, the amr graphs are penman.Graph objects
    """
    graph_pairs = dict()

    for dish in os.listdir(action_graph_dir):
        for recipe in os.listdir('/'.join([action_graph_dir, dish, 'recipes'])):

            ac_graph = read_action_graph('/'.join([action_graph_dir, dish, 'recipes', recipe]))

            recipe_name = recipe.split('.')[:-1]
            recipe_name = '.'.join(recipe_name)
            corresponding_amr = recipe_name + '_sentences_amr.txt'
            amr_graphs = read_aligned_amr_file('/'.join([amr_graph_dir, dish, 'amrs', corresponding_amr]))

            graph_pairs[recipe_name] = dict()
            graph_pairs[recipe_name]['action'] = ac_graph
            graph_pairs[recipe_name]['amrs'] = amr_graphs

    return graph_pairs


def inspect_amr_action_alignments(action_graph_dir, amr_graph_dir):
    """

    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return:
    """
    graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)
    all_one2many_amrs = []
    graphs_with_missing_alignments = dict()
    count_missed_action_nodes = 0
    count_action_nodes = 0
    count_instructions = 0          # should correspond to the number of AMRs
    count_alignments = defaultdict(int)

    for recipe_name in graph_pairs.keys():

        action_graph = graph_pairs[recipe_name]['action']
        amr_graphs = graph_pairs[recipe_name]['amrs']
        count_instructions += len(amr_graphs)
        count_action_nodes += len(action_graph.nodes)

        # find all AMRs that are aligned to more than one action node
        one2many_amrs_recipe, alignment_counts = get_one2many_amrs_and_counts(action_graph, amr_graphs)
        all_one2many_amrs.extend(one2many_amrs_recipe)

        # update how often an AMR is aligned to X action nodes
        for key, value in alignment_counts.items():
            count_alignments[key] += value

        # find all action nodes that do not have an aligned AMR node
        non_aligned_actions = get_unaligned_action_nodes(action_graph, amr_graphs)
        if non_aligned_actions:
            graphs_with_missing_alignments[recipe_name] = non_aligned_actions
            count_missed_action_nodes += len(non_aligned_actions)

    print(f'Number of AMRs aligned to more than one action node: {len(all_one2many_amrs)}\n')
    print(f'Number of action nodes without aligned AMR: {count_missed_action_nodes}\n')
    print(f'Number of total action nodes: {count_action_nodes}')
    print(f'Number of instructions: {count_instructions}')
    print(count_alignments)

    with open('./action_nodes_without_alignments.txt', "w", encoding="utf-8") as file:
        for re_name, nodes in graphs_with_missing_alignments.items():
            for n in nodes:
                file.write(f'{re_name}\t{n}\n')


def get_one2many_amrs_and_counts(action_graph: nx.Graph, amr_graphs: List[penman.Graph]) -> (List[penman.Graph], Dict):
    """
    Finds all amr gaphs in 'amr_graphs' that are aligned to more than one node in the action_graph
    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of penman.Graph AMRs for the individual instructions of the recipe
    :return: list of all AMRs that are aligned to more than one action node
            dictionary with the counts how often an AMR is aligned to X action nodes
            e.g. {0: 3, 1: 6, ...} would mean that 3 AMRs are not aligned to any action node,
                                    6 AMRs are aligned to 1 action node
    """
    one_to_many_amrs = []
    action_nodes = list(action_graph.nodes)
    count_alignments = defaultdict(int)

    action_nodes_already_covered = []

    for amr_graph in amr_graphs:
        count = 0

        token_aligned_nodes = surface.alignments(amr_graph)     # alignments for all aligned nodes
        for instance, alignment in token_aligned_nodes.items():
            token_id = alignment.indices[0]                     # get the token id of the aligned token

            # IMPORTANT! token ids in the recipe graphs are strings but surface.alignments give ints
            if str(token_id) in action_nodes:                   # check if aligned token is an action node
                if str(token_id) not in action_nodes_already_covered:       # don't count alignments multiple times for structbart alignments
                    count += 1
                    action_nodes_already_covered.append(str(token_id))

        if count > 1:
            one_to_many_amrs.append(amr_graph)
        count_alignments[count] += 1

    return one_to_many_amrs, count_alignments


def get_one2many_amrs_and_alignments(action_graph: nx.Graph,
                                     amr_graphs: List[penman.Graph]) -> List[(penman.Graph, List)]:
    """

    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of penman.Graph AMRs for the individual instructions of the recipe
    :return: returns a list of amr graphs aligned to more than one action node paired with
             the list of the action_node - amr_node alignments
             [(amr_graph, [(amr_node1, action_node1), (amr_node2, action_node2)), ...]
    """
    one_to_many_amrs = []
    action_nodes = list(action_graph.nodes)

    for amr_graph in amr_graphs:

        token_aligned_nodes = surface.alignments(amr_graph)
        action_amr_alignments = []

        for instance, alignment in token_aligned_nodes.items():
            token_id = alignment.indices[0]
            if str(token_id) in action_nodes:
                amr_node = instance[0]
                action_amr_alignments.append((amr_node, str(token_id)))

        if len(action_amr_alignments) > 1:
            one_to_many_amrs.append((amr_graph, action_amr_alignments))

    return one_to_many_amrs


def get_unaligned_action_nodes(action_graph: nx.Graph, amr_graphs: List[penman.Graph]) -> List:
    """
    Finds all nodes in the recipe action_graph that are not aligned to any node in one of the
    AMRs for the recipe instructions
    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of penman.Graph AMRs for the individual instructions of the recipe
    :return: returns a list of the unaligned action nodes
    """
    amr_node_aligned_tokens = []

    for amr_graph in amr_graphs:
        token_aligned_nodes = surface.alignments(amr_graph)
        for instance, alignment in token_aligned_nodes.items():
            token_id = alignment.indices[0]
            amr_node_aligned_tokens.append(str(token_id))

    non_aligned_action_nodes = []

    for action_node in list(action_graph.nodes):
        if action_node not in amr_node_aligned_tokens:
            multi_token_ids = nx.get_node_attributes(action_graph, 'ids')[action_node]
            found_alignment = False
            for mt_id in multi_token_ids:
                if mt_id in amr_node_aligned_tokens:
                    found_alignment = True

            if not found_alignment:
                non_aligned_action_nodes.append(action_node)

    return non_aligned_action_nodes


if __name__=="__main__":

    inspect_amr_action_alignments('../../Corpora/Mapped_Ara/new_ara_data_new_action_graphs',
                                  './aligned_recipe_amrs_ibm')
