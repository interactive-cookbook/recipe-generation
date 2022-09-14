import networkx as nx
import penman

from graph_processing.read_graphs import read_aligned_amr_file, read_action_graph
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def get_graph_pairs(action_graph_dir, amr_graph_dir) -> Dict:
    """
    Reads all action graphs and AMR graphs and pairs the corresponding graphs with each other
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return: dictionary with one subdict per recipe, each containing the action graph and the
            sentence-level AMRs for the recipe
            {recipe1: {'action': action_graph, 'amrs': [amr_isntr1, amr_instr2, ...]}, recipe2: {}}
            The action graphs and the amr graphs are networkX.Graph objects
            Each AMR graph gets a new graph (i.e. metadata) attribute 'alignments' with the node variables
            of all nodes aligned to an action node, as a string, e.g. "z1, n, t"
    """
    graph_pairs = dict()

    for dish in os.listdir(action_graph_dir):
        for recipe in os.listdir('/'.join([action_graph_dir, dish, 'recipes'])):

            ac_graph = read_action_graph('/'.join([action_graph_dir, dish, 'recipes', recipe]))
            action_nodes = list(ac_graph.nodes)

            recipe_name = recipe.split('.')[:-1]
            recipe_name = '.'.join(recipe_name)
            corresponding_amr = recipe_name + '_sentences_amr.txt'
            amr_graphs = read_aligned_amr_file('/'.join([amr_graph_dir, dish, 'amrs', corresponding_amr]))

            nx_amr_graphs = []

            for pen_gr in amr_graphs:
                aligned_amr_nodes = []
                nx_graph = penman2networkx(pen_gr)
                for node in list(nx_graph.nodes):
                    token_id = nx.get_node_attributes(nx_graph, 'alignment')[node]  # get the token id of the aligned token

                    if token_id in action_nodes:
                        aligned_amr_nodes.append(node)

                nx_graph.graph['alignments'] = ', '.join(aligned_amr_nodes)
                nx_amr_graphs.append(nx_graph)

            graph_pairs[recipe_name] = dict()
            graph_pairs[recipe_name]['action'] = ac_graph
            graph_pairs[recipe_name]['amrs'] = nx_amr_graphs

    return graph_pairs


def inspect_amr_action_alignments(action_graph_dir, amr_graph_dir) -> None:
    """
    Computes and prints a number of counts related to the alignments between AMRs and action nodes such as
    - the number of AMRs aligned to more than one action node
    - the number of AMRs not aligned to any action node
    - the number of action nodes not aligned to any AMR node
    - the number of instructions, i.e. of AMRs
    - the number of action nodes
    and creates two files:
    - action_nodes_without_alignments.txt
    - amrs_without_action_alignment.txt
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return:
    """
    graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)
    all_one2many_amrs = []
    graphs_with_missing_alignments = dict()
    unaligned_amr_graphs = []
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

        # find all AMRs that are not aligned to any action node
        unaligned_amr_graphs.extend(get_unaligned_amrs(action_graph, amr_graphs))

    print(f'Number of AMRs aligned to more than one action node: {len(all_one2many_amrs)}\n')
    print(f'Number of AMRs not aligned to any action node: {len(unaligned_amr_graphs)}\n')
    print(f'Number of action nodes without aligned AMR: {count_missed_action_nodes}\n')
    print(f'Number of total action nodes: {count_action_nodes}')
    print(f'Number of instructions: {count_instructions}')
    print(count_alignments)

    with open('./action_nodes_without_alignments.txt', "w", encoding="utf-8") as file:
        for re_name, nodes in graphs_with_missing_alignments.items():
            for n in nodes:
                file.write(f'{re_name}\t{n}\n')

    with open('./amrs_without_action_alignment.txt', 'w', encoding="utf-8") as file:
        for un_al_amr in unaligned_amr_graphs:
            pen_gr = networkx2penman(un_al_amr)
            graph_str = penman.encode(pen_gr)
            file.write(f'{graph_str}\n\n')


def get_one2many_amrs_and_counts(action_graph: nx.Graph, amr_graphs: List[nx.Graph]) -> Tuple[List[nx.Graph], Dict]:
    """
    Finds all amr gaphs in 'amr_graphs' that are aligned to more than one node in the action_graph
    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of network.Graph AMRs for the individual instructions of the recipe
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

        for node in list(amr_graph.nodes):
            token_id = nx.get_node_attributes(amr_graph, 'alignment')[node]  # get the token id of the aligned token
            if token_id in action_nodes:                                    # check if aligned token is an action node
                if token_id not in action_nodes_already_covered:            # don't count alignments multiple times for structbart alignments
                    count += 1
                    action_nodes_already_covered.append(token_id)

        if count > 1:
            one_to_many_amrs.append(amr_graph)
        count_alignments[count] += 1

    return one_to_many_amrs, count_alignments


def get_one2many_amrs_and_alignments(action_graph: nx.Graph,
                                     amr_graphs: List[nx.Graph]) -> List[Tuple[nx.Graph, List]]:
    """
    Finds all amr gaphs in 'amr_graphs' that are aligned to more than one node in the action_graph
    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of network.Graph AMRs for the individual instructions of the recipe
    :return: returns a list of amr graphs aligned to more than one action node paired with
             the list of the action_node - amr_node alignments
             [(amr_graph, [(amr_node1, action_node1), (amr_node2, action_node2), ...]), ...]
    """
    one_to_many_amrs = []
    action_nodes = list(action_graph.nodes)

    for amr_graph in amr_graphs:

        action_amr_alignments = []

        for node in list(amr_graph.nodes):
            token_id = nx.get_node_attributes(amr_graph, 'alignment')[node]  # get the token id of the aligned token

            if token_id in action_nodes:
                action_amr_alignments.append((node, token_id))

        # extract AMRs that are aligned to more than one action node
        # but not AMRs where more than one AMR node are aligned to the same action node
        involved_action_nodes = [ac_node for (amr_node, ac_node) in action_amr_alignments]
        unique_action_nodes = set(involved_action_nodes)
        if len(unique_action_nodes) > 1:
            one_to_many_amrs.append((amr_graph, action_amr_alignments))

    return one_to_many_amrs


def get_unaligned_amrs(action_graph: nx.Graph, amr_graphs: List[nx.Graph]) -> List[nx.Graph]:
    """
    Finds all amr gaphs in 'amr_graphs' that are not aligned to any node in the action_graph
    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of networkx.Graph AMRs for the individual instructions of the recipe
    :return: list of all networkx.Graph AMRs that are not aligned to any action node
    """
    unaligned_amrs = []
    action_nodes = list(action_graph.nodes)

    for amr_graph in amr_graphs:
        aligned = False

        for node in list(amr_graph.nodes):
            token_id = nx.get_node_attributes(amr_graph, 'alignment')[node] # get the token id of the aligned token

            if token_id in action_nodes:  # check if aligned token is an action node
                aligned = True

        if not aligned:
            unaligned_amrs.append(amr_graph)

    return unaligned_amrs


def get_unaligned_action_nodes(action_graph: nx.Graph, amr_graphs: List[nx.Graph]) -> List:
    """
    Finds all nodes in the recipe action_graph that are not aligned to any node in one of the
    AMRs for the recipe instructions
    :param action_graph: networkx.Graph recipe action graph
    :param amr_graphs: list of networkx.Graph AMRs for the individual instructions of the recipe
    :return: returns a list of the unaligned action nodes
    """
    amr_node_aligned_tokens = []

    for amr_graph in amr_graphs:
        for node in list(amr_graph.nodes):
            token_id = nx.get_node_attributes(amr_graph, 'alignment')[node]
            amr_node_aligned_tokens.append(token_id)

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

    inspect_amr_action_alignments('../data/ara1.1',
                                  '../data/recipe_amrs_sentences')
