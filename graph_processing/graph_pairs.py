import networkx as nx
import os
from .read_graphs import read_aligned_amr_file, read_action_graph
from typing import Dict, List, Tuple
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman


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

    action_graph_dir = str(action_graph_dir)
    amr_graph_dir = str(amr_graph_dir)

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
