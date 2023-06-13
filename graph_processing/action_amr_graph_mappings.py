import networkx as nx
import os
from .read_graphs import read_aligned_amr_file
from .recipe_graph import read_graph_from_conllu
from typing import Dict, List
from amr_processing.penman_networkx_conversions import penman2networkx
from utils.paths import ACTION_AMR_DIR

"""
Function to pair each action recipe graph with all aligned sentence-level AMR graphs 
Function to add the corresponding action-level AMR graph to each node of an action graph
"""


def get_graph_pairs(action_graph_dir, amr_graph_dir) -> Dict:
    """
    Reads all action graphs and AMR graphs and pairs the corresponding graphs with each other
    The amr graphs should be sentence-level AMR graphs!
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return: dictionary with one subdict per recipe, each containing the action graph and the
            sentence-level AMRs for the recipe
            {recipe1: {'action': action_graph, 'amrs': [amr_instr1, amr_instr2, ...]}, recipe2: {}}
            The action graphs and the amr graphs are networkX.Graph objects
            Each AMR graph gets a new graph (i.e. metadata) attribute 'alignments' with the node variables
            of all nodes aligned to an action node, as a string, e.g. "z1, n, t"
    """
    graph_pairs = dict()

    action_graph_dir = str(action_graph_dir)
    amr_graph_dir = str(amr_graph_dir)

    for dish in os.listdir(action_graph_dir):
        for recipe in os.listdir('/'.join([action_graph_dir, dish, 'recipes'])):

            ac_graph = read_graph_from_conllu('/'.join([action_graph_dir, dish, 'recipes', recipe]), False)
            action_nodes = list(ac_graph.nodes)

            recipe_name = recipe.split('.')[:-1]
            recipe_name = '.'.join(recipe_name)
            corresponding_amr = recipe_name + '_sentences_amr.txt'
            amr_graphs = read_aligned_amr_file('/'.join([amr_graph_dir, dish, corresponding_amr]))

            nx_amr_graphs = []

            for pen_gr in amr_graphs:
                aligned_amr_nodes = []
                nx_graph = penman2networkx(pen_gr)
                for node in list(nx_graph.nodes):
                    token_id = nx.get_node_attributes(nx_graph, 'alignment')[node]  # get the token id of the aligned token

                    if token_id in action_nodes:
                        aligned_amr_nodes.append(node)

                nx_graph.graph['alignments'] = aligned_amr_nodes
                nx_amr_graphs.append(nx_graph)

            graph_pairs[recipe_name] = dict()
            graph_pairs[recipe_name]['action'] = ac_graph
            graph_pairs[recipe_name]['amrs'] = nx_amr_graphs

    return graph_pairs


def add_semantic_representations(action_graph: nx.Graph) -> nx.Graph:
    """
    Add the semantic representation, i.e. the action-level AMR graph, to each node in the action graph
    AMR graph gets added to the corresponding node attributes
        action node: {..., 'amr': networkX amr graph}
    :param action_graph: a networkX object; action graph following graph structure in recipe_graph.py
    :return: the same action graph with the AMR graphs (networkX graph objects) added
    """

    # Determine from which original recipes the action nodes stem
    node_origins = list(set(nx.get_node_attributes(action_graph, 'origin').values()))
    relevant_recipes = [origin[2:] for origin in node_origins]  # remove 'G_'
    dish_name = '_'.join(relevant_recipes[0].split('_')[:-1])    # remove the specific recipe number to get the dish
    assert all([name.startswith(dish_name) for name in relevant_recipes])   # all recipes should be for the same dish

    # get all AMR graphs from the relevant recipes
    relevant_amrs = dict()
    for rel_recipe in relevant_recipes:
        corresponding_amrs_pen = read_aligned_amr_file(os.path.join(ACTION_AMR_DIR, dish_name, rel_recipe+'_instructions_amr.txt'))
        corresponding_amrs = [penman2networkx(pen_gr) for pen_gr in corresponding_amrs_pen]
        relevant_amrs['G_' + rel_recipe] = corresponding_amrs

    # find the corresponding AMR graph for each node and add it as attribute
    for action_node, ac_node_attr in action_graph.nodes(data=True):
        if action_node == 'end':    # skip 'end' node to avoid Key error below ('end' node has empty attr dict)
            continue
        origin = ac_node_attr['origin']
        potential_amrs = relevant_amrs[origin]
        corresponding_amr = find_action_aligned_amr(action_node, potential_amrs)
        ac_node_attr['amr'] = corresponding_amr
        action_graph.add_nodes_from([(action_node, ac_node_attr)])

    return action_graph


def find_action_aligned_amr(action_node: str, amr_graphs: List[nx.Graph]):
    """
    Find the amr graph from the input list which corresponds to the action_node,
    i.e. which includes a node aligned to the same token as the action node
    :param action_node: the action node name, i.e. id of the action token
    :param amr_graphs: a list of amr graphs as networkX graphs
    :return: the amr graph from the list which is aligned to the action node
            None if no corresponding amr graph was found in the list
    """
    for amr in amr_graphs:
        action_aligned_amr_nodes = amr.graph['alignments']
        for amr_node in action_aligned_amr_nodes:
            aligned_token = nx.get_node_attributes(amr, 'alignment')[amr_node]
            if aligned_token == action_node:
                return amr

    print(f'Warning: no aligned AMR graph found for action node {action_node}')
    return None

