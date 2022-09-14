import networkx as nx
from networkx import DiGraph
import os
from typing import Dict
from collections import Counter


# TODO: make working with the new read_action_graph method
def read_action_graph(recipe_action_file: str) -> Dict[str, Dict]:
    """
    Read in a file with the action graph of a recipe
    :param recipe_action_file: the .conllu file for the recipe
    :return: dictionary with the nodes and edges of the action graph
            'nodes': a dictionary with an item for each node in the graph
                        key: the id of the first token of this node (i.e. with B-A label)
                            'label': all tokens belonging to this node
                            'ids': the token ids of all tokens
            'edges: a list of tuples, (source_node_id, target_node_id) for all edges
    """
    node_dict = dict()
    edge_list = []

    with open(recipe_action_file, "r", encoding="utf-8") as grf:
        complete_token = ""
        complete_ids = []
        prev_id = 0
        for line in grf:
            columns = line.strip().split()
            id = columns[0]
            token = columns[1]
            label = columns[4]
            edge = columns[6]
            edge_label = columns[7]

            if label == "O":
                if complete_token != "":
                    node_dict[prev_id] = {"label": complete_token, "ids": complete_ids}
                    complete_token = ""
                    complete_ids = []

            elif label[0] == "B":
                if complete_token != "":
                    node_dict[prev_id] = {"label": complete_token, "ids": complete_ids}

                complete_token = token
                complete_ids = [id]
                prev_id = id
                if edge != "0":
                    edge_list.append((id, edge))

            elif label[0] == "I":
                complete_token += " " + token
                complete_ids.append(id)

        if complete_token != "":
            node_dict[prev_id] = {"label": complete_token, "ids": complete_ids}

    return {'nodes': node_dict, 'edges': edge_list}


def conllu2graph(recipe_file: str) -> DiGraph:
    """
    Reads an action graph from a .conllu file and converts it into a network x graph
    :param recipe_file: the .conllu file with the parsed action graph
    :return: network x graph for the action graph
    """
    graph_dict = read_action_graph(recipe_file)

    action_graph = nx.DiGraph()
    action_graph.add_nodes_from(graph_dict['nodes'].items())
    action_graph.add_edges_from(graph_dict['edges'])

    children = []
    parents = []

    for edge in graph_dict['edges']:
        children.append(edge[0])
        parents.append(edge[1])

    connect_to_end = []
    action_graph.add_nodes_from([('start', {'label': 'start'})])
    action_graph.add_nodes_from([('end', {'label': 'end'})])
    for node in graph_dict['nodes']:
        if node in parents and node not in children:
            connect_to_end.append(node)
            action_graph.add_edge(node, 'end')
        if node in children and node not in parents:
            action_graph.add_edge('start', node)

    if len(connect_to_end) != 1:
        print(recipe_file)

    return action_graph


def linearize_action_graph(ac_graph: DiGraph) -> DiGraph:
    """
    Returns a linearized version of the action graph, i.e. chooses the longest path from
    'start' node to 'end' node and creates a new graph with all nodes on this path
    and the direct predecessor nodes
    :param ac_graph: network x Graph object, original graph
    :return: network x Graph object, linearized graph
    """
    # get all paths from 'start' node to 'end' node
    paths_start2end = nx.all_simple_paths(ac_graph, 'start', 'end')
    paths_start2end = list(paths_start2end)

    # find the longest path
    current_max_len = 0
    longest_path = []
    for path in paths_start2end:
        current_len = len(path)
        if current_len > current_max_len:
            current_max_len = current_len
            longest_path = path

    # create a new directed graph with all nodes and edges on the path
    # and all direct predecessors
    lin_graph = nx.DiGraph()
    prev_node = 'start'
    for node in longest_path:
        attr = nx.get_node_attributes(ac_graph, 'label')[node]
        lin_graph.add_nodes_from([(node, {'label': attr})])

        if node != 'start':
            lin_graph.add_edge(prev_node, node)
            prev_node = node

        if node != 'start' and node != 'end':
            predecessors = ac_graph.predecessors(node)
            for pred in predecessors:
                lin_graph.add_edge(pred, node)
                attr = nx.get_node_attributes(ac_graph, 'label')[pred]
                lin_graph.add_nodes_from([(pred, {'label': attr})])

    return lin_graph


def get_action_nx_graphs(action_graph_dir) -> Dict[str, Dict]:
    """
    For all recipes for all dishes in action_graph_dir create a network x graph of the action graph
    :param action_graph_dir:
    :return: a dictionary
            {dish1: {recipe1: graph, recipe2: graph, ...}, dish2: ...}
    """

    action_graphs = dict()

    for dish in os.listdir(action_graph_dir):
        action_graphs[dish] = dict()

        for recipe in os.listdir('/'.join([action_graph_dir, dish, 'recipes'])):

            ac_graph = conllu2graph('/'.join([action_graph_dir, dish, 'recipes', recipe]))
            action_graphs[dish][recipe] = ac_graph

    return action_graphs


def create_linear_action_graphs(action_graph_dict: dict) -> Dict[str, Dict]:
    """
    For all recipes for all dishes in action_graph_dict create a linear graph of the action
    graph with only the direct parents
    :param action_graph_dict: a dictionary
            {dish1: {recipe1: graph, recipe2: graph, ...}, dish2: ...}
    :return: a dictionary
                {dish1: {recipe1: linearized_graph, recipe2: linearized_graph, ...}, dish2: ...}
    """

    linearized_action_graphs = dict()

    for dish, recipe_dict in action_graph_dict.items():
        linearized_action_graphs[dish] = dict()
        for recipe, ac_graph in recipe_dict.items():

            lin_ac_graph = linearize_action_graph(ac_graph)
            linearized_action_graphs[dish][recipe] = lin_ac_graph

    return linearized_action_graphs


def create_dot_file_graph(lin_ac_graph: DiGraph, file_name: str):
    """
    Takes a network x directed graph and writes it in a file in the dot language
    :param lin_ac_graph: a
    :param file_name: path/filename for the dot file, should have .gv ending
    :return:
    """
    dot_graph = nx.nx_pydot.to_pydot(lin_ac_graph)
    dot_graph.write(file_name, encoding="utf-8")


def linearize_corpus(corpus_dir):
    # TODO: go on with actual alignments between graphs
    action_graphs = get_action_nx_graphs(corpus_dir)
    linearized_action_graphs = create_linear_action_graphs(action_graphs)


def count_lost_nodes(corpus_dir):
    """
    Inspect how many action nodes get lost during the linearization process
    :param corpus_dir:
    :return:
    """
    action_graphs = get_action_nx_graphs(corpus_dir)
    linearized_action_graphs = create_linear_action_graphs(action_graphs)

    lost_nodes = dict()

    for dish, recipe_dict in action_graphs.items():
        for recipe, original_graph in recipe_dict.items():

            linearized_graph = linearized_action_graphs[dish][recipe]
            lost_nodes[recipe] = 0

            for orig_node in original_graph.nodes:
                if orig_node not in linearized_graph.nodes:
                    lost_nodes[recipe] += 1

    counts_per_recipe = list(lost_nodes.values())
    total = sum(counts_per_recipe)
    min_nodes = min(counts_per_recipe)
    max_nodes = max(counts_per_recipe)
    avg_nodes = total / len(counts_per_recipe)
    print(f'Total number of lost nodes: {total}')
    print(f'Minimum number of nodes lost for a graph: {min_nodes}')
    print(f'Maximum number of nodes lost for a graph: {max_nodes}')
    print(f'Average number of nodes lost: {avg_nodes}')
    print(Counter(counts_per_recipe))


if __name__=="__main__":

    count_lost_nodes('../../Corpora/Ara_Punctuation/new_ara_data_new_action_graphs')

