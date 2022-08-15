from prelim_inspection_amr_action_alignments import get_graph_pairs
from penman_networkx_conversions import networkx2penman
import networkx as nx
from typing import List, Dict
from collections import defaultdict, Counter
import operator
import penman


def inspect_action2multiple_amr_node_labels(action_graph_dir: str, amr_graph_dir: str, ignore: bool) -> Dict[str, int]:
    """
    Extract the labels of all AMR nodes that are aligned to the same action node and count how often
    each of these labels is a label of a node that is not the only node aligned to a specific action node
    :param action_graph_dir:
    :param amr_graph_dir:
    :param ignore: whether nodes with label 'you' and 'imperative' should be ignored
    :return: Dictionary with the node labels as keys and the counts as values
    """

    amr_ac_graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)

    count_nodes_per_action = []
    node_labels = []

    for recipe_name in amr_ac_graph_pairs.keys():

        action_graph = amr_ac_graph_pairs[recipe_name]['action']
        amr_graphs = amr_ac_graph_pairs[recipe_name]['amrs']

        # For each action node, find all AMR nodes that are aligned to it
        amr_nodes_per_action_node = find_aligned_amr_nodes(action_graph, amr_graphs, ignore)

        for action_node, amr_nodes in amr_nodes_per_action_node.items():
            count_nodes_per_action.append(len(amr_nodes))

            if len(amr_nodes) > 1:
                node_labels.extend(amr_nodes)

    number_of_amr_nodes_per_action = Counter(count_nodes_per_action)
    multi_times_aligned_amr_nodes = Counter(node_labels)

    print(number_of_amr_nodes_per_action)

    return multi_times_aligned_amr_nodes


def get_amr2same_action_overview(action_graph_dir: str, amr_graph_dir: str, ignore: bool) -> None:
    """
    Create an overview over the labels of nodes that are aligned to the same action node
    :param action_graph_dir:
    :param amr_graph_dir:
    :param ignore: whether nodes with label 'you' and 'imperative' should be ignored
    :return:
    """
    multi_times_aligned_amr_nodes = inspect_action2multiple_amr_node_labels(action_graph_dir, amr_graph_dir, ignore)
    amr_nodes_sorted_by_freq = sorted(multi_times_aligned_amr_nodes.items(), key=operator.itemgetter(1))

    with open('./amr_nodes_aligned_to_same_action_node.csv', 'w', encoding='utf-8') as out:
        for (node_label, count) in amr_nodes_sorted_by_freq:
            out.write(f'{node_label}\t{count}\n')


def find_aligned_amr_nodes(action_graph: nx.Graph, amr_graphs: List[nx.Graph], ignore=False, labels=True) -> Dict:
    """
    For each action node in the action graph of a recipe, find the nodes in all the amrs of all
     instructions of that recipe that are aligned to the action node
    :param action_graph: an action graph for a recipe
    :param amr_graph: list of amrs for the recipe instructions
    :param ignore: whether 'you' and 'imperative' AMR nodes should be ignored
    :param labels:
    :return: Dictionary with the action nodes as keys and list of aligned AMR nodes as values
             if labels = True: then the AMR nodes are represented by their labels
             if labels = False: then the AMR nodes are represented by their node variables
    """

    alignments_per_action_node = defaultdict(list)

    for amr_graph in amr_graphs:

        aligned_nodes_amr = find_aligned_amr_nodes_single_amr(action_graph, amr_graph, ignore, labels)

        for action_node, amr_nodes in aligned_nodes_amr.items():
            alignments_per_action_node[action_node].extend(amr_nodes)

    alignments_per_action_node = dict(alignments_per_action_node)

    return alignments_per_action_node


def find_aligned_amr_nodes_single_amr(action_graph: nx.Graph, amr_graph: nx.Graph, ignore=False, labels=False) -> Dict[str, List]:
    """
    For each action node in the action graph, find the nodes in the input amr that are aligned to it
    :param action_graph: an action graph for a recipe
    :param amr_graph: an AMR graph corresponding to one instruction of the recipe
    :param ignore: whether 'you' and 'imperative' AMR nodes should be ignored
    :param labels:
    :return: Dictionary with the action nodes as keys and list of aligned AMR nodes as values
             if labels = True: then the AMR nodes are represented by their labels
             if labels = False: then the AMR nodes are represented by their node variables
    """
    alignments_per_action_node = defaultdict(list)

    action_nodes = list(action_graph.nodes)

    for node in list(amr_graph.nodes):
        token_id = nx.get_node_attributes(amr_graph, 'alignment')[node]
        if token_id in action_nodes:
            node_label = nx.get_node_attributes(amr_graph, 'label')[node]
            if ignore:
                if node_label != 'you' and node_label != 'imperative':
                    if labels:
                        alignments_per_action_node[token_id].append(node_label)
                    else:
                        alignments_per_action_node[token_id].append(node)
            else:
                if labels:
                    alignments_per_action_node[token_id].append(node_label)
                else:
                    alignments_per_action_node[token_id].append(node)

    alignments_per_action_node = dict(alignments_per_action_node)

    return alignments_per_action_node


def check_connectivity(action_graph_dir: str, amr_graph_dir: str, ignore: bool):
    """
    Check whether those AMR nodes that are aligned to the same action node form a connected subgraph
    of the specific AMR
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :param ignore:  whether 'you' and 'imperative' AMR nodes should be ignored
    :return: None but prints for how many action nodes the aligned AMR nodes are connected and for how many
            they are not; (if only one AMR node is aligned it is counted as connected)
    """
    connected_groups = 0
    unconnected_groups = 0
    amrs_with_unconnected_groups = []

    amr_ac_graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)

    for recipe_name in amr_ac_graph_pairs.keys():

        action_graph = amr_ac_graph_pairs[recipe_name]['action']
        amr_graphs = amr_ac_graph_pairs[recipe_name]['amrs']

        for amr in amr_graphs:
            aligned_nodes = find_aligned_amr_nodes_single_amr(action_graph, amr, ignore, False)

            for action_node, amr_nodes in aligned_nodes.items():
                sub_amr = amr.subgraph(amr_nodes)
                undirected_sub_amr = sub_amr.to_undirected()

                connected = nx.is_connected(undirected_sub_amr)
                if connected:
                    connected_groups += 1
                else:
                    unconnected_groups += 1
                    amrs_with_unconnected_groups.append(amr)

                    with open('./unconnected_amrs.txt', "a", encoding="utf-8") as f:
                        pen_amr = networkx2penman(amr)
                        pen_amr.metadata['alignments'] = ', '.join(amr_nodes)

                        pen_str = penman.encode(pen_amr)
                        f.write(f'{pen_str}\n\n')

    print(f'Connected: {connected_groups}')
    print(f'Unconnected: {unconnected_groups}')

    for amr in amrs_with_unconnected_groups:
        print(amr)


if __name__=="__main__":
    check_connectivity('../../Corpora/Mapped_Ara/new_ara_data_new_action_graphs',
                                            './aligned_recipe_amrs_ibm', True)

    # ignore=False
    #Connected: 1579
    #Unconnected: 27

    # ignore=True
    #Connected 1858
    #Unconnected 18