import networkx as nx
from collections import defaultdict
from itertools import product, combinations
from typing import List, Dict, Tuple


def get_paths_between_actions(amr_graph: nx.Graph,
                              aligned_nodes: List[Tuple],
                              ignore: bool = True,
                              shortest: bool = True) -> Dict[Tuple, List[List[Tuple]]]:
    """
    For pairs of nodes in a sentence-level AMR that are aligned to an action node
    Find all paths of shortest length between the two nodes
    :param amr_graph: one sentence-level networkX.Graph AMR graph that has alignments to several action nodes
    :param aligned_nodes: list of the pairs of nodes of the AMR and aligned action nodes
    :param ignore: whether 'you' and 'imperative' AMR nodes  should be ignored
    :param shortest: whether for each amr node only the path to the closest other AMR node should be considered
    :return: dictionary
            keys: all pairings of AMR nodes that have an alignment to an action node
            values: list of all (shortest) paths between the nodes,
                    paths are edges -> list of tuples of the nodes ,
                    e.g. path from node1 to node3 could be: [(node1, node2), (node2, node3)]
    """
    path_dict = dict()

    amr_nodes = [pair[0] for pair in aligned_nodes]
    amr_node_pairs = compute_amr_node_pairs(amr_graph, amr_nodes, ignore, shortest)
    for (node, node2) in amr_node_pairs:

        shortest_paths = find_shortest_path(amr_graph, node, node2)

        path_dict[(node, node2)] = shortest_paths

    return path_dict


def find_shortest_path(graph: nx.Graph, node1, node2):
    """
    Finds all shortest paths between two nodes in a graph
    :param graph: networkx graph
    :param node1: node in the graph
    :param node2: another node in the graph
    :return: a list of all paths between node1 and node2 of minimum distance between the nodes
            paths are edges -> list of tuples of the nodes ,
                    e.g. path from node1 to node3 could be: [(node1, node2), (node2, node3)]
    """

    # needs to be a undirected graph to find all paths
    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    paths = nx.all_simple_edge_paths(G=graph, source=node1, target=node2)
    paths = list(paths)

    current_min_len = 80  # just any large number
    for p in paths:
        current_len = len(p)
        if current_len < current_min_len:
            current_min_len = current_len

    shortest_paths = [p for p in paths if len(p) == current_min_len]

    return shortest_paths


def get_all_paths(graph: nx.Graph, node1, node2):
    """
    Finds all paths between two nodes in a graph
    :param graph: networkx graph
    :param node1: node in the graph
    :param node2: another node in the graph
    :return: a list of all paths between node1 and node2 sorted by length in ascending order
             paths are edges -> list of tuples of the nodes ,
                    e.g. path from node1 to node3 could be: [(node1, node2), (node2, node3)]
    """

    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    paths = nx.all_simple_edge_paths(G=graph, source=node1, target=node2)
    paths = list(paths)
    paths.sort(key=len)

    return paths


def pair_all_nodes(node_list: List) -> List[Tuple]:
    """
    Create all pairings of the nodes in node_list ignoring ordering,
    e.g. if node_list = [1,2,3] then node_pairs = [(1,2), (1,3), (2,3)]
    :param node_list: list of nodes
    :return: list of node pairs
    """
    node_pairs = list(combinations(node_list, 1))

    return node_pairs


def pair_clustered_nodes(node_clusters: List[List]) -> List[Tuple]:
    """
    Creates all pairings of nodes belonging to different clusters
    e.g. if node_clusters = [[1,2], [3,4], [5]] then
    node_pairs = [(1,3), (1,4), (2,3), (2,4), (1,5), (2,5), (3,5), (4,5)]
    :param node_clusters: list of list of nodes, i.e. list of node clusters
    :return: list of the node pairs
    """
    node_pairs = []

    cluster_pairs = combinations(node_clusters, 2)
    for cl_pair in cluster_pairs:
        cluster_node_pairs = list(product(cl_pair[0], cl_pair[1]))
        node_pairs.extend(cluster_node_pairs)

    return node_pairs


def pair_closest_clustered_nodes(graph: nx.Graph, node_clusters: List[List]):
    """
    For each node in the node_list find the closest other node in the graph from node_list
    :param graph: a network x graph
    :param node_clusters: a list of list of nodes, i.e. list of clustered nodes from the graph
    :return: list of pairs of the nodes from node_clusters such that each node is paired with the
            closest other node form another cluster
            if there are several closest nodes than a pair for all of them is included
            e.g. if "n1" is equally close to "n2" as to "n3" and no other node from a different
            cluster is closer than the output includes both ("n1", "n2") and ("n1", "n3")
    """

    undirected_graph = graph.to_undirected()
    node_pairs = []

    node_list = []
    for cluster in node_clusters:
        node_list.extend(node_list)

    for cluster in node_clusters:
        for node in cluster:
            current_shortest_distance = 80  # just any high number
            current_closest_nodes = []
            for node2 in node_list:
                if node2 in cluster:        # do not pair nodes from the same cluster
                    continue

                shortest_paths = find_shortest_path(undirected_graph, node, node2)
                shortest_len = len(shortest_paths[0])
                if shortest_len < current_shortest_distance:
                    current_shortest_distance = shortest_len
                    current_closest_nodes = [node2]

                elif shortest_len == current_shortest_distance:
                    current_closest_nodes.append(node2)

            assert current_closest_nodes != []

            for close_node in current_closest_nodes:
                node_pairs.append((node, close_node))

    # remove duplicates
    unique_pairs = set(node_pairs)
    unique_pairs_unordered = []
    for pair in unique_pairs:
        if (pair[1], pair[0]) not in unique_pairs_unordered:
            unique_pairs_unordered.append(pair)

    return unique_pairs_unordered


def get_labelled_path(amr_graph: nx.Graph, unlabelled_paths) -> List[List]:
    """
    Converts paths where edges are represented by a list of node pairs into paths where edges are represented by
    a list of the edge labels
    e.g. original path: [('a', 's'), ('s', 'y')]
         created path: ['op1', 'ARG1]
    :param amr_graph: network x graph of the amr
    :param unlabelled_paths: list of paths, i.e. list of list of node pairs
    :return: list of labelled paths, i.e. list of list of edge labels
    """
    labelled_paths = []

    for path in unlabelled_paths:
        edge_labels = []

        for edge in path:
            node1 = edge[0]
            node2 = edge[1]

            attributes = amr_graph.get_edge_data(node1, node2)  # get the label for the edge

            if attributes:
                edge_labels.append(attributes['label'])
            # if there does not exist the edge from the path in the directed AMR, check the opposite edge direction
            # and add '-of' to the label
            else:
                attributes = amr_graph.get_edge_data(node2, node1)
                new_label = attributes['label'] + '-of'
                edge_labels.append(new_label)

        labelled_paths.append(edge_labels)

    return labelled_paths


def get_path_triples(amr_graph: nx.Graph, unlabelled_paths) -> List[List]:
    """
        Converts paths where edges are represented by a list of node pairs into paths where edges are represented by
        a list of triples of the label of node1, edge label and label of node2
        e.g. original path: [('a', 's'), ('s', 'y')]
             created path: [('and', 'op1', 'stir-01), ('stir-01, 'ARG1', 'you')]
        :param amr_graph: network x graph of the amr
        :param unlabelled_paths: list of paths, i.e. list of list of node pairs
        :return: list of labelled paths, i.e. list of list of node and edge labels
        """
    labelled_paths = []

    for path in unlabelled_paths:
        labelled_triples = []

        for edge in path:
            node1 = edge[0]
            node2 = edge[1]
            node1_lab = nx.get_node_attributes(amr_graph, 'label')[node1]
            node2_lab = nx.get_node_attributes(amr_graph, 'label')[node2]

            attributes = amr_graph.get_edge_data(node1, node2)  # get the label for the edge

            if attributes:
                edge_label = attributes['label']
            # if there does not exist the edge from the path in the directed AMR, check the opposite edge direction
            # and add '-of' to the label
            else:
                attributes = amr_graph.get_edge_data(node2, node1)
                edge_label = attributes['label'] + '-of'

            labelled_triples.append((node1_lab, edge_label, node2_lab))

        labelled_paths.append(labelled_triples)

    return labelled_paths
