import networkx as nx
from typing import List, Dict, Tuple
from .paths_between_actions import get_all_paths, get_triples_single_path, pair_nodes_from_cluster
from .helpers import find_direction_changes, includes_node_from_list, includes_all_nodes_from_list, cluster_dict2list


def split_amr(amr_graph: nx.DiGraph, action_clusters: List[Dict], log_path) -> List[nx.Graph]:
    """
    Splits a sentence-level AMR into action-level AMRs
    Information about which action-aligned AMR nodes are treated as belonging to one action comes from the clusters
    :param amr_graph: the sentence-level AMR
    :param action_clusters: the action-node clusters
    :param log_path: path for the file were non-separable AMRs get logged
    :return: a list of the separated AMRs
    """

    amr_clusters, _ = cluster_dict2list(action_clusters=action_clusters)

    final_split_amrs = []
    any_fallback = False

    for current_cluster in amr_clusters:
        cluster_pairings = pair_nodes_from_cluster(cluster_to_pair=current_cluster, all_clusters=amr_clusters)
        manipulated_amr, undirected_manipulated_amr, fallback = separate_current_cluster(amr_graph=amr_graph,
                                                                                         node_pairs=cluster_pairings)
        if fallback:
            any_fallback = True

        if not manipulated_amr:
            with open(log_path, 'a', encoding='utf-8') as out_file:
                out_file.write(f'AMR graph {amr_graph.name} was not separable.\n')
            return [amr_graph]

        components = nx.connected_components(undirected_manipulated_amr)
        component_subgraphs = [nx.subgraph(manipulated_amr, comp_nodes) for comp_nodes in components]

        other_cluster_nodes = [pair[1] for pair in cluster_pairings]
        subgraph_current_cluster = None
        for comp_gr in component_subgraphs:
            if includes_all_nodes_from_list(comp_gr, current_cluster) and not includes_node_from_list(comp_gr, other_cluster_nodes):
                subgraph_current_cluster = comp_gr

        assert subgraph_current_cluster

        final_split_amrs.append(subgraph_current_cluster)

    lost_nodes = check_for_lost_nodes(amr_graph, final_split_amrs)
    if lost_nodes:
        with open(log_path, 'a', encoding='utf-8') as out_file:
            out_file.write(f'AMR graph {amr_graph.name} was not separable without losing nodes.\n')
        final_split_amrs = [amr_graph]

    if any_fallback:
        with open(log_path, 'a', encoding='utf-8') as out_file:
            out_file.write(f'AMR graph {amr_graph.name} was separated with the fallback rules.\n')

    return final_split_amrs


def check_for_lost_nodes(original_amr: nx.Graph, separated_amrs: List[nx.Graph]) -> bool:
    """
    Check whether the splitting of the original AMR resulted in losing nodes of the original AMR
    i.e. check whether each node of the original AMR is included in at least one of the separated AMRs
    If a node with label 'before' or 'after' is missing, this is expected behavior and gets ignored therefore
    :param original_amr: the original sentence-level AMR
    :param separated_amrs: list of the action-level AMRs obtained by splitting the original AMR
    :return: True if at least one of the original AMR nodes is not included in any of the separate graphs
            (ignoring 'before' and 'after')
    """
    original_nodes = original_amr.nodes
    covered_nodes = []
    for sep_amr in separated_amrs:
        covered_nodes.extend(sep_amr.nodes)

    lost_nodes = set(original_nodes) - set(covered_nodes)
    unexpected_lost_nodes = []
    for node in lost_nodes:
        node_label = nx.get_node_attributes(original_amr, 'label')[node]
        if node_label != 'before' and node_label != 'after':
            unexpected_lost_nodes.append(node)

    if unexpected_lost_nodes:
        return True
    else:
        return False


def separate_current_cluster(amr_graph: nx.Graph, node_pairs: list):
    """
    Tries to remove edges and nodes from the original sentence-level AMR such that the currently considered action
    cluster gets separated from the action nodes from all other clusters
    :param amr_graph: original sentence-level AMR graph
    :param node_pairs: list of node pairs such that each pair consists of one action-aligned node from the currently
                        considered action cluster and one node from another cluster
    :return: manipulated_amr: nx.Graph, same as input graph with all meeting nodes and adjacent edges removed
             undirected_manipulated_amr: same as manipulated_amr but undirected
             fallback: whether fallback rules were used to split the graph
    """
    manipulated_amr = nx.Graph.copy(amr_graph)
    undirected_manipulated_amr = amr_graph.to_undirected()
    removed_edges = dict()
    meeting_nodes = []

    fallback_case = False

    while True:

        # consider all paths that connect two action nodes
        connecting_paths = []
        for (node1, node2) in node_pairs:
            paths_current_pair = get_all_paths(undirected_manipulated_amr, node1, node2)
            connecting_paths.extend(paths_current_pair)

        # if there are no paths left between any pair of action nodes from different clusters
        # then the splitting is done
        if not connecting_paths:
            break

        # sort the paths by lengths in order to start with the shortest ones first
        connecting_paths.sort(key=len)
        graph_changed = False  # if True then need to recompute all connecting paths

        for con_path in connecting_paths:

            if fallback_case:
                return_type, meeting_edge, meeting_node = find_edge_to_remove_fallback(manipulated_amr, con_path)
            else:
                return_type, meeting_edge, meeting_node = find_edge_to_remove(manipulated_amr, con_path)

            # if there is a meeting node / edge found than save all information that is relevant
            # for this node (i.e. data and edges) and remove the node from the manipulated AMR
            if return_type == 'edge':

                edge_attributes = manipulated_amr.edges(data=True)
                meeting_edge_data = None
                for edge_data in edge_attributes:
                    if edge_data[0] == meeting_edge[0] and edge_data[1] == meeting_edge[1]:
                        meeting_edge_data = edge_data
                assert meeting_edge_data
                removed_edges[meeting_edge] = meeting_edge_data
                meeting_nodes.append(meeting_node)

                manipulated_amr.remove_edge(meeting_edge[0], meeting_edge[1])
                undirected_manipulated_amr.remove_edge(meeting_edge[0], meeting_edge[1])

                # break the loop in order to recompute the paths again for the new resulting graph
                graph_changed = True
                break

            elif return_type == 'node':
                manipulated_amr.remove_node(meeting_node)
                undirected_manipulated_amr.remove_node(meeting_node)
                meeting_nodes.append(meeting_node)
                graph_changed = True
                break

        if graph_changed:
            continue

        elif not fallback_case:
            fallback_case = True
        else:
            return None, None, None

    return manipulated_amr, undirected_manipulated_amr, fallback_case


def find_edge_to_remove(amr_graph: nx.Graph, path) -> Tuple:
    """
    Find the node or edge to remove from the path between two action nodes to separate them appropriately
    :param amr_graph: the amr graph
    :param path: path between two action nodes
    :return: a triple
             'node', ('dummy', 'dummy'), node_to_remove if a 'before' or 'after' node should be removed
             'edge', edge_to_remove, meeting_node if an edge fulfilling the splitting criteria was found
             None, None, None if no node or edge to remove was found
    """

    path_triples = get_triples_single_path(amr_graph, path)

    path_edges = [trip[1] for trip in path_triples]
    direction_changes = find_direction_changes(path_edges)

    if len(direction_changes) == 0:
        path_nodes = [trip[0] for trip in path_triples]
        path_nodes.append(path_triples[-1][-1])
        if 'before' in path_nodes:
            node_ind = path_nodes.index('before')
        elif 'after' in path_nodes:
            node_ind = path_nodes.index('after')
        else:
            return None, None, None

        relevant_node = path[node_ind][0]
        return 'node', ('dummy', 'dummy'), relevant_node

    elif len(direction_changes) == 1:
        relevant_position = direction_changes[0]
        relevant_edge1 = path[relevant_position[0]]     # edge to meeting node
        relevant_edge2 = path[relevant_position[1]]     # edge from meeting node

        assert relevant_edge1[1] == relevant_edge2[0]   # meeting node

        # in order to remove the edge, the direction needs to match the original direction
        # i.e. direction on the connection path is different from original direction then swap edge
        relevant_edge2_label = path_edges[relevant_position[1]]
        if relevant_edge2_label.endswith('-of'):
            relevant_edge2 = tuple(reversed(relevant_edge2))

        return 'edge', relevant_edge2, relevant_edge1[1]

    # TODO: decide how to deal with these cases
    elif len(direction_changes) > 1:
        return None, None, None


def find_edge_to_remove_fallback(amr_graph: nx.Graph, path) -> Tuple:
    """
    The fallback case for splitting
    Find the edge to remove from the path between two action nodes to separate them appropriately if none
    of the main rules for splitting can be applied anymore
    :param amr_graph: the amr graph
    :param path: path between two action nodes
    :return: a triple
             'edge', edge_to_remove, meeting_node if an edge fulfilling the fallback splitting criteria was found
             None, None, None if no edge to remove was found
    """

    path_triples = get_triples_single_path(amr_graph, path)

    path_edges = [trip[1] for trip in path_triples]
    direction_changes = find_direction_changes(path_edges)

    assert len(direction_changes) != 1

    if len(direction_changes) < 1:
        return None, None, None

    else:
        # look at first meeting node
        relevant_position = direction_changes[0]
        relevant_edge1 = path[relevant_position[0]]     # edge to meeting node
        relevant_edge2 = path[relevant_position[1]]     # edge from meeting node

        assert relevant_edge1[1] == relevant_edge2[0]   # meeting node

        # in order to remove the edge, the direction needs to match the original direction
        # i.e. direction on the connection path is different from original direction then swap edge
        relevant_edge2_label = path_edges[relevant_position[1]]
        if relevant_edge2_label.endswith('-of'):
            relevant_edge2 = tuple(reversed(relevant_edge2))

        return 'edge', relevant_edge2, relevant_edge1[1]
