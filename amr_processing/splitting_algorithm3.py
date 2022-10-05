import networkx as nx
from typing import List, Dict
from .paths_between_actions import pair_clustered_nodes, get_all_paths, get_triples_single_path, pair_nodes_from_cluster
from .helpers import find_direction_changes, includes_node_from_list, includes_all_nodes_from_list, cluster_dict2list


def split_amr3(amr_graph: nx.DiGraph, action_clusters: List[Dict]) -> List[nx.Graph]:
    """

    :param amr_graph:
    :param action_clusters:
    :return:
    """

    amr_clusters, _ = cluster_dict2list(action_clusters=action_clusters)

    final_split_amrs = []

    for current_cluster in amr_clusters:
        cluster_pairings = pair_nodes_from_cluster(cluster_to_pair=current_cluster, all_clusters=amr_clusters)
        manipulated_amr, undirected_manipulated_amr, removed_edges = remove_meeting_edges(amr_graph=amr_graph,
                                                                                          node_pairs=cluster_pairings)

        if not manipulated_amr:
            # TODO: what to do in this case?
            print(f'AMR graph {amr_graph.name} was not separable.')
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
        print(f'AMR graph {amr_graph.name} was not separable without losing nodes.')
        final_split_amrs = [amr_graph]

    return final_split_amrs


def check_for_lost_nodes(original_amr: nx.Graph, separated_amrs: List[nx.Graph]):

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



def remove_meeting_edges(amr_graph: nx.Graph, node_pairs: list):
    """

    :param amr_graph:
    :param node_pairs:
    :return: manipulated_amr: nx.Graph, same as input graph with all meeting nodes and adjacent edges removed
             undirected_manipulated_amr: same as manipulated_amr but undirected
             removed_edges: dictionary with removed_meeting_node: [adjacent edges] for each removed node
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
                return_type, meeting_edge, meeting_node = find_meeting_edge_fallback(manipulated_amr, con_path)
            else:
                return_type, meeting_edge, meeting_node = find_meeting_edge(manipulated_amr, con_path)

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

    return manipulated_amr, undirected_manipulated_amr, removed_edges


def find_meeting_edge(amr_graph, path):
    """

    :param amr_graph:
    :param path:
    :return:
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


def find_meeting_edge_fallback(amr_graph, path):
    """

    :param amr_graph:
    :param path:
    :return:
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