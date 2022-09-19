import networkx as nx
from typing import List, Dict
from .paths_between_actions import pair_clustered_nodes, get_all_paths, get_triples_single_path
from .helpers import find_direction_changes, includes_node_from_list


def split_amr(amr_graph: nx.DiGraph, action_clusters: List[Dict]) -> List[nx.Graph]:
    """

    :param amr_graph:
    :param action_clusters:
    :return:
    """
    main_action_amr_nodes = []
    amr_clusters = []
    for cluster in action_clusters:
        corresponding_amr_nodes = cluster.values()
        flattened_amr_nodes = []
        for amr_list in corresponding_amr_nodes:
            flattened_amr_nodes.extend(amr_list)
        amr_clusters.append(flattened_amr_nodes)
        main_action_amr_nodes.extend(flattened_amr_nodes)

    # pair the action nodes from different clusters
    node_pairs = pair_clustered_nodes(amr_clusters)

    manipulated_amr, undirected_manipulated_amr, removed_nodes = remove_meeting_nodes(amr_graph,
                                                                                      node_pairs,
                                                                                      main_action_amr_nodes)
    if not manipulated_amr:
        # TODO: what to do in this case?
        print(f'AMR graph {amr_graph.name} was not separable.')
        return [amr_graph]

    # now the manipulated amr should consist of at least as many components as there are
    # action node clusters

    components = nx.connected_components(undirected_manipulated_amr)
    component_subgraphs = [nx.subgraph(manipulated_amr, comp_nodes) for comp_nodes in components]

    action_subgraphs = []
    non_action_sugraphs = []
    for comp_gr in component_subgraphs:
        if includes_node_from_list(comp_gr, main_action_amr_nodes):
            action_subgraphs.append(comp_gr)
        else:
            non_action_sugraphs.append(comp_gr)

    shared_subgraphs = add_removed_nodes(non_action_sugraphs, removed_nodes, amr_graph)
    action_subgraphs_w_meeting_nodes = add_removed_nodes(action_subgraphs, removed_nodes, amr_graph)

    final_split_amrs = []

    for ac_sg in action_subgraphs_w_meeting_nodes:
        final_action_amr = ac_sg.copy()
        ac_sg_nodes = set(ac_sg.nodes)
        for sh_sg in shared_subgraphs:
            sh_sg_nodes = set(sh_sg.nodes)
            if ac_sg_nodes.intersection(sh_sg_nodes):
                final_action_amr = nx.compose(final_action_amr, sh_sg)

        final_split_amrs.append(final_action_amr)

    return final_split_amrs


def add_removed_nodes(sub_graphs: List[nx.Graph], removed_nodes: dict, original_amr: nx.Graph):
    """

    :param sub_graphs:
    :param removed_nodes:
    :param original_amr:
    :return:
    """
    new_subgraphs = []
    removed_nodes_list = list(removed_nodes.keys())

    for sub_gr in sub_graphs:
        new_subg = sub_gr.copy()
        for rm_node in removed_nodes_list:
            for rm_edge in removed_nodes[rm_node]:
                if rm_edge[0] in sub_gr.nodes or rm_edge[1] in sub_gr.nodes:
                    # if the removed node not in the graph yet then add it with its original attributes
                    if rm_node not in new_subg.nodes:
                        rm_node_attributes = original_amr.nodes(data=True)[rm_node]
                        new_subg.add_nodes_from([(rm_node, rm_node_attributes)])
                    # add the original edge(s) back
                    new_subg.add_edges_from([rm_edge])

        new_subgraphs.append(new_subg)

    return new_subgraphs


def remove_meeting_nodes(amr_graph: nx.Graph, node_pairs: list, action_amr_nodes: list):
    """

    :param amr_graph:
    :param node_pairs:
    :param action_amr_nodes:
    :return: manipulated_amr: nx.Graph, same as input graph with all meeting nodes and adjacent edges removed
             undirected_manipulated_amr: same as manipulated_amr but undirected
             removed_nodes: dictionary with removed_meeting_node: [adjacent edges] for each removed node
    """
    manipulated_amr = nx.Graph.copy(amr_graph)
    undirected_manipulated_amr = amr_graph.to_undirected()
    removed_nodes = dict()

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

            meeting_node = find_meeting_node(manipulated_amr, con_path)

            # if there is a meeting node found than save all information that is relevant
            # for this node (i.e. data and edges) and remove the node from the manipulated AMR
            if meeting_node:

                if meeting_node in action_amr_nodes:
                    print('Warning! Main action node gets removed!')
                    return None, None, None

                incoming_edges = list(manipulated_amr.in_edges(nbunch=meeting_node, data=True))
                outgoing_edges = list(manipulated_amr.out_edges(nbunch=meeting_node, data=True))
                removed_nodes[meeting_node] = incoming_edges + outgoing_edges

                manipulated_amr.remove_node(meeting_node)
                undirected_manipulated_amr.remove_node(meeting_node)

                # break the loop in order to recompute the paths again for the new resulting graph
                graph_changed = True
                break

        if graph_changed:
            continue

        else:           # this means no meeting node was found for any connecting path
            return None, None, None

    return manipulated_amr, undirected_manipulated_amr, removed_nodes


def find_meeting_node(amr_graph, path):
    """

    :param amr_graph:
    :param path:
    :return:
    """

    path_triples = get_triples_single_path(amr_graph, path)

    path_edges = [trip[1] for trip in path_triples]
    direction_changes = find_direction_changes(path_edges)

    # TODO: decide how to deal with these cases
    if len(direction_changes) != 1:
        return None

    relevant_position = direction_changes[0]
    relevant_edge1 = path[relevant_position[0]]
    relevant_edge2 = path[relevant_position[1]]

    assert relevant_edge1[1] == relevant_edge2[0]

    return relevant_edge1[1]
