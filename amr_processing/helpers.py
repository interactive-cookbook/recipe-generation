import networkx as nx
from typing import List, Dict, Tuple
import re
import penman
from penman import surface


def count_aligned_actions(sentence_amr: nx.Graph, action_nodes: list) -> int:
    """
    Count the number of action nodes to which the input AMR has alignments
    :param sentence_amr:
    :param action_nodes:
    :return:
    """
    aligned_actions = set()

    amr_nodes = sentence_amr.nodes
    for amr_node in amr_nodes:
        token_id = nx.get_node_attributes(sentence_amr, 'alignment')[amr_node]
        if token_id in action_nodes:
            aligned_actions.add(token_id)

    return len(aligned_actions)


def remove_role_numbering_paths(path: List[str]) -> List[str]:
    """
    Removes the enumeration from the roles on the input path, e.g. ARG0, ARG1, ARG2 all become ARG and opX becomes op
    :param path: list of edge labels (i.e. AMR roles)
    :return: list of the same edge labels with X removed from ARGX, ARGX-of and opX and opX-of
    """
    cleaned_path = []
    for edge_label in path:
        cleaned_edge_label = remove_role_numbering_edge(edge_label)
        cleaned_path.append(cleaned_edge_label)

    return cleaned_path


def remove_role_numbering_edge(edge_label: str) -> str:
    """
    Removes the enumeration from the edge label, i.e. ARG0, ARG1, ARG2 all become ARG and opX becomes op
    :param edge_label:
    :return:
    """
    cleaned_label = edge_label

    if edge_label.startswith('op'):
        if edge_label.endswith('-of'):
            cleaned_label = 'op-of'
        else:
            cleaned_label = 'op'

    elif edge_label.startswith('ARG'):
        if edge_label.endswith('-of'):
            cleaned_label = 'ARG-of'
        else:
            cleaned_label = 'ARG'

    return cleaned_label


def find_direction_changes(path: List[str]) -> List[List]:
    """
    Find the indices of all edge pairs on the input path where the direction of the edges changes
    e.g if path is ["op", "op-of", "ARG-of"] then the only direction change is at (0,1)
    Special case: "consist" and "part" edges are treated as edges in the opposite direction
    :param path: list of edge labels such that
                 all edges where the label ends with '-of' go in the same direction
                 and all edges without '-of' suffix go in the same (opposite) direction
    :return: list of pairs for the direction change positions
    """
    changes = []
    special_labels = ['consist', 'part', 'consist-of', 'part-of']

    if path[0] in special_labels:
        prev_dir = 'forward' if path[0].endswith('-of') else 'backward'
    else:
        prev_dir = 'backward' if path[0].endswith('-of') else 'forward'

    for i, edge in enumerate(path):
        if edge in special_labels:
            current_dir = 'forward' if edge.endswith('-of') else 'backward'
        else:
            current_dir = 'backward' if edge.endswith('-of') else 'forward'
        if current_dir != prev_dir:
            changes.append([i-1, i])
        prev_dir = current_dir

    return changes


def count_direction_changes(path: List[str]) -> int:
    """
    Count how often the edges on the input path change their direction
    Special case: "consist" and "part" edges are treated as edges in the opposite direction
    :param path: list of edge labels such that
                 all edges where the label ends with '-of' go in the same direction
                 and all edges without '-of' suffix go in the same (opposite) direction
    :return:
    """
    changing_positions = find_direction_changes(path)
    return len(changing_positions)


def includes_node_from_list(subgraph: nx.graph, node_list: list) -> bool:
    """
    Checks whether the input (sub)graph includes at least one node from the input list
    :param subgraph: networkX (sub) graph
    :param node_list: list of nodes to compare graph nodes to
    :return: True if at least one of the nodes in node_list is a node of subgraph
    """
    graph_nodes = nx.nodes(subgraph)
    for ac_node in node_list:
        if ac_node in graph_nodes:
            return True
    return False


def includes_all_nodes_from_list(subgraph: nx.graph, node_list: list) -> bool:
    """
    Checks whether the input (sub)graph includes all nodes from the input list
    :param subgraph: networkX (sub) graph
    :param node_list: list of nodes to compare graph nodes to
    :return: True if all nodes in node_list is a node of subgraph
    """
    graph_nodes = nx.nodes(subgraph)
    for node in node_list:
        if node not in graph_nodes:
            return False
    return True


def find_highest_node(node_list: list, graph: nx.Graph):
    """
    Determine which of the nodes in node_list is the 'highest' node in the input graph,
    e.g. if there is a path from node1 to node2 in the graph, then node1 is higher
         if there is a path from node2 to node1, then node2 is higher
    Comparisons start by treating the first element of node_list as the highest until a higher
    one is found
    -> if there is no path connecting two nodes nothing changes; but should in practice not happen
    for the use cases of this function
    :param node_list: list of action nodes
    :param graph: action graph
    :return: the name of the highest node
    """
    current_highest_node = node_list[0]
    for node in node_list:
        if node != current_highest_node:
            # Make it a list! Otherwise nx.all_simple_paths returns a generator that is always(!) True
            p = list(nx.all_simple_paths(graph, node, current_highest_node))
            if p:
                current_highest_node = node

    return current_highest_node


def find_new_root(node_list: list, graph: nx.Graph):
    """
    Find the node that is an appropriate new root for the AMR
    the new root node should be either from node_list or the lowest common
    ancestor of two nodes
    Method does no work perfect if len(node_list) is larger than 2 because
    only pairwise comparisons, but this case never occurred in ARA 1 and ARA 2
    :param node_list:
    :param graph:
    :return:
    """

    current_new_root = node_list[0]
    for node in node_list[1:]:
        n1 = current_new_root
        n2 = node
        g = graph.copy()
        common_anc = nx.lowest_common_ancestor(g, n1, n2)
        if common_anc:
            current_new_root = common_anc

    return current_new_root


def cluster_dict2list(action_clusters: List[Dict]) -> Tuple[List[List], List]:
    """
    Extract only the main action aligned AMR nodes for all action clusters
    :param action_clusters: list of dictionaries;
             one dictionary per action node cluster covered by a specific AMR
             one key-value pair per action node of the cluster
             key: ID of the action node
             value: list of main corresponding AMR node(s)
    :return: (amr_node_clusters, main_action_amr_nodes)
            amr_node_clusters: a list with one sublist per cluster; containing all AMR nodes of that cluster
            main_action_amr_nodes: a flattened version of amr_node_clusters, i.e. list of all AMR nodes of the cluster
    """
    main_action_amr_nodes = []
    amr_node_clusters = []

    for cluster in action_clusters:
        corresponding_amr_nodes = cluster.values()
        flattened_amr_nodes = []
        for amr_list in corresponding_amr_nodes:
            flattened_amr_nodes.extend(amr_list)
        amr_node_clusters.append(flattened_amr_nodes)
        main_action_amr_nodes.extend(flattened_amr_nodes)

    return amr_node_clusters, main_action_amr_nodes


def post_process_imperative(amr_graph: nx.Graph):
    """
    Post processing step to correct amrs where the parser missed an imperative
    An attribute :mode imperative and a new node and edge :ARG0 you are added to all
    nodes that fulfill the following conditions:
    1a. is the root node, is a predicate node (a frame) and is aligned to an action node
    1b. or is a direct child of the root node connected with an :opX edge, is a predicate node (a frame)
        and is aligned to an action node
    2. the node does not have an :ARG0 argument and does not have an imperative argument
       (if the imperative argument is there but no ARG0 argument, then only the ARG0 gets added)
    Both, the imperative attributes as well as the ARG0 you nodes get the predicate token as the
    aligned token
    :param amr_graph: an action-level AMR graph
    :return: the modified action-level AMR graph
    """
    nodes_to_extend = []

    root_node = amr_graph.graph['root']
    root_label = nx.get_node_attributes(amr_graph, 'label')[root_node]
    action_nodes = amr_graph.graph['alignments']
    predicate_reg = r'[a-zA-Z]+-[0-9]+$'

    if re.search(predicate_reg, root_label) and root_node in action_nodes:
        nodes_to_extend.append(root_node)
    elif root_label in ['and', 'or']:
        for edge in amr_graph.edges(root_node):
            edge_label = nx.get_edge_attributes(amr_graph, 'label')[edge]
            child_node = edge[1]
            child_label = nx.get_node_attributes(amr_graph, 'label')[child_node]
            if edge_label.startswith('op') and child_node in action_nodes and re.search(predicate_reg, child_label):
                nodes_to_extend.append(child_node)

    for node in nodes_to_extend:
        has_subj = False
        has_imp = False
        # check whether predicate has an imperative attribute
        try:
            attr_data = nx.get_node_attributes(amr_graph, 'attr')[node]
            for a in attr_data:
                role_lab = a['label']
                attr_lab = a['target']
                if role_lab == 'mode' and attr_lab == 'imperative':
                    has_imp = True
        except KeyError:
            pass

        # check whether predicate has an ARG0 arguments
        child_edges = amr_graph.edges(node)
        for ce in child_edges:
            edge_label = nx.get_edge_attributes(amr_graph, 'label')[ce]
            if edge_label == 'ARG0':
                has_subj = True

        if not has_subj:
            # create unique variable name
            node_var = 'y'
            num = 1
            while node_var in amr_graph.nodes():
                node_var += str(num)
                num += 1
            # add the new 'you' ARG0 node with appropriate alignment information
            aligned_token = nx.get_node_attributes(amr_graph, 'alignment')[node]
            int_alignment = int(aligned_token)
            new_pen_alignment = penman.surface.Alignment((int_alignment,), prefix='e.')
            amr_graph.add_nodes_from([(node_var, {'label': 'you', 'type': 'instance', 'epi': [new_pen_alignment],
                                       'alignment': aligned_token})])
            amr_graph.add_edge(node, node_var, label='ARG0', epi=[])

            # if also no imperative marker add it
            if not has_imp:
                predicate_data = amr_graph.nodes(data=True)[node]
                try:
                    predicate_data['attr'].append({'source': node, 'label': 'mode', 'target': 'imperative',
                                                   'epi': [new_pen_alignment], 'alignment': aligned_token})
                except KeyError:
                    predicate_data['attr'] = [{'source': node, 'label': 'mode', 'target': 'imperative',
                                                   'epi': [new_pen_alignment], 'alignment': aligned_token}]

    return amr_graph
