import networkx as nx
from prelim_inspection_amr_action_alignments import get_one2many_amrs_and_alignments, get_graph_pairs
from penman_networkx_conversions import networkx2penman
from collections import defaultdict, Counter
import penman
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


def compute_amr_node_pairs(amr_graph: nx.Graph,
                           aligned_nodes: List,
                           ignore: bool,
                           shortest: bool):
    """
    Computes pairs of those nodes of the input AMR graph that are in the aligned_nodes list
    in the following way
    if ignore: ignore all AMR nodes with label 'you' or 'imperative'
    if shortest: pair each amr node from aligned_nodes only with the closest other node from the list
                else pair each node with each other node but ignore ordering (e.g. only include (1,2) or (2,1))
    :param amr_graph:
    :param aligned_nodes:
    :param ignore:
    :param shortest:
    :return:
    """

    relevant_amr_nodes = []
    # if ignore=True: exclude 'you' and 'imperative' nodes which are often aligned to an action node
    for amr_node in aligned_nodes:
        if ignore:
            label = nx.get_node_attributes(amr_graph, 'label')[amr_node]
            if label != 'you' and label != 'imperative':
                relevant_amr_nodes.append(amr_node)
        else:
            relevant_amr_nodes.append(amr_node)

    if shortest:
        amr_node_pairs = pair_closest_nodes(amr_graph, relevant_amr_nodes)
    else:
        amr_node_pairs = pair_all_nodes(relevant_amr_nodes)

    return amr_node_pairs


def pair_all_nodes(node_list: List) -> List[Tuple]:
    """
    Create all pairings of the nodes in node_list ignoring ordering,
    e.g. if node_list = [1,2,3] then node_pairs = [(1,2), (1,3), (2,3)]
    :param node_list: list of nodes
    :return: list of node pairs
    """
    node_pairs = []
    for i, node in enumerate(node_list):
        i2 = i + 1
        while i2 < len(node_list):
            node2 = node_list[i2]
            node_pairs.append((node, node2))
            i2 += 1
    return node_pairs


def pair_closest_nodes(graph: nx.Graph, node_list: List):
    """
    For each node in the node_list find the closest other node in the graph from node_list
    :param graph: a network x graph
    :param node_list: a list of nodes from the graph
    :return: list of pairs of the nodes from node_list such that each node is paired with the
            closest other node
    """

    undirected_graph = graph.to_undirected()
    node_pairs = []
    for i, node in enumerate(node_list):
        current_shortest_distance = 80      # just any high number
        current_closest_node = ""
        for i2, node2 in enumerate(node_list):
            if i == i2:
                continue
            shortest_paths = find_shortest_path(undirected_graph, node, node2)
            shortest_len = len(shortest_paths[0])
            if shortest_len < current_shortest_distance:
                current_shortest_distance = shortest_len
                current_closest_node = node2
        if current_closest_node != "":
            node_pairs.append((node, current_closest_node))

    # remove duplicates
    unique_pairs = set(node_pairs)
    unique_pairs_unordered = []
    for pair in unique_pairs:
        if (pair[1], pair[0]) not in unique_pairs_unordered:
            unique_pairs_unordered.append(pair)

    return unique_pairs_unordered


def get_labelled_paths(amr_graph: nx.Graph, path_var_dict) -> Dict:
    """
    Converts paths where edges are represented by a list of node pairs into paths where edges are represented by
    a list of the edge labels
    e.g. original path: [(and, sprinkle-01), (sprinkle-01, you)]
         created path: [('op1', 'ARG1)]
    :param amr_graph: an networkx AMR graph
    :param path_var_dict: the dictionary with paths (value) between node pairs (key) in the AMR
                          each value is a list of paths where each path is a list of edges as node pairs
    :return: dictionary with the same paths for all node pairs in the input dict but edges are represented
            by their label instead of the pair of nodes connected by it
    """
    labelled_paths = defaultdict(list)

    for node_pair, all_paths in path_var_dict.items():

        for path_edges in all_paths:
            labelled_edges = []
            for var_edge in path_edges:
                attributes = amr_graph.get_edge_data(var_edge[0], var_edge[1])  # get the label for the edge
                if attributes:
                    labelled_edges.append(attributes['label'])
                # if there does not exist the edge from the path in the directed AMR, check the opposite edge direction
                # and add '-of' to the label
                else:
                    attributes = amr_graph.get_edge_data(var_edge[1], var_edge[0])
                    new_label = attributes['label'] + '-of'
                    labelled_edges.append(new_label)
            labelled_paths[node_pair].append(labelled_edges)

    return dict(labelled_paths)


def extract_relevant_paths(action_graph_dir, amr_graph_dir) -> (List[Dict], List[Dict]):
    """
    For all action graphs and corresponding sentence level AMRs
        Finds all AMRs that are aligned to more than one action node
        Extracts the paths between the nodes in one AMR that are aligned to different action nodes
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return: two lists of the dictionaries of all shortest paths between all action-node-aligned AMR nodes
             dicts in first list:
                            keys: all pairings of AMR nodes that have an alignment to an action node
                            values: list of paths, where paths are edges -> list of tuples of the nodes ,
                                    e.g. path from node1 to node3 could be: [(node1, node2), (node2, node3)]
             dicts in second list:
                            keys: all pairings of AMR nodes that have an alignment to an action node
                            values: list of paths, where paths are the edge labels,
                                    e.g. [edge_label1, edge_label2]
                                    note that all edges where the label ends with '-of' go in the same direction
                                    and all edges without '-of' suffix go in the same (opposite) direction
    """
    all_paths_vars = []
    all_paths_labels = []

    # all action graphs - AMR list pairs for all recipes
    amr_ac_graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)

    # extract all AMRs that are aligned to more than one action node together with information
    # about the aligned nodes
    for recipe_name in amr_ac_graph_pairs.keys():

        one2many_current_recipe = get_one2many_amrs_and_alignments(amr_ac_graph_pairs[recipe_name]['action'],
                                                                   amr_ac_graph_pairs[recipe_name]['amrs'])

        for (nx_amr, alignments) in one2many_current_recipe:
            relevant_paths_vars = get_paths_between_actions(nx_amr, alignments)
            relevant_paths_roles = get_labelled_paths(nx_amr, relevant_paths_vars)

            all_paths_vars.append(relevant_paths_vars)
            all_paths_labels.append(relevant_paths_roles)

    return all_paths_vars, all_paths_labels


def inspect_path_patterns(action_graph_dir, amr_graph_dir):
    """

    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return:
    """
    relevant_paths_vars, relevant_paths_labels = extract_relevant_paths(action_graph_dir, amr_graph_dir)

    all_path_patterns = list()

    for amr_paths in relevant_paths_labels:
        for node_pair, node_path in amr_paths.items():
            all_path_patterns.extend(node_path)

    path_patterns_num_removed = [remove_role_numbering_paths(pp) for pp in all_path_patterns]

    return path_patterns_num_removed


def remove_role_numbering_paths(path: List[str]) -> List[str]:
    """
    Removes the enumeration from the roles on the input path, e.g. ARG0, ARG1, ARG2 all become ARG and opX becomes op
    :param path: list of edge labels (i.e. AMR roles)
    :return: list of the same edge labels with X removed from ARGX, ARGX-of and opX and opX-of
    """
    cleaned_path = []
    for edge_label in path:
        if edge_label.startswith('op'):
            if edge_label.endswith('-of'):
                cleaned_path.append('op-of')
            else:
                cleaned_path.append('op')

        elif edge_label.startswith('ARG'):
            if edge_label.endswith('-of'):
                cleaned_path.append('ARG-of')
            else:
                cleaned_path.append('ARG')
        else:
            cleaned_path.append(edge_label)

    return cleaned_path


def count_direction_changes(path: List[str]) -> int:
    """
    Count how often the edges on the input path change their direction
    :param path: list of edge labels such that
                 all edges where the label ends with '-of' go in the same direction
                 and all edges without '-of' suffix go in the same (opposite) direction
    :return:
    """
    changes = 0
    prev_dir = 'backward' if path[0].endswith('-of') else 'forward'
    for edge in path:
        current_dir = 'backward' if edge.endswith('-of') else 'forward'
        if current_dir != prev_dir:
            changes += 1
        prev_dir = current_dir

    return changes


def create_path_pattern_overview(action_graph_dir, amr_graph_dir, output_file):
    """
    Extract all path patterns between the closest amr nodes in an amr graph aligned to action
    nodes and write all paths and their counts into output_file
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :param output_file:
    :return:
    """
    path_patterns = inspect_path_patterns(action_graph_dir, amr_graph_dir)
    path_pattern_tuples = [tuple(pp) for pp in path_patterns]   # cannot apply counter for lists -> convert to tuples
    pattern_counts = Counter(path_pattern_tuples)
    overall_pattern_direction_changes = []

    paths_per_changes = defaultdict(int)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for pattern, count in pattern_counts.items():
            direction_changes = count_direction_changes(list(pattern))
            overall_pattern_direction_changes.append(direction_changes)
            out_file.write(f'{pattern}\t{count}\t{direction_changes}\n')

            paths_per_changes[direction_changes] += count

    count_changes_total = Counter(overall_pattern_direction_changes)
    print(count_changes_total)
    print(paths_per_changes)


def extract_amrs_with_direction_changes(action_graph_dir, amr_graph_dir, output_file):
    """
    Extract all amr graphs where the shortest path between any two closest action-aligned node pairs
    changes the direction more than one time and write the amrs into the output_file
    :param action_graph_dir:
    :param amr_graph_dir:
    :param output_file:
    :return:
    """
    changing_amrs = []

    # all action graphs - AMR list pairs for all recipes
    amr_ac_graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)

    # extract all AMRs that are aligned to more than one action node together with information
    # about the aligned nodes
    for recipe_name in amr_ac_graph_pairs.keys():

        one2many_current_recipe = get_one2many_amrs_and_alignments(amr_ac_graph_pairs[recipe_name]['action'],
                                                                   amr_ac_graph_pairs[recipe_name]['amrs'])

        for (nx_amr, alignments) in one2many_current_recipe:
            relevant_paths_vars = get_paths_between_actions(nx_amr, alignments)
            relevant_paths_roles = get_labelled_paths(nx_amr, relevant_paths_vars)

            change_larger_one = False
            for possible_paths in relevant_paths_roles.values():
                for path in possible_paths:
                    direction_change = count_direction_changes(path)
                    if direction_change > 1:
                        change_larger_one = True

            if change_larger_one:
                aligned_amr_nodes = [al[0] for al in alignments]
                changing_amrs.append((nx_amr, aligned_amr_nodes))

    with open(output_file, 'w', encoding='utf-8') as out:

        for (amr, aligns) in changing_amrs:
            pen_amr = networkx2penman(amr)
            pen_amr.metadata['alignments'] = ', '.join(aligns)
            pen_str = penman.encode(pen_amr)
            out.write(f'{pen_str}\n\n')


if __name__=="__main__":

    create_path_pattern_overview('../../Corpora/Mapped_Ara/new_ara_data_new_action_graphs',
                                 './aligned_recipe_amrs_ibm',
                                 './relevant_path_patterns_ignore_shortest.csv')
    


    extract_amrs_with_direction_changes('../../Corpora/Mapped_Ara/new_ara_data_new_action_graphs',
                                 './aligned_recipe_amrs_ibm',
                                 './amrs_direction_changes_larger_one.txt')

