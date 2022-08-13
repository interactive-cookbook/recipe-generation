import networkx as nx
from prelim_inspection_amr_action_alignments import get_one2many_amrs_and_alignments, get_graph_pairs
from penman_networkx_conversions import penman2networkx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


# TODO documentation
# TODO Adapt Code for finding and anaylyzing path patterns
def get_paths_between_actions(amr_graph: nx.Graph, aligned_nodes, all_paths=False) -> Dict[Tuple: List[List[Tuple]]]:
    """
    For each pair of nodes in a sentence-level AMR that are aligned to an action node
    Find all paths / all paths of shortes length between the two nodes
    :param amr_graph: one sentence-level networX.Graph AMR graph that has alignments to several action nodes
    :param aligned_nodes: list of the pairs of nodes of the AMR and aligned action nodes
    :param all_paths: if set to True then all paths between two nodes get extracted, otherwise only the shortest paths
    :return: dictionary
            keys: all pairings of AMR nodes that have an alignment to an action node
            values: list of all (shortest) paths between the nodes,
                    paths are edges -> list of tuples of the nodes ,
                    e.g. path from node1 to node3 could be: [(node1, node2), (node2, node3)]
    """
    path_dict = dict()

    relevant_amr_nodes = [pair[0] for pair in aligned_nodes]    # the AMR nodes
    amr_node_pairs = []
    for i, node in enumerate(relevant_amr_nodes):
        i2 = i + 1
        while i2 < len(relevant_amr_nodes):
            node2 = relevant_amr_nodes[i2]
            amr_node_pairs.append((node, node2))
            i2 += 1

    # need to ignore directionality in order to find paths
    non_dir_graph = amr_graph.to_undirected()

    for (node, node2) in amr_node_pairs:

        paths = nx.all_simple_edge_paths(G=non_dir_graph, source=node, target=node2)
        paths = list(paths)
        if all_paths:
            path_dict[(node, node2)] = paths
        else:
            current_min_len = 80    # just any large number
            for p in paths:
                current_len = len(p)
                if current_len < current_min_len:
                    current_min_len = current_len

            path_dict[(node, node2)] = [p for p in paths if len(p) == current_min_len]

    return path_dict


def get_labelled_paths(amr_graph: nx.Graph, path_var_dict) -> Dict:
    """

    :param amr_graph:
    :param path_var_dict:
    :return:
    """
    labelled_paths = defaultdict(list)

    for node_pair, all_paths in path_var_dict.items():

        for path_edges in all_paths:
            labelled_edges = []
            for var_edge in path_edges:
                attributes = amr_graph.get_edge_data(var_edge[0], var_edge[1])
                if attributes:
                    labelled_edges.append(attributes['label'])
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
    :return: list of
             list of
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

        for (penman_amr, alignments) in one2many_current_recipe:
            nx_amr = penman2networkx(penman_amr)
            relevant_paths_vars = get_paths_between_actions(nx_amr, alignments)
            relevant_paths_roles = get_labelled_paths(nx_amr, relevant_paths_vars)

            all_paths_vars.append(relevant_paths_vars)
            all_paths_labels.append(relevant_paths_roles)

    return all_paths_vars, all_paths_labels


def inspect_paths_between_nodes(action_graph_dir, amr_graph_dir):
    """

    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :return:
    """
    relevant_paths_vars, relevant_paths_labels = extract_relevant_paths(action_graph_dir, amr_graph_dir)

    all_path_patterns = list()

    for amr_paths in relevant_paths_labels:
        for node_pair, node_path in amr_paths.items():
            all_path_patterns.append(node_path)

    #analyze_paths(all_path_patterns)

    print_and_write(relevant_paths_labels, all_path_patterns)


def analyze_paths(path_pattern_list):
    """

    :param path_pattern_list:
    :return:
    """
    pattern_counts_node_level = defaultdict(int)
    pattern_counts_total = defaultdict(int)
    pattern_counts_node_level_num_removed = defaultdict(int)
    pattern_counts_total_num_removed = defaultdict(int)

    for node_pair_paths in path_pattern_list:
        path_dict_key = []
        path_dict_key_num_removed = []
        for possible_path in node_pair_paths:
            path_dict_key.append(tuple(possible_path))

            # remove the 0 from arg0, 1 from op1 etc.
            new_possible_path = []
            for edge in possible_path:
                if edge[:3] == ':op':
                    if edge[-3:] == '-of':
                        new_possible_path.append(':op-of')
                    else:
                        new_possible_path.append(':op')
                elif edge[:4] == ':arg':
                    if edge[-3:] == '-of':
                        new_possible_path.append('arg-of')
                    else:
                        new_possible_path.append('arg')
                else:
                    new_possible_path.append(edge)
            path_dict_key_num_removed.append(tuple(new_possible_path))

        pattern_counts_node_level[tuple(path_dict_key)] += 1
        pattern_counts_node_level_num_removed[tuple(path_dict_key_num_removed)] += 1

        for possible_path in node_pair_paths:
            pattern_counts_total[tuple(possible_path)] += 1

            # remove the 0 from arg0, 1 from op1 etc.
            new_possible_path = []
            for edge in possible_path:
                if edge[:3] == ':op':
                    if edge[-3:] == '-of':
                        new_possible_path.append(':op-of')
                    else:
                        new_possible_path.append(':op')
                elif edge[:4] == ':arg':
                    if edge[-3:] == '-of':
                        new_possible_path.append('arg-of')
                    else:
                        new_possible_path.append('arg')
                else:
                    new_possible_path.append(edge)
            pattern_counts_total_num_removed[tuple(new_possible_path)] += 1

    with open('./pattern_counts_node_level.tsv', 'w', encoding='utf-8') as file:
        for pattern, count in pattern_counts_node_level.items():
            file.write(f'{pattern}\t{count}\n')

    with open('./pattern_counts_total.tsv', 'w', encoding='utf-8') as file:
        for pattern, count in pattern_counts_total.items():
            file.write(f'{pattern}\t{count}\n')

    with open('./pattern_counts_node_level_numeration_removed.tsv', 'w', encoding='utf-8') as file:
        for pattern, count in pattern_counts_node_level_num_removed.items():
            file.write(f'{pattern}\t{count}\n')

    with open('./pattern_counts_total_numeration_removed.tsv', 'w', encoding='utf-8') as file:
        for pattern, count in pattern_counts_total_num_removed.items():
            file.write(f'{pattern}\t{count}\n')


def print_and_write(relevant_paths_labels, all_path_patterns):
    """
    Only for getting some counts and the file with the paths
    :param relevant_paths_labels:
    :param all_path_patterns:
    :return:
    """
    print(len(relevant_paths_labels))
    path_lengths_list = []
    path_numbers_list = []
    for path_l in all_path_patterns:
        path_numbers_list.append(len(path_l))
        length_list = [len(path) for path in path_l]
        path_lengths_list.extend(length_list)
    number_of_lengths = Counter(path_lengths_list)
    number_of_paths = Counter(path_numbers_list)
    print(number_of_lengths)
    print(number_of_paths)

    with open('./path_patterns_mininmal_lengths.txt', "w", encoding="utf-8") as file:
        for path in all_path_patterns:
            file.write(f'{path}\n')



if __name__=="__main__":
    """
    pstr1 = "(z1 / and :op1 (z2 / grease-01 :ARG1 (z3 / pan :mod (z4 / loaf))) :op2 (z5 / flour-01 :ARG1 z3))"
    amr = penman.decode(pstr1)
    nx_graph = penman2networkx(amr)
    al_nodes = [('z2', '1'), ('z5', '7')]
    relevant_paths_variables = get_paths_between_actions(nx_graph, al_nodes)
    print(relevant_paths_variables)
    labs = get_labelled_paths(nx_graph, relevant_paths_variables)
    print(labs)
    """

    inspect_paths_between_nodes('../Corpora/Mapped_Ara/new_ara_data_new_action_graphs',
                                './aligned_recipe_amrs_spring')

