from prelim_inspection_amr_action_alignments import get_one2many_amrs_and_alignments, get_graph_pairs
from amr_processing.penman_networkx_conversions import networkx2penman
from paths_between_actions_old import get_paths_between_actions, get_labelled_paths
from collections import defaultdict, Counter
import penman
from typing import List, Dict, Tuple


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


# TODO update documentation
def extract_labelled_paths_and_amrs(action_graph_dir, amr_graph_dir) -> (List[Dict], List[Dict]):
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
    amr_graphs = []
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

            amr_graphs.append(nx_amr)
            all_paths_labels.append(relevant_paths_roles)

    return amr_graphs, all_paths_labels


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


# TODO documentation
def create_path_combinations_overview(action_graph_dir, amr_graph_dir, output_file):
    """
    Extract all path patterns between the closest amr nodes in an amr graph aligned to action
    nodes and write all paths and their counts into output_file
    :param action_graph_dir: parent directory with the action graph files
    :param amr_graph_dir: parent directory with the files with the token-node aligned sentence-level amrs
    :param output_file:
    :return:
    """

    amrs, relevant_paths_labels = extract_labelled_paths_and_amrs(action_graph_dir, amr_graph_dir)

    path_combinations = []
    amrs_per_combination = defaultdict(list)

    for amr_ind, amr_paths in enumerate(relevant_paths_labels):
        for node_pair, node_path in amr_paths.items():
            node_path_num_removed = [remove_role_numbering_paths(pp) for pp in node_path]
            node_path_num_removed.sort()
            tuple_path = [tuple(pp) for pp in node_path_num_removed]
            tuple_path_combination = tuple(tuple_path)
            path_combinations.append(tuple_path_combination)
            amrs_per_combination[tuple_path_combination].append(amrs[amr_ind])

    unique_combinations_list = list(set(path_combinations))
    combination_ids = dict()
    for comb_id, combination in enumerate(unique_combinations_list):
        combination_ids[combination] = comb_id

    combination_counts = Counter(path_combinations)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for pattern, count in combination_counts.items():
            out_file.write(f'{combination_ids[pattern]}\t{pattern}\t{count}\n')

    for combination in combination_ids.keys():
        comb_id = combination_ids[combination]
        amrs = amrs_per_combination[combination]
        with open(f'path_combinations/amrs_combination{comb_id}.txt', 'w', encoding='utf-8') as file:
            for amr in amrs:
                pen_amr = networkx2penman(amr)
                pen_str = penman.encode(pen_amr)
                file.write(f'{pen_str}\n\n')


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

    create_path_combinations_overview('../../Corpora/Ara_Punctuation/new_ara_data_new_action_graphs',
                                 './recipe_amrs_sentences',
                                 './relevant_path_combinations_ignore_shortest.csv')
    

    """
    extract_amrs_with_direction_changes('../../Corpora/Ara_Punctuation/new_ara_data_new_action_graphs',
                                 './recipe_amrs_sentences',
                                 './amrs_direction_changes_larger_one.txt')"""

