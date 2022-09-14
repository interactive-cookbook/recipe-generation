import random
import networkx as nx
from collections import defaultdict
import re
from ..paths_between_actions import pair_closest_clustered_nodes, find_shortest_path, get_path_triples
from helpers import remove_role_numbering_edge


def cluster_action_aligned_amr_nodes(sentence_amr: nx.Graph, all_action_nodes: list):
    """

    :param sentence_amr:
    :param all_action_nodes:
    :return:
    """

    alignments = defaultdict(list)

    for amr_node in list(sentence_amr.nodes):
        token_id = nx.get_node_attributes(sentence_amr, 'alignment')[amr_node]
        if token_id in all_action_nodes:
            alignments[token_id].append(amr_node)

    main_amr_node_per_action = get_main_amr_node_per_action(alignments, sentence_amr)
    amr2action_node = get_amr2action_dict(main_amr_node_per_action)

    # do all pairings and decide which action nodes actually constitute only one single action
    relevant_amr_nodes = list(main_amr_node_per_action.values())
    # TODO: not only closest nodes
    amr_node_pairs = pair_closest_clustered_nodes(sentence_amr, relevant_amr_nodes)

    pairs_to_keep_together = []

    for node_pair in amr_node_pairs:
        separate = check_split_condition(sentence_amr, node_pair[0], node_pair[1])
        if not separate:
            action1 = amr2action_node[node_pair[0]]
            action2 = amr2action_node[node_pair[1]]
            pairs_to_keep_together.append((action1, action2))

    # TODO continue with function




def get_main_amr_node_per_action(action_amr_alignments: dict, amr_graph: nx.Graph):
    """

    :param action_amr_alignments:
    :param amr_graph:
    :return:
    """
    main_amr_nodes = dict()

    for action_node, aligned_amr_nodes in action_amr_alignments.items():
        if len(aligned_amr_nodes) == 1:
            main_amr_nodes[action_node] = [aligned_amr_nodes[0]]
        else:
            candidates = []
            for node in aligned_amr_nodes:
                label = nx.get_node_attributes(amr_graph, 'label')[node]
                if label != 'you' and label != 'imperative':
                    candidates.append(node)
            if len(candidates) == 1:
                main_amr_nodes[action_node] = [candidates[0]]
            elif len(candidates) == 0:
                # should never happen if parsed correctly but occurred for one instruction homemade_pizza_dough_8_instr5
                # where the verb was not represented in the AMR but aligned to 'you' and 'imperative' only
                # so just choose one of them
                main_amr_nodes[action_node] = [aligned_amr_nodes[0]]

            else:
                # consider predicate nodes first
                predicate_reg = r'[a-zA-Z]*-[0-9]*$'
                predicate_candidates = []
                for cand in candidates:
                    label = nx.get_node_attributes(amr_graph, 'label')[cand]
                    if re.search(predicate_reg, label):
                        predicate_candidates.append(cand)

                # if exactly one predicate take the predicate
                if len(predicate_candidates) == 1:
                    main_amr_nodes[action_node] = [predicate_candidates[0]]
                # else keep all of them to find the appropriate ones for the paths later on
                else:
                    main_amr_nodes[action_node] = candidates

    return main_amr_nodes


def get_amr2action_dict(action2amr: dict):
    """

    :param action2amr:
    :return:
    """
    amr2action = dict()
    for action_node, amr_nodes in action2amr.items():
        for node in amr_nodes:
            amr2action[node] = action_node
    return amr2action


def check_split_condition(amr_graph: nx.Graph, node1, node2) -> bool:

    connecting_paths = find_shortest_path(amr_graph, node1, node2)
    labelled_path_triples = get_path_triples(amr_graph, connecting_paths)
    node1_lab = nx.get_node_attributes(amr_graph, 'label')[node1]
    node2_lab = nx.get_node_attributes(amr_graph, 'label')[node2]

    to_split = True     # whether to split the two action nodes

    # go through conditions
    for ind, path in enumerate(labelled_path_triples):

        path_edges = [triple[1] for triple in path]
        path_nodes = [triple[2] for triple in path]
        path_nodes.append(node2_lab)

        # fix actions that should be one action but were tagged as two because of intermediate tokens
        check_fixing_tagger = conditions_fixing_tagger(path)
        if check_fixing_tagger:
            to_split = False
            break

        # keep two action together if the path between consists only of a time, purpose, manner, duration, or
        # instrument edge and an arbitrary number of 'opX' edges;
        # except if time relation is "before" or "after" then split
        edges_of_interest_forward = ['time', 'purpose', 'manner', 'duration', 'instrument']
        edges_of_interest_backward = ['time-of', 'purpose-of', 'manner-of', 'duration-of', 'instrument-of']

        if path_edges[0] in edges_of_interest_forward:
            to_split = check_path_of_interest(path_edges=path_edges, path_nodes=path_nodes, forward=True)
            if not to_split:
                break

        if path_edges[-1] in edges_of_interest_backward:
            to_split = check_path_of_interest(path_edges=path_edges, path_nodes=path_nodes, forward=False)
            if not to_split:
                break

        # other cases
        if len(path) == 2:
            intermediate_node = path_nodes[1]
            edge1 = path_edges[0]
            edge2 = path_edges[1]
            edge1_cleaned = remove_role_numbering_edge(edge1)
            edge2_cleaned = remove_role_numbering_edge(edge2)

            # keep disjunctions together
            if (edge1_cleaned == 'op' and edge2_cleaned == 'op-of') or (edge1_cleaned == 'op-of' and edge2_cleaned == 'op'):
                if intermediate_node == 'or':
                    to_split = False
                    break

            # keep contrast together
            if (edge1_cleaned == 'ARG' and edge2_cleaned == 'ARG-of') or (edge1_cleaned == 'ARG-of' and edge2_cleaned == 'ARG'):
                if intermediate_node == 'contrast-01':
                    to_split = False
                    break

    return to_split


def conditions_fixing_tagger(labelled_path) -> bool:
    """
    Fixes some issues caused by the recipe tagger by returning True if the two input nodes are affected by such an issue
    Issues here are cases were tagger tags two tokens as separate actions although they should be treated as a single
    action:
        let-01 :ARG1 [Action]
        allow-01 :ARG1 [Action]
        start-01 :ARG1 [Action]
        give-01 :ARG1 stir-01
        bring-01 :ARG2 boil-01
        use-01 :ARG2 [Action]
        [Action] :direction [token]     (e.g. 'roll the dough out' where 'out' is tagged as separate action)
    :param labelled_path: list of triples for the path connecting two nodes
                            e.g. for connecting node1 and node3:
                                [(node1, edge1, node2), (node2, edge2, node3)]
    :return:
    """
    if len(labelled_path) > 1:
        return False
    node1_lab = labelled_path[0][0]
    edge_label = labelled_path[0][1]
    node2_lab = labelled_path[0][2]
    keep_together = False
    if (node1_lab == 'let-01' and edge_label == 'ARG1') or (node2_lab == 'let-01' and edge_label == 'ARG1-of'):
        keep_together = True
    elif (node1_lab == 'allow-01' and edge_label == 'ARG1') or (node2_lab == 'allow-01' and edge_label == 'ARG1-of'):
        keep_together = True
    elif (node1_lab == 'start-01' and edge_label == 'ARG1') or (node2_lab == 'start-01' and edge_label == 'ARG1-of'):
        keep_together = True
    elif (node1_lab == 'give-01' and edge_label == 'ARG1' and node2_lab == 'stir-01') or (node2_lab == 'give-01' and edge_label == 'ARG1-of' and node1_lab == 'stir-01'):
        keep_together = True
    elif (node1_lab == 'bring-01' and edge_label == 'ARG2' and node2_lab == 'boil-01') or (node2_lab == 'bring-01' and edge_label == 'ARG2-of' and node1_lab == 'boil-01'):
        keep_together = True
    elif (node1_lab == 'use-01' and edge_label == 'ARG2') or (node2_lab == 'use-01' and edge_label == 'ARG2-of'):
        keep_together = True
    elif edge_label == 'direction' or edge_label == 'direction-of':
        keep_together = True

    return keep_together


def check_path_of_interest(path_edges, path_nodes, forward) -> bool:
    """

    :param path_edges:
    :param path_nodes:
    :param forward:
    :return:
    """
    if len(path_edges) == 1:
        return False

    if 'before' in path_nodes or 'after' in path_nodes:     # split actions if time relation is before or after
        return True

    unique_edges_types = set()
    for edge_label in path_edges:
        unique_edges_types.add(remove_role_numbering_edge(edge_label))

    if len(unique_edges_types) == 2:      # should only include the edge of interest and "opX" edges, all in the same direction
        if forward and ('op' in unique_edges_types):
            return False
        if (not forward) and ('op-of' in unique_edges_types):
            return False

    return True
