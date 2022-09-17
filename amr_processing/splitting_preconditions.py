import networkx as nx
from collections import defaultdict
from typing import Dict, List
import re
from .paths_between_actions import pair_clustered_nodes, find_shortest_path, get_path_triples
from .helpers import remove_role_numbering_edge


# TODO: rename function
def cluster_action_aligned_amr_nodes(sentence_amr: nx.Graph, all_action_nodes: list) -> List[Dict]:
    """
    Cluster all AMR nodes that are aligned to an action node and should not get split because either
    - they belong to the same action node or
    - they are not really separate actions but rather one action with complex structure
    If several AMR nodes are aligned to the same action node then find the main action AMR node, e.g.
    'you' or 'imperative' are often aligned to the action as well but the predicate node should be the
    one considered for splitting and alignments
    :param sentence_amr:
    :param all_action_nodes:
    :return: list of dictionaries; one dictionary per action node cluster covered by the input AMR
             one key-value pair per action node of the cluster
             key: ID of the action node
             value: list of main corresponding AMR node(s)
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
    # TODO: check whether considering all possible pairings leads to expected behavior
    amr_node_pairs = pair_clustered_nodes(relevant_amr_nodes)

    # list of action node pairs that should be kept together
    pairs_to_keep_together = []
    affected_action_nodes = set()

    for node_pair in amr_node_pairs:
        # check the split conditions to identify action node pairs that should not get separated
        separate = check_split_condition(sentence_amr, node_pair[0], node_pair[1])
        if not separate:
            action1 = amr2action_node[node_pair[0]]
            action2 = amr2action_node[node_pair[1]]
            pairs_to_keep_together.append((action1, action2))
            affected_action_nodes.add(action1)
            affected_action_nodes.add(action2)

    action_node_clusters = cluster_node_pairs(pairs_to_keep_together)

    # add all nodes that are not part of a cluster yet as a single-node cluster
    for action_node in alignments.keys():
        if action_node not in affected_action_nodes:
            action_node_clusters.append([action_node])

    # add information about main aligned AMR node for each action node
    extended_clusters = []
    for cluster in action_node_clusters:
        current_extended_cluster = dict()
        for ac_node in cluster:
            current_extended_cluster[ac_node] = main_amr_node_per_action[ac_node]
        extended_clusters.append(current_extended_cluster)

    return extended_clusters



def get_main_amr_node_per_action(action_amr_alignments: Dict[str, List], amr_graph: nx.Graph) -> Dict[str, List]:
    """
    For each action node, choose one main AMR node from all AMR nodes aligned to it in the following way
    - if only one AMR node aligned -> choose that node
    - ignore 'you' and 'imperative' nodes except if they are the only aligned node (due to parsing mistake) then choose randomly
    - if still more than one aligned node left choose
        - the predicate node if exactly one
        - all left aligned nodes otherwise
    :param action_amr_alignments: dictionary with action node - amr node alignments
                                  keys are the action node IDs and values the list of the aligned amr nodes
    :param amr_graph: amr graph
    :return: dictionary with action nodes as keys and a list of the chosen AMR node(s) as value
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
                predicate_reg = r'[a-zA-Z]+-[0-9]+$'
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
                    #print(amr_graph.name)
                    #print(candidates)

    return main_amr_nodes


def get_amr2action_dict(action2amr: Dict[str, List]) -> Dict[str, str]:
    """
    Converts a dictionary to look up an AMR node aligned to a specific action node into a dictionary
    to look up the aligned action node for a specific AMR node
    :param action2amr: a dictionary with action nodes as keys and a list of corresponding AMR nodes as values
    :return: a dictionary with the AMR nodes as keys and the corresponding action node as value
    """
    amr2action = dict()
    for action_node, amr_nodes in action2amr.items():
        for node in amr_nodes:
            amr2action[node] = action_node
    return amr2action


def check_split_condition(amr_graph: nx.Graph, node1, node2) -> bool:
    """
    Checks for two AMR nodes of an AMR graph whether they fulfill one of the defined conditions for
    keeping them together instead of separating them into separate AMRs
    If one condition is fulfilled, False is returned, else True
    :param amr_graph: the AMR graph
    :param node1: one AMR node aligned to an action node (node variable)
    :param node2: another AMR node aligned to an action node (node variable)
    :return:
    """
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
    Check whether the path between two nodes fulfills the conditions for not splitting it
    Pre-condition: the input path needs to fulfill the condition that the path of edges starts with
                    a time, duration, instrument, manner or purpose edge; i.e. the first edge needs to
                    have one of those labels or the last edge needs to have one of those labels + '-of'
    :param path_edges: list of the edge labels of a path between two nodes
    :param path_nodes: list of the node labels of the same path
    :param forward: whether the edges are in their original orientation or reversed (i.e. have the '-of' suffix)
    :return: False if the path should not get split, i.e. fulfills one of the conditions to keep nodes together,
            else True
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


def cluster_node_pairs(node_pairs: List) -> List:

    current_cluster_id = 0
    cluster_dict = dict()

    for (node1, node2) in node_pairs:
        if node1 not in cluster_dict.keys() and node2 not in cluster_dict.keys():
            cluster_dict[node1] = current_cluster_id
            cluster_dict[node2] = current_cluster_id
            current_cluster_id += 1
        elif node1 not in cluster_dict.keys():
            cluster_dict[node1] = cluster_dict[node2]
        elif node2 not in cluster_dict.keys():
            cluster_dict[node2] = cluster_dict[node1]

    number_of_clusters = current_cluster_id
    node_clusters = [[] for cl in range(number_of_clusters)]
    for node, cl_id in cluster_dict.items():
        node_clusters[cl_id].append(node)

    return node_clusters


if __name__=='__main__':

    test_pairs = [(1, 2), (2, 5), (3, 4), (7, 8), (6, 0), (6, 9), (10, 0)]
    clusters = cluster_node_pairs(test_pairs)
    print(clusters)
