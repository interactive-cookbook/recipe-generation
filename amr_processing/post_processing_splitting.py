from typing import List, Dict
import networkx as nx
import penman.surface

from amr_processing.helpers import find_highest_node, remove_role_numbering_edge, find_new_root


def postprocess_split_amrs(separated_amrs: List,
                           original_amr: nx.Graph,
                           action_graph: nx.Graph,
                           action_clusters: List[Dict]):
    """
    Postprocessing of the action-level AMRs obtained by the splitting algorithm.
    Conducts the following steps:
    1. update graph name / instruction ID such that each AMR has a unique name
       e.g. if 'baked_ziti_0_instr0' was separated into 2 action-level graphs,
            the names will be 'baked_ziti_0_instr0_0' and 'baked_ziti_0_instr0_1'
    2. update information about action-aligned amr nodes
    3. remove left-over, now meaningless, root nodes, such as 'and' with only one conjunct after splitting
    4. update the root node if original root node is no longer included in the AMR

    :param separated_amrs: list of all separated action-level AMRs obtained by splitting the original_amr
    :param original_amr: original sentence-level AMR
    :param action_graph: action graph of the corresponding recipe
    :param action_clusters: list of the action-node/action-aligned amr-node clusters
    :return:
    """

    post_sep_amrs = []

    for new_id, sep_amr in enumerate(separated_amrs):
        new_sep_amr = sep_amr.copy()

        new_sep_amr = update_name(sep_graph=new_sep_amr, new_instr_id=new_id)

        new_sep_amr = update_alignments(sep_graph=new_sep_amr, orig_graph=original_amr)

        new_sep_amr = remove_left_over_nodes(sep_graph=new_sep_amr)

        new_sep_amr = update_root_node(sep_graph=new_sep_amr, action_graph=action_graph, action_clusters=action_clusters)

        post_sep_amrs.append(new_sep_amr)

    return post_sep_amrs


def update_name(sep_graph: nx.Graph, new_instr_id: int) -> nx.Graph:
    """
    Update the name of a separated action-level amr graph,
    e.g. if original name was 'baked_ziti_0_instr9' and new_instr_id is 2
    then the new name is 'baked_ziti_0_instr9_2'
    :param sep_graph: separated AMR graph
    :param new_instr_id: new name of the graph / id of the instruction
    :return: the graph with the updated name
    """
    sep_graph.graph['snt_id'] = sep_graph.graph['id']
    new_name = sep_graph.graph['id'] + f'_{new_instr_id}'
    sep_graph.graph['id'] = new_name
    sep_graph.name = new_name
    sep_graph.graph['name'] = new_name

    return sep_graph


def update_alignments(sep_graph: nx.Graph, orig_graph: nx.Graph) -> nx.Graph:
    """
    1. Update the alignment information of a separated action-level amr
    such that it includes all nodes in the action alignment information
    that were aligned nodes in the original AMR and are still part of the
    separated AMR
    2. Update the token alignment of 'you' for predicates in imperative mode if the predicate token to which
    'you' was aligned is no longer present
    :param sep_graph: one of the separated action-level amr
    :param orig_graph: the corresponding original sentence-level amr
    :return: the graph with the updated alignment information
    """
    new_alignments = []
    for orig_al in orig_graph.graph['alignments']:
        if orig_al in sep_graph.nodes:
            new_alignments.append(orig_al)
    sep_graph.graph['alignments'] = new_alignments

    for node in sep_graph.nodes():
        node_data = sep_graph.nodes(data=True)[node]
        aligned_token = node_data['alignment']
        node_label = node_data['label']
        if node_label == 'you':
            incoming_edges = sep_graph.in_edges(node)
            for in_edge in incoming_edges:
                edge_label = nx.get_edge_attributes(sep_graph, 'label')[in_edge]
                if edge_label == 'ARG0':
                    corresponding_predicate_node = in_edge[0]
                    predicate_node_data = sep_graph.nodes(data=True)[corresponding_predicate_node]
                    predicate_token = predicate_node_data['alignment']
                    try:
                        attr_data = predicate_node_data['attr']
                        imperative = False
                        for a in attr_data:
                            if a['target'] == 'imperative':
                                imperative = True
                    except KeyError:
                        # this means the node has no imperative attribute -> not clear whether 'you' was implicit
                        break
                    if aligned_token != predicate_token and imperative:
                        new_aligned_token = predicate_token
                        node_data['alignment'] = new_aligned_token
                        int_alignment = int(new_aligned_token)
                        new_pen_alignment = penman.surface.Alignment((int_alignment,), prefix='e.')
                        node_data['epi'][0] = new_pen_alignment
                        sep_graph.add_nodes_from([(node, node_data)])

    return sep_graph


def remove_left_over_nodes(sep_graph: nx.Graph) -> nx.Graph:
    """
    Remove left-over 'and' and 'multi-sentence' nodes with only one conjunct and 'before' and 'after' nodes that do not
    make sense anymore
    First starts with the root node:
        If the root node of the action-level AMR is labelled 'and', 'before' or 'after' then:
            If the root node has only one child node, then remove the current root node and make the child node the new root
            Elif If the root node has more child nodes and root label is 'and' and exactly one child is connected to the
                root with an 'opX' edge and no child is connected with a 'rel' edge, then remove the root node, make the '
                opX' child the new root and keep track of all other edges that would get lost
            Else Break
        Else Break
        Add back all child edges that would get lost to the new root node
    If there are still 'and', 'before', 'after' nodes left over that have no parent nodes and fulfill
    the same preconditions as in the root node case, then follow the same steps for removing nodes
    and adding edges back but do NOT change the root node during the process
    :param sep_graph:
    :return:
    """
    orig_root_node = sep_graph.graph['root']

    # start with root node
    if orig_root_node in sep_graph.nodes:
        assert orig_root_node == sep_graph.graph['root']

        # only remove if it has no parent nodes
        incoming_edges = sep_graph.in_edges(orig_root_node)
        if len(incoming_edges) == 0:

            edges_removed = []
            while True:
                current_root = sep_graph.graph['root']
                node_to_remove, edges_to_remove, new_root = _remove_left_over_nodes(relevant_node=current_root, sep_graph=sep_graph)

                if not node_to_remove:
                    break

                sep_graph.remove_node(node_to_remove)
                sep_graph.graph['root'] = new_root
                if edges_to_remove:
                    edges_removed.extend(edges_to_remove)

            # add the edges back
            if edges_removed:
                current_root = sep_graph.graph['root']
                for e_rem in edges_removed:
                    new_edge = list(e_rem)
                    new_edge[0] = current_root
                    new_edge = tuple(new_edge)
                    sep_graph.add_edges_from([new_edge])

    # Check whether there are more nodes left that are now meaningless
    relevant_nodes = []
    for current_node in sep_graph.nodes():
        current_node_label = nx.get_node_attributes(sep_graph, 'label')[current_node]
        incoming_edges = sep_graph.in_edges(current_node)
        if len(incoming_edges) == 0 and current_node_label in ['and', 'before', 'after', 'multi-sentence']:
            relevant_nodes.append(current_node)

    # for each of these nodes go down the graph and remove left-over nodes until reaching
    # a node that should stay, then continue with the next node;
    # not very frequent that any other nodes get removed here
    for rel_node in relevant_nodes:
        edges_removed = []
        new_rel_node = rel_node
        while True:
            node_to_remove, edges_to_remove, relevant_child = _remove_left_over_nodes(relevant_node=new_rel_node, sep_graph=sep_graph)
            if not node_to_remove:
                break

            sep_graph.remove_node(node_to_remove)
            new_rel_node = relevant_child
            if edges_to_remove:
                edges_removed.extend(edges_to_remove)

        # very unlikely that this will occur; 0 times true for ARA 1 and ARA 2
        if edges_removed:
            for e_rem in edges_removed:
                new_edge = list(e_rem)
                new_edge[0] = new_rel_node
                new_edge = tuple(new_edge)
                sep_graph.add_edges_from(new_edge)

    return sep_graph


def _remove_left_over_nodes(relevant_node, sep_graph: nx.Graph):
    """
    Function to decide whether relevant_node is fulfills all conditions for being a "left-over" node
    that should get removed from the graph
    :param relevant_node: the node for which it gets checked whether it should be removed
    :param sep_graph: the AMR graph
    :return: a triple: (node_to_remove, removed_edges, node_to_continue_with)
            (None, None, None) if no left-over nodes left
            else:
                node_to_remove: actually identical with relevant_node
                removed_edges: the edges that get lost when removing node_to_remove and should be added back
                node_to_continue_with: the child node of relevant_node that gets the new root node or
                                       is simply the next node to check
    """
    relevant_node_label = nx.get_node_attributes(sep_graph, 'label')[relevant_node]
    out_edges = list(sep_graph.edges(relevant_node, data=True))

    if relevant_node_label not in ['and', 'before', 'after', 'multi-sentence']:
        return None, None, None

    if len(out_edges) == 1:
        only_edge = out_edges[0]
        only_edge = (only_edge[0], only_edge[1])
        new_root = only_edge[1]
        return relevant_node, None, new_root

    elif relevant_node_label == 'and':
        # check whether there is exactly one op edge and no rel edges
        child_edge_labels = []
        op_edges = []
        for out_edge in out_edges:
            lab = out_edge[2]['label']
            non_numbered_lab = remove_role_numbering_edge(lab)
            if non_numbered_lab == 'op':
                op_edges.append(out_edge)
            else:
                child_edge_labels.append(lab)

        if 'rel' in child_edge_labels or len(op_edges) != 1:
            return None, None, None
        else:
            edges_removed = []
            for out_edge in out_edges:
                if out_edge not in op_edges:
                    edges_removed.append(out_edge)
                    # print(out_edge[2]['label'])

            only_op_edge = op_edges[0]
            new_root = only_op_edge[1]
            return relevant_node, edges_removed, new_root

    else:
        return None, None, None


def update_root_node(sep_graph: nx.Graph, action_graph: nx.Graph, action_clusters: List[Dict]) -> nx.Graph:
    """
    Updates the root node of the action-level ARM
    If the current root node (either the original one or derived during removing left-over nodes)
     is still included in the action-level AMR then no changes are made
    Otherwise, among the main amr nodes for the corresponding action cluster, the amr node that is highest
    in the AMR graph is chosen or a lowest_common_ancestor
    :param sep_graph: a separated action-level AMR graph
    :param action_graph: the action graph of the corresponding recipe
    :param action_clusters: list of the action-node/action-aligned amr-node clusters
    :return: the action-level AMR with the updated root node
    """
    current_root_node = sep_graph.graph['root']
    if current_root_node not in sep_graph.nodes:
        for ac_cluster in action_clusters:
            action_nodes = []
            amr_nodes = []
            for ac_node, main_amr_nodes in ac_cluster.items():
                for amr_n in main_amr_nodes:
                    if amr_n in sep_graph.nodes:
                        action_nodes.append(ac_node)
                        amr_nodes.append(amr_n)
            action_nodes.sort()

            # if node cluster is for other separated amr continue
            if not action_nodes:
                continue

            # find the top node: in ARA 1 only 2 graphs with len(action_nodes) > 1, 5 graphs for ARA 2
            new_root_node = find_new_root(amr_nodes, sep_graph)
            # The following 4 lines were the first attempt to find the appropriate root node but
            # most times both led to the same node but if not the find_new_root method was better
            #highest_action_node = find_highest_node(action_nodes, action_graph)
            #main_amr_nodes_cluster = ac_cluster[highest_action_node]
            #if main_amr_nodes_cluster[0] != new_root_node:
                #print(sep_graph.name)

            sep_graph.graph['root'] = new_root_node
            # should never happen that more than one action cluster is related to an AMR
            # which was split (if it wasn't, the original root would still be part of the graph)
            break

    return sep_graph
