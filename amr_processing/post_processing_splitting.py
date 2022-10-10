from typing import List, Dict
import networkx as nx
from amr_processing.helpers import find_highest_node, remove_role_numbering_edge


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
    new_name = sep_graph.graph['id'] + f'_{new_instr_id}'
    sep_graph.graph['id'] = new_name
    sep_graph.name = new_name
    sep_graph.graph['name'] = new_name

    return sep_graph


def update_alignments(sep_graph: nx.Graph, orig_graph: nx.Graph) -> nx.Graph:
    """
    Update the alignment information of a separated action-level amr
    such that it includes all nodes in the alignment information
    that were aligned nodes in the original AMR and are still part of the
    separated AMR
    :param sep_graph: one of the separated action-level amr
    :param orig_graph: the corresponding original sentence-level amr
    :return: the graph with the updated alignment information
    """
    new_alignments = []
    for orig_al in orig_graph.graph['alignments'].split(', '):
        if orig_al in sep_graph.nodes:
            new_alignments.append(orig_al)
    sep_graph.graph['alignments'] = ', '.join(new_alignments)

    return sep_graph


def remove_left_over_nodes(sep_graph: nx.Graph) -> nx.Graph:
    """
    Remove left-over 'and' nodes with only one conjunct and 'before' and 'after' nodes that do not make
    sense anymore
    If the root node of the action-level AMR is labelled 'and', 'before' or 'after' then:
        If the root node has only one child node, then remove the current root node and make the child node the new root
        Elif If the root node has more child nodes and root label is 'and' and exactly one child is connected to the
            root with an 'opX' edge and no child is connected with a 'rel' edge, then remove the root node, make the '
            opX' child the new root and keep track of all other edges that would get lost
        Else Break
    Else Break
    Add back all child edges that would get lost to the new root node
    :param sep_graph:
    :return:
    """
    orig_root_node = sep_graph.graph['root']

    if orig_root_node not in sep_graph.nodes:
        return sep_graph

    # only remove if it has not parent nodes
    incoming_edges = sep_graph.in_edges(orig_root_node)
    if len(incoming_edges):
        return sep_graph

    else:
        edges_removed = []
        while True:
            current_root = sep_graph.graph['root']
            current_root_label = nx.get_node_attributes(sep_graph, 'label')[current_root]

            if current_root_label not in ['and', 'before', 'after']:
                break

            root_out_edges = list(sep_graph.edges(current_root, data=True))
            if len(root_out_edges) == 1:
                only_edge = root_out_edges[0]
                only_edge = (only_edge[0], only_edge[1])
                new_root = only_edge[1]
                sep_graph.remove_node(current_root)
                sep_graph.graph['root'] = new_root

            elif current_root_label == 'and':
                # check whether there is exactly one op edge and no rel edges
                child_edge_labels = []
                op_edges = []
                for out_edge in root_out_edges:
                    lab = out_edge[2]['label']
                    non_numbered_lab = remove_role_numbering_edge(lab)
                    if non_numbered_lab == 'op':
                        op_edges.append(out_edge)
                    else:
                        child_edge_labels.append(lab)

                if 'rel' in child_edge_labels or len(op_edges) != 1:
                    break
                else:
                    for out_edge in root_out_edges:
                        if out_edge not in op_edges:
                            edges_removed.append(out_edge)
                            # print(out_edge[2]['label'])

                    only_op_edge = op_edges[0]
                    new_root = only_op_edge[1]
                    sep_graph.remove_node(current_root)
                    sep_graph.graph['root'] = new_root

            else:
                break

        if edges_removed:
            current_root = sep_graph.graph['root']
            for e_rem in edges_removed:
                new_edge = list(e_rem)
                new_edge[0] = current_root
                new_edge = tuple(new_edge)
                sep_graph.add_edges_from([new_edge])

        return sep_graph


def update_root_node(sep_graph: nx.Graph, action_graph: nx.Graph, action_clusters: List[Dict]) -> nx.Graph:
    """
    Updates the root node of the action-level ARM
    If the original root node is still included in the action-level AMR then no changes are made
    Otherwise, the main amr node aligned to the highest action node in the action graph is chosen
    from all action nodes in the corresponding cluster
    :param sep_graph: a separated action-level AMR graph
    :param action_graph: the action graph of the corresponding recipe
    :param action_clusters: list of the action-node/action-aligned amr-node clusters
    :return: the action-level AMR with the updated root node
    """
    current_root_node = sep_graph.graph['root']
    if current_root_node not in sep_graph.nodes:
        for ac_cluster in action_clusters:
            action_nodes = []
            for ac_node, amr_nodes in ac_cluster.items():
                for amr_n in amr_nodes:
                    if amr_n in sep_graph.nodes:
                        action_nodes.append(ac_node)
            action_nodes.sort()

            # if node cluster is for other separated amr continue
            if not action_nodes:
                continue

            # find the top node
            highest_action_node = find_highest_node(action_nodes, action_graph)
            main_amr_nodes = ac_cluster[highest_action_node]
            sep_graph.graph['root'] = main_amr_nodes[0]
            break

    return sep_graph
