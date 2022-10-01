from typing import List
import networkx as nx
from amr_processing.helpers import find_highest_node, remove_role_numbering_paths, remove_role_numbering_edge


def postprocess_split_amrs(separated_amrs: List,
                           original_amr: nx.Graph,
                           action_graph: nx.Graph,
                           action_clusters):
    """

    :param separated_amrs:
    :param original_amr:
    :param action_graph:
    :param action_clusters:
    :return:
    """

    post_sep_amrs = []

    for new_id, sep_amr in enumerate(separated_amrs):
        new_sep_amr = sep_amr.copy()
        # update graph name / instruction ID
        new_name = new_sep_amr.graph['id'] + f'_{new_id}'
        # if new_name == 'baked_ziti_0_instr9_1':
        # print("here")
        new_sep_amr.graph['id'] = new_name
        new_sep_amr.name = new_name
        new_sep_amr.graph['name'] = new_name

        # update action node aligned amr nodes
        new_alignments = []
        for orig_al in original_amr.graph['alignments'].split(', '):
            if orig_al in new_sep_amr.nodes:
                new_alignments.append(orig_al)
        new_sep_amr.graph['alignments'] = ', '.join(new_alignments)

        # TODO: exclude left-over "and" nodes etc
        # look for nodes without predecessors, check if it is an "and" node and if yes
            # check if there is only one child node, if yes remove the "and" node
                # if the child node is before or after also remove that node and check if there is still "and" left
            # check if there is only one :op edge child -> remove the "and" node

        orig_root_node = new_sep_amr.graph['root']

        if orig_root_node in new_sep_amr.nodes:
            edges_removed = []
            while True:
                current_root = new_sep_amr.graph['root']
                current_root_label = nx.get_node_attributes(new_sep_amr, 'label')[current_root]

                if current_root_label not in ['and', 'before', 'after']:
                    break

                root_out_edges = list(new_sep_amr.edges(current_root, data=True))
                if len(root_out_edges) == 1:
                    only_edge = root_out_edges[0]
                    only_edge = (only_edge[0], only_edge[1])
                    new_root = only_edge[1]
                    new_sep_amr.remove_node(current_root)
                    new_sep_amr.graph['root'] = new_root

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
                                print(out_edge[2]['label'])

                        only_op_edge = op_edges[0]
                        new_root = only_op_edge[1]
                        new_sep_amr.remove_node(current_root)
                        new_sep_amr.graph['root'] = new_root

                else:
                    break

            # TODO add back all removed edges to the new root
            if edges_removed:
                current_root = new_sep_amr.graph['root']
                for e_rem in edges_removed:
                    new_edge = list(e_rem)
                    new_edge[0] = current_root
                    new_edge = tuple(new_edge)
                    new_sep_amr.add_edges_from([new_edge])

        # update root node
        if orig_root_node not in new_sep_amr.nodes:
            for ac_cluster in action_clusters:
                action_nodes = []
                for ac_node, amr_nodes in ac_cluster.items():
                    for amr_n in amr_nodes:
                        if amr_n in new_sep_amr.nodes:
                            action_nodes.append(ac_node)
                action_nodes.sort()

                # if node cluster is for other separated amr continue
                if not action_nodes:
                    continue

                # find the top node
                highest_action_node = find_highest_node(action_nodes, action_graph)
                main_amr_nodes = ac_cluster[highest_action_node]
                new_sep_amr.graph['root'] = main_amr_nodes[0]
                break

        post_sep_amrs.append(new_sep_amr)

    return post_sep_amrs