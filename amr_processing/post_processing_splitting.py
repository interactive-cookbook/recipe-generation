from typing import List
import networkx as nx
from amr_processing.helpers import find_highest_node


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
        # update graph name / instruction ID
        new_name = sep_amr.graph['id'] + f'_{new_id}'
        # if new_name == 'baked_ziti_0_instr9_1':
        # print("here")
        sep_amr.graph['id'] = new_name
        sep_amr.name = new_name
        sep_amr.graph['name'] = new_name

        # update action node aligned amr nodes
        new_alignments = []
        for orig_al in original_amr.graph['alignments'].split(', '):
            if orig_al in sep_amr.nodes:
                new_alignments.append(orig_al)
        sep_amr.graph['alignments'] = ', '.join(new_alignments)

        # TODO: exclude left-over "and" nodes etc
        # look for nodes without predecessors, check if it is an "and" node and if yes
            # check if there is only one child node, if yes remove the "and" node
                # if the child node is before or after also remove that node and check if there is still "and" left
            # check if there is only one :op edge child -> remove the "and" node



        # update root node
        if sep_amr.graph['root'] not in sep_amr.nodes:
            for ac_cluster in action_clusters:
                action_nodes = []
                for ac_node, amr_nodes in ac_cluster.items():
                    for amr_n in amr_nodes:
                        if amr_n in sep_amr.nodes:
                            action_nodes.append(ac_node)
                action_nodes.sort()

                # if node cluster is for other separated amr continue
                if not action_nodes:
                    continue

                # find the top node
                highest_action_node = find_highest_node(action_nodes, action_graph)
                main_amr_nodes = ac_cluster[highest_action_node]
                sep_amr.graph['root'] = main_amr_nodes[0]
                break

        post_sep_amrs.append(sep_amr)

    return post_sep_amrs