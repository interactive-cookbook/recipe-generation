import networkx as nx
from typing import List


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
