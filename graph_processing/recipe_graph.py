import networkx as nx
import os

"""
Functions to read a recipe / action graph from a file in conllu format and to create a networkX graph for it
"""

def _read_graph_conllu(conllu_graph_file, token_ids):
    """
    Reads in a graph - either a full recipe graph or action graph - and extracts all nodes,
    edges, the tag labels from the file
    :param conllu_graph_file: path to graph file in conllu format
    :param token_ids: whether the node labels should include the token ids
                e.g. if ids = True than node is labelled with "1_Preheat" otherwise with "Preheat"
    :return: list of nodes, list of edges including labels, tags_dict (key = node, value = tag)
    """
    node_tuples = []
    edge_list = []

    parents = []
    children = []

    with open(conllu_graph_file, "r", encoding="utf-8") as grf:
        complete_token = ""
        prev_id = 0
        tags_dict = {}

        for line in grf:

            if line == "\n" or line == "":
                break
            columns = line.strip().split()
            id = columns[0]
            token = columns[1]
            tag = columns[4]
            edge = columns[6]
            edge_label = columns[7]

            if tag == "O":
                tag = tag
                if complete_token != "":
                    node_tuple = (prev_id, {"label": complete_token})
                    node_tuples.append(node_tuple)
                    complete_token = ""

            elif tag[0] == "B":
                if complete_token != "":
                    node_tuple = (prev_id, {"label": complete_token})
                    node_tuples.append(node_tuple)
                if token_ids:
                    complete_token = str(id) + "_" + token
                else:
                    # complete_token = token
                    complete_token = id
                prev_id = id
                if edge != "0":
                    edge_list.append((id, edge, {"label": edge_label}))
                    parents.append(id)
                    children.append(edge)

            elif tag[0] == "I":
                complete_token += " " + token

        if complete_token != "":
            node_tuple = (prev_id, {"label": complete_token})
            node_tuples.append(node_tuple)
            complete_token = ""

    # add 'end' node
    for child in children:
        if child not in parents:
            edge_list.append((child, "end", {"label": "end"}))

    # create dictionary with node as key and tag as value
    with open(conllu_graph_file, "r", encoding="utf-8") as grf:
        for line in grf:
            if line == "\n" or line == "":
                break
            columns = line.strip().split()
            id = columns[0]
            tag = columns[4]
            for node_tuple in node_tuples:
                if node_tuple[0] == id:
                    tags_dict[str(node_tuple[1]["label"])] = tag

    return node_tuples, edge_list, tags_dict


def read_graph_from_conllu(conllu_graph_file, token_ids=True):
    """
    Reads a graph file - either recipe or action graph - in conllu format and transforms it into
    a NetworkX object
    :param conllu_graph_file: path to graph file in conllu format
    :param token_ids: whether the node labels should include the token ids
                e.g. if ids = True than node is labelled with "1_Preheat" otherwise with "Preheat"
    :return: a graph in NetworkX format
    """
    import collections
    node_list = []
    nodes, edges, tags_dict = _read_graph_conllu(conllu_graph_file, token_ids)

    # the following extracts not only the node indices, but also their labels
    for node_tuple in nodes:
        node_tuple = list(node_tuple)
        label = node_tuple[1]["label"]
        node_list.append(label)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    conllu_graph_file_name = conllu_graph_file.split(os.sep)[-1]    # remove path and keep only file name
    conllu_graph_file_name = '.'.join(conllu_graph_file_name.split('.')[:-1])   # remove file ending .conllu
    G_name = "G_" + str(conllu_graph_file_name)

    nodes_attributes = collections.defaultdict(dict)
    for node_tuple in nodes:
        label = node_tuple[1]["label"]
        nodes_attributes[node_tuple[0]] = {"label": label, "tag": tags_dict[label],
                                           "origin": G_name}
    nx.set_node_attributes(G, nodes_attributes)

    return G


if __name__=='__main__':

    file_p1 = "../data/ara1.1/baked_ziti/recipes/baked_ziti_0.conllu"
    file_p2 = "../data_ara2/ara2.0/bananas_foster/recipes/bananas_foster_0.conllu"
    g1 = read_graph_from_conllu(file_p1)
    g2 = read_graph_from_conllu(file_p2)
