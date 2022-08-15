import penman
from penman import surface
import networkx as nx


"""
Functions to convert a penman Graph object into a network x Graph object and 
to convert a network x Graph object into a penman Graph object
"""


def penman2networkx(penman_graph: penman.Graph) -> nx.Graph:
    """
    Converts a penman Graph object into a networkX Graph object
    :param penman_graph:
    :return:
    """
    # extract all relevant information from AMR
    edges = penman_graph.edges()
    instances = penman_graph.instances()
    attributes = penman_graph.attributes()
    ep_data = penman_graph.epidata
    alignments = surface.alignments(penman_graph)

    # create the nx Graph object
    nx_graph = nx.DiGraph()
    nx_graph.graph['graph'] = dict()

    for meta_key, meta_value in penman_graph.metadata.items():
        nx_graph.graph[meta_key] = meta_value
        if meta_key == 'id':
            nx_graph.name = meta_value      # give the networkx graph the graph id as name
        if meta_key == 'snt':
            nx_graph.graph['graph']['label'] = meta_value       # needed for including the sentence in the dot files for pictures

    # add all nodes and edges of the original AMR,
    # Keep track of the epidata to be able to reconstruct token-node alignments
    # Keep track of the type of node that gets added to correctly reconstruct a penman AMR
    # Remove colon at beginning of role labels, i.e. :ARG1 -> ARG1
    for inst in instances:
        node_epi_data = ep_data[(inst.source, inst.role, inst.target)]
        try:
            aligned_token = alignments[(inst.source, inst.role, inst.target)].indices[0]
        except KeyError:
            aligned_token = 0
        nx_graph.add_nodes_from([(inst.source,
                                  {'label': inst.target, 'type': 'instance', 'epi': node_epi_data, 'alignment': str(aligned_token)})])

    for e in edges:
        edge_epi_data = ep_data[(e.source, e.role, e.target)]
        role_label = e.role[1:] if e.role[0] == ':' else e.role
        nx_graph.add_edges_from([(e.source, e.target, {'label': role_label, 'type': 'edge', 'epi': edge_epi_data})])

    for a in attributes:
        try:
            aligned_token = alignments[(a.source, a.role, a.target)].indices[0]
        except KeyError:
            aligned_token = 0
        nx_graph.add_nodes_from([(a.target, {'label': a.target, 'type': 'attribute', 'alignment': str(aligned_token)})])
        edge_epi_data = ep_data[(a.source, a.role, a.target)]  # add epi data only for edge to avoid issues when converting back
        role_label = a.role[1:] if a.role[0] == ':' else a.role
        nx_graph.add_edges_from([(a.source, a.target, {'label': role_label, 'type': 'attribute_edge', 'epi': edge_epi_data})])

    return nx_graph


def networkx2penman(nx_graph: nx.Graph) -> penman.Graph:
    """
    Converts a networkX Graph object into a penman Graph object
    The networkX Graph object should have been originally derived from a penman Graph object
    or should include the same graph, edge and node attributes
    :param nx_graph:
    :return:
    """
    triples = []
    ep_data = dict()

    # add the triples for all nodes
    for node, attr_dict in nx_graph.nodes.data():
        label = attr_dict['label']
        triple_type = attr_dict['type']

        if triple_type == 'instance':
            triples.append((node, ':instance', label))
            node_epi_data = attr_dict['epi']
            ep_data[(node, ':instance', label)] = node_epi_data
        # attributes are added as edges, otherwise they occur twice
        elif triple_type == 'attribute':
            continue

    # add the triples for all edges
    for source, target, attr_dict in nx_graph.edges.data():
        role = attr_dict['label']
        role = role if role[0] == ':' else ':' + role       # add colons back for correct penman processing
        edge_epi_data = attr_dict['epi']
        triples.append((source, role, target))
        ep_data[(source, role, target)] = edge_epi_data

    # create penman graph from the triples and add the epidata
    penman_graph = penman.graph.Graph(triples)
    for trip, ep in ep_data.items():
        penman_graph.epidata[trip] = ep

    # add metadata
    for graph_attr, attr_val in nx_graph.graph.items():
        if graph_attr != 'graph':       # skip the subdict that is only for the visualization
            penman_graph.metadata[graph_attr] = attr_val

    return penman_graph


if __name__=="__main__":
    pstr3 = '(z1 / sprinkle-01~e.11 :mode imperative :arg0 (z2 / you) :arg1 (z3 / beef~e.12) :arg2 (z4 / and~e.21 :op1 (z5 / salt~e.16 :mod (z6 / sea~e.15)) :op2 (z7 / pepper~e.18) :op3 (z8 / oregano~e.20) :op4 (z9 / basil~e.22)) :arg1-of (z10 / liberal-02~e.13) :time~e.9 (z11 / brown-01~e.10))'
    pen_graph = penman.decode(pstr3)
    pen_graph.metadata['snt'] = 'Sprinkle and so on .'
    pen_graph.metadata['id'] = 'test_name'
    nx_gr = penman2networkx(pen_graph)
    print(nx_gr.graph)
    print(list(nx_gr.nodes))
    #p = nx.drawing.nx_pydot.to_pydot(nx_gr)
    #print(p)
    #for n in nx_gr.edges():
        #print(n)
        #print(nx_gr.get_edge_data(n[0], n[1]))

    new_p = networkx2penman(nx_gr)
    print(new_p.metadata)
    print(new_p.epidata)
    print(new_p.triples)
