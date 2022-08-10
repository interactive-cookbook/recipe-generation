import penman
from penman import surface
import networkx as nx


# TODO add documentation
def penman2networkx(penman_graph: penman.Graph):

    nx_graph = nx.DiGraph()

    edges = penman_graph.edges()
    instances = penman_graph.instances()
    attributes = penman_graph.attributes()
    ep_data = penman_graph.epidata

    for inst in instances:
        node_epi_data = ep_data[(inst.source, inst.role, inst.target)]
        nx_graph.add_nodes_from([(inst.source, {'label': inst.target, 'type': 'instance', 'epi': node_epi_data})])

    for e in edges:
        edge_epi_data = ep_data[(e.source, e.role, e.target)]
        nx_graph.add_edges_from([(e.source, e.target, {'label': e.role, 'epi': edge_epi_data})])

    for a in attributes:
        nx_graph.add_nodes_from([(a.target, {'label': a.target, 'type': 'attribute'})])
        edge_epi_data = ep_data[(a.source, a.role, a.target)]
        nx_graph.add_edges_from([(a.source, a.target, {'label': a.role, 'epi': edge_epi_data})])

    return nx_graph


def networkx2penman(nx_graph: nx.Graph):

    triples = []
    ep_data = dict()
    for node, attr_dict in nx_graph.nodes.data():
        label = attr_dict['label']
        triple_type = attr_dict['type']

        if triple_type == 'instance':
            triples.append((node, ':instance', label))
            node_epi_data = attr_dict['epi']
            ep_data[(node, ':instance', label)] = node_epi_data
        elif triple_type == 'attribute':
            continue

    for source, target, attr_dict in nx_graph.edges.data():
        role = attr_dict['label']
        edge_epi_data = attr_dict['epi']
        triples.append((source, role, target))
        ep_data[(source, role, target)] = edge_epi_data

    penman_graph = penman.graph.Graph(triples)
    for trip, ep in ep_data.items():
        penman_graph.epidata[trip] = ep

    return penman_graph


if __name__=="__main__":
    pstr3 = '(z1 / sprinkle-01~e.11 :mode imperative :arg0 (z2 / you) :arg1 (z3 / beef~e.12) :arg2 (z4 / and~e.21 :op1 (z5 / salt~e.16 :mod (z6 / sea~e.15)) :op2 (z7 / pepper~e.18) :op3 (z8 / oregano~e.20) :op4 (z9 / basil~e.22)) :arg1-of (z10 / liberal-02~e.13) :time~e.9 (z11 / brown-01~e.10))'
    pen_graph = penman.decode(pstr3)
    nx_gr = penman2networkx(pen_graph)
    p = nx.drawing.nx_pydot.to_pydot(nx_gr)
    #print(p)
    for n in nx_gr.edges():
        print(n)
        print(nx_gr.get_edge_data(n[0], n[1]))

    new_p = networkx2penman(nx_gr)
    #(new_p.epidata)
