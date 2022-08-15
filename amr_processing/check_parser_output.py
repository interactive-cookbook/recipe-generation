import os
import networkx as nx
import penman.surface
from penman_networkx_conversions import penman2networkx, networkx2penman
from read_graphs import read_aligned_amr_file
from typing import List, Tuple

"""
Script to check for unaligned AMR nodes and for AMRs including cycles 
"""


def get_penman_amrs_complete_corpus(corpus_dir) -> List[penman.Graph]:
    """
    Creates the penman graphs for all individual sentence level AMRs
    in the subdirectories of the corpus_dir
    :param corpus_dir:
    :return: returns a list of all AMRs as penman Graph objects
    """
    amr_graphs_pen = []

    for dish in os.listdir(corpus_dir):
        for recipe in os.listdir('/'.join([corpus_dir, dish, 'amrs'])):
            pen_amrs = read_aligned_amr_file('/'.join([corpus_dir, dish, 'amrs', recipe]))

            for gr in pen_amrs:
                amr_graphs_pen.append(gr)

    return amr_graphs_pen


def get_nx_amrs_complete_corpus(corpus_dir) -> List[nx.Graph]:
    """
    Creates the network X graphs for all individual sentence level AMRs
    in the subdirectories of the corpus_dir
    :param corpus_dir:
    :return: returns a list of all AMRs as network x Graph objects
    """
    amr_graphs_nx = []

    for dish in os.listdir(corpus_dir):
        for recipe in os.listdir('/'.join([corpus_dir, dish, 'amrs'])):
            pen_amrs = read_aligned_amr_file('/'.join([corpus_dir, dish, 'amrs', recipe]))

            for gr in pen_amrs:
                amr_graphs_nx.append(penman2networkx(gr))

    return amr_graphs_nx


def check_cycles(corpus_dir: str) -> Tuple[List[nx.Graph], int]:
    """
    Finds all amr graphs containing cycles in all AMR files in the subdirectories of corpus_dir
    and counts them
    :param corpus_dir:
    :return: list of all networkx.Graph AMRs that contain a cycle
             and the number of cycles
    """
    amr_graphs_nx = get_nx_amrs_complete_corpus(corpus_dir)

    # check for cycles in the created AMR graphs
    cyclic_graphs = []
    number_of_cycles = 0
    for nx_gr in amr_graphs_nx:
        cycles = nx.simple_cycles(nx_gr)
        cycles = list(cycles)
        number_of_cycles += len(cycles)
        if len(cycles) > 0:
            cyclic_graphs.append(nx_gr)

    return cyclic_graphs, number_of_cycles


def check_unaligned_nodes(corpus_dir) -> Tuple[List[penman.Graph], List[penman.Triple]]:
    """
    Finds all sentence level AMRs in the subdirectories of the corpus_dir where not all nodes of the
    AMR are aligned to a token
    :param corpus_dir:
    :return: returns a list of all penman.Graph AMRs with unaligned nodes
            and the number of unaligned nodes in the corpus
    """
    amr_graphs_pen = get_penman_amrs_complete_corpus(corpus_dir)

    # check for nodes that do not have an alignment
    graphs_with_missing_alignments = []
    unaligned_nodes = []
    for pen_gr in amr_graphs_pen:
        includes_unaligned_nodes = False

        for inst in pen_gr.instances():
            ep_data = pen_gr.epidata[inst]
            aligned = False
            for ep in ep_data:
                if isinstance(ep, penman.surface.Alignment):
                    aligned = True
            if not aligned:
                includes_unaligned_nodes = True
                unaligned_nodes.append(inst)

        for attr in pen_gr.attributes():
            ep_data = pen_gr.epidata[attr]
            aligned = False
            for ep in ep_data:
                if isinstance(ep, penman.surface.Alignment):
                    aligned = True
            if not aligned:
                includes_unaligned_nodes = True
                unaligned_nodes.append(attr)
        if includes_unaligned_nodes:
            graphs_with_missing_alignments.append(pen_gr)

    return graphs_with_missing_alignments, unaligned_nodes


def get_cyclic_graphs(corpus_dir, output_file):
    """
    Finds all amr graphs containing cycles in all AMR files in the subdirectories of corpus_dir
    and writes them into the file output_file
    :param corpus_dir:
    :param output_file: path to output file
    :return:
    """
    cyclic_graphs, numb_of_cycles = check_cycles(corpus_dir)

    with open(output_file, 'w', encoding='utf-8') as cyc_file:
        for cg in cyclic_graphs:
            penman_cyc_gr = networkx2penman(cg)
            pen_str = penman.encode(penman_cyc_gr)
            cyc_file.write(f'{pen_str}\n\n')

    print(f'Number of cycles: {numb_of_cycles}')


# TODO: move to the script which will process the AMRs
def remove_cycle(cyclic_amr_graph: nx.Graph) -> nx.Graph:
    """
    Transforms an cyclic amr graph into an acyclic graph by removing edges in the following way:
    If cycle is between 2 nodes:
        If  :quant involved -> delete :quant
        If :ARG involved -> delete :ARG
        If :purpose involved -> delete :purpose
        If :op involved -> delete :op
    If number of nodes in the cycle is larger than 2 then a special rule is hardcoded for the two
    cases that occur in the ARA corpus
    :param cyclic_amr_graph: a nx.Graph AMR with cycles
    :return: an acyclic version of the input AMR
    """
    cycles = nx.simple_cycles(cyclic_amr_graph)

    acyclic_amr_graph = cyclic_amr_graph.copy()

    for cycle in cycles:

        if len(cycle) == 2:
            node1 = cycle[0]
            node2 = cycle[1]
            edge1 = acyclic_amr_graph.get_edge_data(node1, node2)['label']
            edge2 = acyclic_amr_graph.get_edge_data(node2, node1)['label']

            # Apply rules to eliminate cycle
            if edge1 == ':quant':
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2 == ':quant':
                acyclic_amr_graph.remove_edge(node2, node1)
            elif edge1.find(':ARG') != -1:
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2.find(':ARG') != -1:
                acyclic_amr_graph.remove_edge(node2, node1)
            elif edge1 == ':purpose':
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2 == ':purpose':
                acyclic_amr_graph.remove_edge(node2, node1)
            elif edge1.find(':op') != -1:
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2.find(':op') != -1:
                acyclic_amr_graph.remove_edge(node2, node1)

        # for the 2 longer cycles remove the incoming ARG1 edge of the heat node
        else:
            for ind, node in enumerate(cycle):
                label = nx.get_node_attributes(acyclic_amr_graph, 'label')[node]
                if label == 'heat':
                    parent = cycle[ind - 1]
                    edge = acyclic_amr_graph.get_edge_data(parent, node)['label']
                    assert edge == ':ARG1'
                    acyclic_amr_graph.remove_edge(parent, node)

    return acyclic_amr_graph



if __name__=="__main__":

    #get_cyclic_graphs('./aligned_recipe_amrs_ibm', './cyclic_graphs.txt')
    graphs_missing_align, unaligned = check_unaligned_nodes('./aligned_recipe_amrs_ibm')
    #print(f'Number of cycles: {numb_of_cycles}')

    # Number of cycles: 30
    # Number of unaligned nodes: 0



