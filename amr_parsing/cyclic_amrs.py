import os
import networkx as nx
import penman.surface
from pathlib import Path
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman
from graph_processing.read_graphs import read_aligned_amr_file
from typing import List, Tuple

"""
Script to check for AMRs including cycles and to replace the cyclic AMRs with a acyclic one 
in the corpus files
"""


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

        fixed = False

        if len(cycle) == 1:
            node = cycle[0]
            acyclic_amr_graph.remove_edge(node, node)
            fixed = True

        if len(cycle) == 2:
            node1 = cycle[0]
            node2 = cycle[1]
            edge1 = acyclic_amr_graph.get_edge_data(node1, node2)['label']
            edge2 = acyclic_amr_graph.get_edge_data(node2, node1)['label']
            fixed = True

            # Apply rules to eliminate cycle
            if edge1 == 'quant':
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2 == 'quant':
                acyclic_amr_graph.remove_edge(node2, node1)
            elif edge1.find('ARG') != -1:
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2.find('ARG') != -1:
                acyclic_amr_graph.remove_edge(node2, node1)
            elif edge1 == 'purpose':
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2 == 'purpose':
                acyclic_amr_graph.remove_edge(node2, node1)
            elif edge1.find('op') != -1:
                acyclic_amr_graph.remove_edge(node1, node2)
            elif edge2.find('op') != -1:
                acyclic_amr_graph.remove_edge(node2, node1)
            else:
                fixed = False

        # for the 2 longer cycles remove the incoming ARG1 edge of the heat node
        if not fixed:
            graph_root = acyclic_amr_graph.graph['root']
            current_shortest_length = 100
            current_closest_node = ""

            for ind, node in enumerate(cycle):
                if node == graph_root:
                    current_shortest_length = 0
                    current_closest_node = node
                else:
                    paths = nx.all_simple_paths(acyclic_amr_graph, graph_root, node)
                    paths = list(paths)
                    paths.sort(key=len)
                    if len(paths[0]) < current_shortest_length:
                        current_shortest_length = len(paths[0])
                        current_closest_node = node

            # find all incoming edges of the current closest node and remove the one that belongs to the cycle
            incoming_edges = acyclic_amr_graph.in_edges(current_closest_node)
            edge_to_remove = None
            for in_edge in incoming_edges:
                if in_edge[0] in cycle and in_edge[1] in cycle:
                    edge_to_remove = in_edge
                    break
            acyclic_amr_graph.remove_edge(edge_to_remove[0], edge_to_remove[1])

            """
            label = nx.get_node_attributes(acyclic_amr_graph, 'label')[node]
            if label == 'heat':
                parent = cycle[ind - 1]
                edge = acyclic_amr_graph.get_edge_data(parent, node)['label']
                assert edge == 'ARG1'
                acyclic_amr_graph.remove_edge(parent, node)
            """

    return acyclic_amr_graph


def fix_cyclic_amrs(amr_corpus_dir, fixed_amr_corpus_dir):
    """
    Create a copy of the input amr corpus where all cyclic AMRs are manipulated (i.e. an edge is removed)
    to create acyclic AMRs
    :param amr_corpus_dir: the parent directory of the AMRs potentially including cycles
    :param fixed_amr_corpus_dir: the name of the new directory for the fixed corpus version
    :return:
    """
    Path(fixed_amr_corpus_dir).mkdir(exist_ok=True, parents=True)
    count_fixes = 0

    for dish in os.listdir(amr_corpus_dir):
        Path('/'.join([fixed_amr_corpus_dir, dish])).mkdir(exist_ok=True, parents=True)

        for recipe in os.listdir('/'.join([amr_corpus_dir, dish])):

            with open('/'.join([fixed_amr_corpus_dir, dish, recipe]), 'w', encoding='utf-8') as new_file:

                pen_amrs = read_aligned_amr_file('/'.join([amr_corpus_dir, dish, recipe]))

                for gr in pen_amrs:
                    nx_graph = penman2networkx(gr)
                    cycles = nx.simple_cycles(nx_graph)
                    if len(list(cycles)) > 0:
                        nx_graph = remove_cycle(nx_graph)
                        count_fixes += 1
                    cycles = nx.simple_cycles(nx_graph)
                    if len(list(cycles)) > 0:
                        print(f'Did not work for {nx_graph.name}')
                    pen_gr = networkx2penman(nx_graph)
                    pen_str = penman.encode(pen_gr)
                    new_file.write(f'{pen_str}\n\n')

    print(f'Fixed {count_fixes} cyclic AMR graphs.')


if __name__=="__main__":

    cyclic_amr_dir = "./aligned_cyclic_shifted_recipe_amrs_ara2"
    output_dir = "../data_ara2/recipe_amrs_sentences"
    fix_cyclic_amrs(cyclic_amr_dir, output_dir)

    # Number of cycles in original ARA 1.1: 29
    # Number of cycles in ARA 2: 33

