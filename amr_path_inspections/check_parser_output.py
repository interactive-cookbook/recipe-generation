import os
import penman.surface
from graph_processing.read_graphs import read_aligned_amr_file
from typing import List, Tuple

"""
Script to check for unaligned AMR nodes  
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


if __name__=="__main__":

    graphs_missing_align, unaligned = check_unaligned_nodes('./aligned_recipe_amrs_ibm')
    print(unaligned)

    # Number of unaligned nodes: 0



