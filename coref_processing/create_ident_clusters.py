
from typing import List

from create_joined_coref import find_post_splitting_coref
from graph_processing.read_graphs import read_aligned_amr_file
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman


def create_ident_clusters(action_amr_file, ara_file):
    """
    Creates the information about which nodes in the A-AMRs where previously a single node in the S-AMR, i.e.
    extracts coreference information from the graphs
    :param action_amr_file: path to the file with the A-AMRs
    :param ara_file: path to the file with the corresponding action graph
    :return: list of the cluster information (see beginning of create_joined_coref.py for an example)
    """

    # read the AMR file and convert to networkX graphs
    amr_graphs_penman = read_aligned_amr_file(action_amr_file)
    amr_graphs = []
    for pen_gr in amr_graphs_penman:
        amr_graphs.append(penman2networkx(pen_gr))

    # read the recipe text from the action graph files
    recipe_text = get_recipe_text(ara_file)

    ident_clusters = find_post_splitting_coref(amr_graphs, recipe_text)

    # ignore the names of the clusters
    ident_clusters_list = [value for key, value in ident_clusters.items()]

    return ident_clusters_list


def get_recipe_text(recipe_path) -> List[str]:
    """
    Reads in the tokenized recipe text from the .conllu file of an action graph
    :param recipe_path: path to a .conllu file of an action graph
    :return: list of the tokens of the recipe
    """
    text = []
    with open(recipe_path, 'r', encoding='utf-8') as f:
        for line in f:
            columns = line.split('\t')
            token = columns[0]
            text.append(token)
    return text


