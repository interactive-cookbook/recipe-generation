import json
import os
from pathlib import Path
import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict
from graph_processing.read_graphs import read_aligned_amr_file
from amr_processing.penman_networkx_conversions import penman2networkx
from coref_utils import get_coref_clusters_original, get_coref_clusters_extended

"""
Functions to create .jsonlines files with the combined information from the coreference files, the coreference 
information about previously shared amr nodes and then node variables of the amr nodes aligned to tokens from 
the coref clusters

Example: 
If text-based coreference file is: {"text": ["Preheat", "oven", "to", ...], "sentence_map": [...], 
                                    "clusters": [[[4, 4], [11, 12], [81, 82]], [[84, 87], [114, 115]]]
                                    }
Then the created joined coref file includes one entry for each cluster with the shifted token IDs (and only one ID if only one token)
the corresponding tokens in the sentence and the amr_nodes information which has the same list structure as 
"tokens" but for each token it includes a list of [the amr node label, the amr node variable, the ID of the amr graph]
{"coref-rel-0": {"token_ids": [[5], [12, 13], [82, 83]], 
                 "tokens": [["pasta"], ["the", "pasta"], ["the", "pasta"]], 
                 "amr_nodes": [[[["pasta", "p", "baked_ziti_1_instr1"]]], [[["pasta", "p", "baked_ziti_1_instr5_0"]]]]}}
{"coref-rel-1": {"token_ids": [[85, 86, 87, 88], [115, 116]], 
                 "tokens": [["the", "shredded", "mozzarella", "cheese"], ["the", "mozzarella"]], 
                 "amr_nodes": [[[["shred-01", "s", "baked_ziti_1_instr5_1"]], [["mozzarella", "m", "baked_ziti_1_instr5_0"], ["mozzarella", "m", "baked_ziti_1_instr5_1"]], [["cheese", "c", "baked_ziti_1_instr5_0"], ["cheese", "c", "baked_ziti_1_instr5_1"]]], [[["mozzarella", "m", "baked_ziti_1_instr6"]]]]}}

Additionally, for each node in the original sentence-level AMRs that occurs now in more than one action-level AMR an
entry of the following format is added:
{"split-rel-0": {"token_id": 28,            # ID of the token aligned to the node in the original sentence
                "variable": "p2",           # the amr node, i.e. its variable name
                "concept": "pasta",         # the label / concept of the AMR node
                "token": "pasta",           # the token itself
                "amr_ids": ["baked_ziti_1_instr2_1", "baked_ziti_1_instr2_0"]   # list of the IDs of all AMRs in which it occurs
                }
}
"""


def find_post_splitting_coref(recipe_action_amrs: List[nx.Graph], recipe_text: List[str], shift=True) -> Dict:
    """
    Extract the information about co-referring AMR nodes that does not original from co-references within the original
    text but from separating the sentence-level amrs into action-level amrs
    E.g. if the AMR for "Cut the onions and add to the sauce" gets split into the action-level AMRs
    for "Cut the onions." and "Add the onions to the sauce." then a coreference for the two "onion" nodes in the
    two AMRs should be created
    :param recipe_action_amrs: list of the action-level amr graphs for the recipe
    :param recipe_text: list of the tokens of the recipe text
    :param shift: whether token IDs in the recipe_coref_data dict start at 0 and therefore need to be shifted by 1
    :return: a dictionary with amr node coreference clusters arising from the splitting
             one subdict for each (enumerated) cluster, e.g.
             {'split-rel-0': {'token_id': ID of the token in the text,
                              'variable': the AMR node variable,
                              'concept': the node label,
                              'token': the token,
                              'amr_ids': list of the action-level AMR IDs for the graphs in which that node occurs},
              'split-rel-1': {...}}
    """
    splitting_coref = defaultdict()

    # group all graphs that come from the same original AMR; keys are the IDs of the sentence-level AMRs
    instruction_amr_mappings = defaultdict(list)
    for action_amr in recipe_action_amrs:
        original_sentence_id = action_amr.graph['snt_id']
        instruction_amr_mappings[original_sentence_id].append(action_amr)

    rel_id = 0      # name each "split"-coref-cluster by enumerating them
    # loop over all original sentence-level IDs and the corresponding action-level AMRs
    for orig_instr, separated_amrs in instruction_amr_mappings.items():

        if len(separated_amrs) == 1:        # no splitting happened -> no additional coreferences created
            continue

        nodes_to_corefer_ids = defaultdict(set)     # keeps track for each node that has coreferences, in which
                                                    # AMRs the coreferring nodes occur
        nodes_to_corefer_data = dict()
        for sep_amr in separated_amrs:
            for sep_amr2 in separated_amrs:
                if sep_amr.name != sep_amr2.name:
                    nodes1 = list(sep_amr.nodes)
                    nodes2 = list(sep_amr2.nodes)
                    shared_nodes = set(nodes1) & set(nodes2)    # only shared nodes can have an identical original node
                    shared_nodes = list(shared_nodes)
                    for sh in shared_nodes:

                        # add the names of the AMRs in which the node occurs
                        nodes_to_corefer_ids[sh].add(sep_amr.name)
                        nodes_to_corefer_ids[sh].add(sep_amr2.name)
                        # keep track of the data of the node, e.g. label, alignments, etc.
                        nodes_to_corefer_data[sh] = sep_amr.nodes(data=True)[sh]

        for shared_n in nodes_to_corefer_ids.keys():
            rel_id_name = f'split-rel-{rel_id}'         # create a new coreference cluster

            node_var = shared_n
            amr_ids = list(nodes_to_corefer_ids[shared_n])      # list of amrs the node occurs in

            node_label = nodes_to_corefer_data[shared_n]['label']
            token_id_str = nodes_to_corefer_data[shared_n]['alignment']     # ID of the token aligned to the node
            token_id = int(token_id_str)
            if shift:
                token = recipe_text[token_id-1]
            else:
                token = recipe_text[token_id]

            splitting_coref[rel_id_name] = {'token_id': token_id, 'variable': node_var,
                                               'concept': node_label, 'token': token, 'amr_ids': amr_ids}

            rel_id += 1

    return splitting_coref


def map_coref_to_amr(recipe_coref_data: dict, recipe_action_amrs: List[nx.Graph], shift=True) -> Tuple[Dict, Dict]:
    """

    :param recipe_coref_data: dict with the coreference information for a recipe; needs to include the following
                                key, value information at least:
                                "clusters": list with one sublist per cluster, each consisting of a sublist with the
                                            token IDs of one mention
                                "text": list of the tokens of the corresponding recipe
    :param recipe_action_amrs: list of the action-level amr graphs for the recipe
    :param shift: whether token IDs in the recipe_coref_data dict start at 0 and therefore need to be shifted by 1
    :return:
    """
    coref_token_data: dict = extract_relevant_cluster_data(recipe_coref_data, shift)
    recipe_text = recipe_coref_data['text']
    coref_splitting_data: dict = find_post_splitting_coref(recipe_action_amrs, recipe_text, True)

    joined_coref_information = dict()

    # add the information about the aligned AMR node(s) for all tokens in the text-based coreference clusters
    for coref_cluster, token_data in coref_token_data.items():
        cluster_corresponding_amr_nodes = []
        # for each mention / token span in a coref cluster
        for span in token_data['token_ids']:
            span_corresponding_amr_nodes = []
            # find the amr node aligned to each of the tokens
            for token_id in span:
                token_corresponding_amr_nodes = []      # several AMR nodes can be aligned to the same token
                for ac_amr in recipe_action_amrs:
                    node_attr_data = nx.get_node_attributes(ac_amr, 'alignment')
                    # find the node whose alignment is identical with current token
                    for amr_node in ac_amr.nodes:
                        node_aligned_token = node_attr_data[amr_node]
                        if int(node_aligned_token) == token_id:
                            current_node_data = ac_amr.nodes(data=True)[amr_node]
                            node_label = current_node_data['label']
                            amr_id = ac_amr.name
                            node_var = amr_node
                            token_corresponding_amr_nodes.append((node_label, node_var, amr_id))

                # some tokens do not have an aligned AMR node, but when skipping them we would lose information about
                # which tokens those are
                span_corresponding_amr_nodes.append(token_corresponding_amr_nodes.copy())
                # some tokens do not have an aligned AMR node, but when skipping them we would lose information about
                # which tokens those are

            if span_corresponding_amr_nodes:
                cluster_corresponding_amr_nodes.append(span_corresponding_amr_nodes.copy())

        # keep the information about coreference clusters represented by the token IDs and add the corresponding
        # amr node information
        joined_coref_information[coref_cluster] = token_data.copy()
        joined_coref_information[coref_cluster]['amr_nodes'] = cluster_corresponding_amr_nodes.copy()

    return joined_coref_information, coref_splitting_data


def extract_relevant_cluster_data(recipe_coref_data: dict, shift: bool) -> dict:
    """
    Extract the potentially shifted token IDs and the corresponding tokens from the available
    coreference data and give each coreference cluster a name based on enumerating them
    Example: if recipe_coref_data includes "clusters": [[[6, 7], [12, 12]]] and shift is True
             then returns {'rel-0':
                                    {'token_ids' [[7, 8], [13]],
                                    'tokens': [['ground', 'beef'], ['beef']]}
                            }
    :param recipe_coref_data: dict with the coreference information for a recipe; needs to include the following
                                key, value information at least:
                                "clusters": list with one sublist per cluster, each consisting of a sublist with the
                                            token IDs of one mention
                                "text": list of the tokens of the corresponding recipe
    :param shift: whether token IDs in coref data need to be shifted by 1 to match ara token IDs
    :return: dictionary, one key per cluster, value is a dictionary itself with key 'token_ids' and 'tokens'
                    'token_ids': list of lists where each list corresponds to a token span
                                 with all token IDs of the tokens included in the span
                    'tokens': the corresponding tokens from the text

    """
    coref_clusters = recipe_coref_data['predicted_clusters']
    recipe_text = recipe_coref_data['text']

    relevant_data = dict()

    for cluster_id, cluster in enumerate(coref_clusters):
        cluster_name = f'coref-rel-{cluster_id}'
        relevant_data[cluster_name] = {'token_ids': [], 'tokens': []}
        for span in cluster:
            span_start = span[0]
            span_end = span[-1]
            spanned_tokens_ids = []
            spanned_tokens = []
            for token_ind in range(span_start, span_end + 1):
                spanned_tokens.append(recipe_text[token_ind])
                if shift:
                    spanned_tokens_ids.append(token_ind + 1)
                else:
                    spanned_tokens_ids.append(token_ind)
            relevant_data[cluster_name]['token_ids'].append(spanned_tokens_ids)
            relevant_data[cluster_name]['tokens'].append(spanned_tokens)

    return relevant_data


def create_coref_amr_files(action_amr_dir, coref_file_path, output_path,  extended=False):
    """
    For all recipes in ACTION_AMR_DIR
    Requires
    :param coref_file_path:
    :param output_path:
    :param extended: whether the coreference information for the extended recipes, i.e. including the predicted
                     explicit mentions, should be extracted or the coreference information about the original texts
    :return:
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if extended:
        corpus_coreferences: dict = get_coref_clusters_extended(coref_file_path)
    else:
        corpus_coreferences: dict = get_coref_clusters_original(coref_file_path)

    for dish in os.listdir(action_amr_dir):
        Path(output_path / Path(dish)).mkdir(parents=True, exist_ok=True)
        for recipe in os.listdir(action_amr_dir / dish):
            file_name_tokens = recipe.split('_')
            recipe_name = '_'.join(file_name_tokens[:-2])

            # read the AMR file and convert to networkX graphs
            amr_graphs_penman = read_aligned_amr_file(action_amr_dir / dish / recipe)
            amr_graphs = []
            for pen_gr in amr_graphs_penman:
                amr_graphs.append(penman2networkx(pen_gr))

            # get the corresponding coreference data
            coref_data = corpus_coreferences[recipe_name]

            joined_coref_info, coref_from_splitting = map_coref_to_amr(coref_data, amr_graphs)

            with open(output_path/ dish / f'{recipe_name}_joined.jsonlines', 'w', encoding='utf-8') as f:

                for rel_id, value in joined_coref_info.items():
                    json_line = {rel_id: value}
                    json.dump(json_line, f)
                    f.write('\n')
                for rel_id, value in coref_from_splitting.items():
                    json_line = {rel_id: value}
                    json.dump(json_line, f)
                    f.write('\n')

    """

    # for testing
    action_graphs_penman = read_aligned_amr_file('../data/recipe_amrs_actions/cauliflower_mash/cauliflower_mash_0_instructions_amr.txt')
    action_graphs = []
    for pen_gr in action_graphs_penman:
        action_graphs.append(penman2networkx(pen_gr))
    coref_data = read_coref_file('../data/coref_data/cauliflower_mash/cauliflower_mash_0.jsonlines')

    res = map_coref_to_amr(coref_data, action_graphs)
    return res
    """


if __name__ == '__main__':
    #create_coref_amr_files('./ara_pronoun_merged_pred.jsonlines', JOINED_COREF_DIR, extended=False)
    create_coref_amr_files(Path('../data_ara1_explicit/explicit_action_amrs'), './ara_explicit_merged_pred.jsonlines',
                           '../data_ara1_explicit/coref_data_joined', extended=True)
    #create_coref_amr_files(Path('../data/recipe_amrs_actions'), './ara_pronoun_merged_pred.jsonlines',
    #                       '../data/coref_data_joinedtest', extended=False)


