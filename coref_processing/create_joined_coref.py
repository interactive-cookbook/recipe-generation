import json
import os
from pathlib import Path
import networkx as nx
from typing import List
from collections import defaultdict
from utils.paths import RAW_COREF_DIR, ACTION_AMR_DIR, JOINED_COREF_DIR
from graph_processing.read_graphs import read_aligned_amr_file
from amr_processing.penman_networkx_conversions import penman2networkx


def read_coref_file(coref_file: str):
    """

    :param coref_file:
    :return:
    """
    coref_dict = dict()
    with open(coref_file, "r", encoding="utf-8") as cf:
        for line in cf:
            coref_dict = json.loads(line)
            break
    return coref_dict


def find_post_splitting_coref(recipe_action_amrs: List[nx.Graph], recipe_text: List[str], shift=True):

    splitting_coref = defaultdict()
    # group all graphs that come from the same original AMR
    instruction_amr_mappings = defaultdict(list)

    for action_amr in recipe_action_amrs:
        original_sentence_id = action_amr.graph['snt_id']
        instruction_amr_mappings[original_sentence_id].append(action_amr)

    rel_id = 0
    for orig_instr, separated_amrs in instruction_amr_mappings.items():

        if len(separated_amrs) == 1:
            continue

        nodes_to_corefer_ids = defaultdict(set)
        nodes_to_corefer_data = dict()
        for sep_amr in separated_amrs:
            for sep_amr2 in separated_amrs:
                if sep_amr.name != sep_amr2.name:
                    nodes1 = list(sep_amr.nodes)
                    nodes2 = list(sep_amr2.nodes)
                    shared_nodes = set(nodes1) & set(nodes2)
                    shared_nodes = list(shared_nodes)
                    for sh in shared_nodes:

                        #assert sep_amr.nodes(data=True)[sh] == sep_amr2.nodes(data=True)[sh]
                        nodes_to_corefer_ids[sh].add(sep_amr.name)
                        nodes_to_corefer_ids[sh].add(sep_amr2.name)
                        nodes_to_corefer_data[sh] = sep_amr.nodes(data=True)[sh]

        for shared_n in nodes_to_corefer_ids.keys():
            rel_id_name = f'split-rel-{rel_id}'

            node_var = shared_n
            amr_ids = list(nodes_to_corefer_ids[shared_n])

            node_label = nodes_to_corefer_data[shared_n]['label']
            token_id_str = nodes_to_corefer_data[shared_n]['alignment']
            token_id = int(token_id_str)
            if shift:
                token = recipe_text[token_id-1]
            else:
                token = recipe_text[token_id]

            splitting_coref[rel_id_name] = {'token_id': token_id, 'variable': node_var,
                                               'concept': node_label, 'token': token, 'amr_ids': amr_ids}

            rel_id += 1

    return splitting_coref



def map_coref_to_amr(recipe_coref_data: dict, recipe_action_amrs: List[nx.Graph], shift=True):
    """

    :param recipe_coref_data:
    :param recipe_action_amrs:
    :param shift:
    :return:
    """

    coref_token_data = extract_relevant_cluster_data(recipe_coref_data, shift)
    coref_splitting_data = find_post_splitting_coref(recipe_action_amrs,
                                                     recipe_coref_data['text'],
                                                     True)

    joined_coref_information = dict()

    for coref_cluster, token_data in coref_token_data.items():
        cluster_corresponding_amr_nodes = []
        for span in token_data['token_ids']:
            span_corresponding_amr_nodes = []
            for token_id in span:
                token_corresponding_amr_nodes = []
                for ac_amr in recipe_action_amrs:
                    node_attr_data = nx.get_node_attributes(ac_amr, 'alignment')
                    for amr_node in ac_amr.nodes:
                        node_aligned_token = node_attr_data[amr_node]
                        if int(node_aligned_token) == token_id:
                            current_node_data = ac_amr.nodes(data=True)[amr_node]
                            node_label = current_node_data['label']
                            amr_id = ac_amr.name
                            node_var = amr_node
                            token_corresponding_amr_nodes.append((node_label, node_var, amr_id))

                if token_corresponding_amr_nodes:
                    span_corresponding_amr_nodes.append(token_corresponding_amr_nodes.copy())

            if span_corresponding_amr_nodes:
                cluster_corresponding_amr_nodes.append(span_corresponding_amr_nodes.copy())

        joined_coref_information[coref_cluster] = token_data.copy()
        joined_coref_information[coref_cluster]['amr_nodes'] = cluster_corresponding_amr_nodes.copy()

    return joined_coref_information, coref_splitting_data


def extract_relevant_cluster_data(recipe_coref_data: dict, shift):
    """
    Extract the potentially shifted token IDs and the corresponding tokens from the available
    coreference data and give each coreference cluster a name based on enumerating them
    Example: if recipe_coref_data includes "clusters": [[[6, 7], [12, 12]]] and shift is True
             then returns {'rel-0':
                                    {'token_ids' [[7, 8], [13]],
                                    'tokens': [['ground', 'beef'], ['beef']]}
                            }
    :param recipe_coref_data:
    :param shift:
    :return: dictionary, one key per cluster, value is a dictionary itself with key 'token_ids' and 'tokens'
                    'token_ids': list of lists where each list corresponds to a token span
                                 with all token IDs of the tokens included in the span
                    'tokens': the corresponding tokens from the text

    """
    coref_clusters = recipe_coref_data['clusters']
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


def create_coref_amr_files():
    """

    :return:
    """
    Path(JOINED_COREF_DIR).mkdir(parents=True, exist_ok=True)
    for dish in os.listdir(ACTION_AMR_DIR):
        Path(JOINED_COREF_DIR / Path(dish)).mkdir(parents=True, exist_ok=True)
        for recipe in os.listdir(ACTION_AMR_DIR / dish):
            file_name_tokens = recipe.split('_')
            recipe_name = '_'.join(file_name_tokens[:-2])

            action_graphs_penman = read_aligned_amr_file(ACTION_AMR_DIR / dish / recipe)
            action_graphs = []
            for pen_gr in action_graphs_penman:
                action_graphs.append(penman2networkx(pen_gr))

            corresponding_coref_file = recipe_name + '.jsonlines'
            coref_data = read_coref_file(RAW_COREF_DIR / dish / corresponding_coref_file)
            joined_coref_info, coref_from_splitting = map_coref_to_amr(coref_data, action_graphs)

            with open(JOINED_COREF_DIR/ dish / f'{recipe_name}_joined.jsonlines', 'w', encoding='utf-8') as f:

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



if __name__=='__main__':
    create_coref_amr_files()


