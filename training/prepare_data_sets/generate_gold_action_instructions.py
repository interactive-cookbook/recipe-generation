from typing import List, Dict
import networkx as nx
import os
from graph_processing.recipe_graph import read_graph_from_conllu
from graph_processing.read_graphs import read_aligned_amr_file
from amr_processing.penman_networkx_conversions import penman2networkx
from utils.paths import ACTION_AMR_DIR, ARA_DIR, SENT_AMR_DIR
from collections import Counter, defaultdict
from pathlib import Path

# take each recipe
# look at all split AMRs
# if got split then look at action graph to get ordering
# start by choosing all tokens from the original sentence that have an alignment to one of the nodes
# what about unaligned tokens?
# if they were also not part of the original sentence-level AMR then AMR does not represent them -> copy them also
# but not all of them -> need to figure out which ones; probably those that are surrounded by included tokens or
# preceed or follow it
# if they were part of the original sentence-level AMR then the node was removed, e.g. 'and', 'before', 'after' and
# the corresponding token should not get added
# then look at sentences and think about next steps, e.g. what to do with the determiners


def create_gold_sentence(split_amr: nx.Graph, sentence_amr: nx.Graph, shift_value: int, version: int) -> str:
    """

    :param split_amr:
    :param sentence_amr:
    :param shift_value:
    :param version:
    :return:
    """
    original_sentence = split_amr.graph['snt']
    orig_snt_tokenized = original_sentence.split(' ')
    inds_to_add = set()
    potential_tokens = []

    new_gold_sent = []
    # Important! token IDs in the alignments were shifted to match the token IDs in the conllu files
    # -> need to be re-shifted here
    for token_ind, token in enumerate(orig_snt_tokenized):
        shifted_token_ind = token_ind + shift_value
        if version == 4:
            in_current_amr = token_has_alignment_ignore_you(str(shifted_token_ind), split_amr)
        else:
            in_current_amr = token_has_alignment(str(shifted_token_ind), split_amr)
        in_orig_amr = token_has_alignment(str(shifted_token_ind), sentence_amr)

        # if token is represented in current amr then add it to the sentence
        # if token was in original amr but is not in current amr then it is either part of a different split amr
        # or was removed during splitting, e.g. 'and', 'before' and should not be added
        # if it was not in the original amr and is not in the current amr then the token is not represented in amr
        if in_current_amr:
            inds_to_add.add(token_ind)
        elif not in_current_amr and in_orig_amr:
            continue
        else:
            potential_tokens.append(token_ind)

    # add only direct adjacent tokens
    if version == 2:
        unmodified_inds_to_add = inds_to_add.copy()
        for pt_ind in potential_tokens:
            if pt_ind + 1 in unmodified_inds_to_add or pt_ind - 1 in unmodified_inds_to_add:
                inds_to_add.add(pt_ind)

    # Add the unaligned tokens if they are adjacent to tokens that get added
    # do this in both directions to be able to add e.g. 'In a' at the beginning of an instruction
    elif version == 3 or version == 4:
        for pt_ind in potential_tokens:
            if pt_ind + 1 in inds_to_add or pt_ind - 1 in inds_to_add:
                inds_to_add.add(pt_ind)
        potential_tokens.reverse()
        for pt_ind in potential_tokens:
            if pt_ind + 1 in inds_to_add or pt_ind - 1 in inds_to_add:
                inds_to_add.add(pt_ind)

    for tok_i in range(len(orig_snt_tokenized)):
        if tok_i in inds_to_add:
            new_gold_sent.append(orig_snt_tokenized[tok_i])

    new_gold_sent = ' '.join(new_gold_sent)
    #print(new_gold_sent)
    return new_gold_sent


def token_has_alignment(token_index: str, graph: nx.Graph) -> bool:
    """

    :param token_index:
    :param graph:
    :return:
    """
    for node, node_attributes in graph.nodes(data=True):
        if node_attributes['alignment'] == token_index:
            return True
        try:
            amr_node_attr = node_attributes['attr']
            for attr_dict in amr_node_attr:
                if attr_dict['alignment'] == token_index and attr_dict['target'] != 'imperative':
                    return True
        except KeyError:
            continue

    return False


def token_has_alignment_ignore_you(token_index: str, graph: nx.Graph) -> bool:
    """

    :param token_index:
    :param graph:
    :return:
    """
    for node, node_attributes in graph.nodes(data=True):
        if node_attributes['alignment'] == token_index and node_attributes['label'] != 'you':
            return True
        try:
            amr_node_attr = node_attributes['attr']
            for attr_dict in amr_node_attr:
                if attr_dict['alignment'] == token_index and attr_dict['target'] != 'imperative':
                    return True
        except KeyError:
            continue

    return False


def create_gold_recipe(recipe_amrs: List[nx.Graph],
                       orig_amrs: Dict,
                       action_graph: nx.Graph,
                       version: int,
                       order_ac_graph: bool = False):
    """

    :param recipe_amrs:
    :param orig_amrs:
    :param action_graph:
    :param order_ac_graph:
    :param version:
    :return:
    """

    orig_instr2amr = defaultdict(list)
    shift_value = 0
    prev_sent_len = 1
    already_shifted = []

    for amr in recipe_amrs:
        original_id = amr.graph['snt_id']
        post_split_id = amr.graph['id']
        original_sentence = amr.graph['snt']

        # only shift for new sentences, otherwise sentences for which AMRs were split contribute several times to the shift value
        if original_id not in already_shifted:
            shift_value += prev_sent_len
            already_shifted.append(original_id)

        if original_id == post_split_id:    # then the AMR was not split
            gold_sentence = original_sentence
        else:
            orig_snt_amr = orig_amrs[original_id]
            assert original_sentence == orig_snt_amr.graph['snt']
            gold_sentence = create_gold_sentence(amr, orig_snt_amr, shift_value, version)
            # change sentence metadata
            amr.graph['snt'] = gold_sentence

        original_position = original_id.split('instr')[-1]
        original_position = int(original_position)
        orig_instr2amr[original_position].append(amr)

        prev_sent_len = len(original_sentence.split(' '))

    ordered_amrs = []

    instr_amr_pairs = list(orig_instr2amr.items())

    # order all amrs according to action graph
    if order_ac_graph:
        all_amrs = []
        for pos, amrs in instr_amr_pairs:
            all_amrs.extend(amrs)
        ordered_sep_amrs = order_amrs_based_on_action_graph(action_graph, all_amrs)
        ordered_amrs.extend(ordered_sep_amrs)
    # keep original order of instructions and order separated amrs for the same instruction according to action graph
    else:
        instr_amr_pairs.sort(key=lambda x: x[0])
        for pos, amrs in instr_amr_pairs:
            if len(amrs) == 1:
                ordered_amrs.extend(amrs)
            else:
                ordered_sep_amrs = order_amrs_based_on_action_graph(action_graph, amrs)
                ordered_amrs.extend(ordered_sep_amrs)

    return ordered_amrs


def order_amrs_based_on_action_graph(action_graph: nx.Graph, amrs_to_order: List[nx.Graph]) -> List[nx.Graph]:
    """
    Order the amrs based on a topological sort for the corresponding action nodes in the action graph
    :param action_graph:
    :param amrs_to_order:
    :return:
    """
    ordered_amrs = []

    action_node2amr = dict()
    for amr in amrs_to_order:
        action_aligned_amr_nodes = amr.graph['alignments']
        for amr_node in action_aligned_amr_nodes:
            action_node = nx.get_node_attributes(amr, 'alignment')[amr_node]
            action_node2amr[action_node] = amr

    partially_sorted_action_nodes = list(nx.topological_sort(action_graph))
    for ac_n in partially_sorted_action_nodes:
        if ac_n in action_node2amr.keys():
            amr_to_add = action_node2amr[ac_n]
            if amr_to_add not in ordered_amrs:
                ordered_amrs.append(amr_to_add)

    return ordered_amrs


def create_gold_corpus(action_amr_corpus: str,
                       sentence_amr_corpus: str,
                       action_graph_corpus: str,
                       gold_corpus_dir: str,
                       version: int):
    """

    :param action_amr_corpus:
    :param sentence_amr_corpus:
    :param action_graph_corpus:
    :param gold_corpus_dir:
    :param version:
    :return:
    """

    Path(gold_corpus_dir).mkdir(exist_ok=True, parents=True)

    for dish in os.listdir(action_amr_corpus):
        Path(os.path.join(gold_corpus_dir, dish)).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir(os.path.join(action_amr_corpus, dish)):
            recipe_path = os.path.join(action_amr_corpus, dish, recipe)
            recipe_amrs_pen = read_aligned_amr_file(recipe_path)
            recipe_amrs = [penman2networkx(amr) for amr in recipe_amrs_pen]

            recipe_name = '_'.join(recipe.split('_')[:-2])

            # get the corresponding original amrs and the corresponding action graph
            sentence_amr_file = recipe_name + '_sentences_amr.txt'
            orig_amrs_pen = read_aligned_amr_file(os.path.join(sentence_amr_corpus, dish, sentence_amr_file))
            orig_amrs_list = [penman2networkx(amr) for amr in orig_amrs_pen]
            orig_amrs = dict()
            for orig_amr in orig_amrs_list:
                orig_id = orig_amr.graph['id']
                orig_amrs[orig_id] = orig_amr

            action_graph_file = recipe_name + '.conllu'
            action_graph_path = os.path.join(action_graph_corpus, dish, 'recipes', action_graph_file)
            recipe_action_graph = read_graph_from_conllu(action_graph_path)

            new_ordered_recipe_amrs = create_gold_recipe(recipe_amrs, orig_amrs, recipe_action_graph, version)
            with open(os.path.join(gold_corpus_dir, dish, recipe_name+'_text.txt'), 'w', encoding='utf-8') as f:
                for amr_gr in new_ordered_recipe_amrs:
                    new_sent = amr_gr.graph['snt']
                    f.write(f'{new_sent}\n')


if __name__=='__main__':
    create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_sentences_version4', 4)


    #with open('./unaligned_tokens_ara1.txt', 'r', encoding='utf-8') as f:
        #tokens = []
        #for line in f:
            #tokens.append(line.strip())
        #print(Counter(tokens))
