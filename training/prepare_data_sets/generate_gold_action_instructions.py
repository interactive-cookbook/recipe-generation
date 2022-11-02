from typing import List, Dict
import networkx as nx
import os
from graph_processing.recipe_graph import read_graph_from_conllu
from graph_processing.read_graphs import read_aligned_amr_file
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman
from utils.paths import ACTION_AMR_DIR, ARA_DIR, SENT_AMR_DIR
from collections import Counter, defaultdict
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import penman

"""
Functions to create gold instructions for the split AMRs using a simple string-match and rule-based approach
"""


def create_gold_sentence(split_amr: nx.Graph, sentence_amr: nx.Graph, shift_value: int, version: int) -> str:
    """

    :param split_amr: the action-level amr
    :param sentence_amr: the corresponding sentence-level amr
    :param shift_value: the value for shifting the sentence-level token indices to document-level token indices
    :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
                    1: do not add any unaligned tokens
                    2: add unaligned token if directly adjacent to an aligned token
                    3: add contiguous spans of unaligned tokens if span boundary adjacent to an aligned token
    :return: the substring of the original sentence that was extracted as the gold instruction for the split_amr
    """
    original_sentence = split_amr.graph['snt']
    orig_snt_tokenized = original_sentence.split(' ')

    modified_snt_tokenized = []
    stemmer = PorterStemmer()       # a bit better than simply removing -ed
    modified_inds = []

    inds_to_add = set()     # the sentence-level indices of the tokens to use for the new instruction sentence
    potential_tokens = []   # a list of sentence-level indices of tokens that might be used for the new instructions

    # Important! token IDs in the alignments were shifted to match the token IDs in the conllu files
    # -> need to be re-shifted here
    for token_ind, token in enumerate(orig_snt_tokenized):
        shifted_token_ind = token_ind + shift_value

        # check whether the token has an alignment to any node in the amr and if yes, whether it corresponds to an action
        in_current_amr, is_action = token_has_alignment(str(shifted_token_ind), split_amr)
        # check whether the token had an alignment in original AMR or is not represented in AMR in general
        in_orig_amr, _ = token_has_alignment(str(shifted_token_ind), sentence_amr)

        # Remove -ed ending from actions that are participles, e.g. "shredded cheese" should become "shred cheese"
        if is_action:
            token, pos_tag = nltk.pos_tag(orig_snt_tokenized)[token_ind]
            # decide whether to keep VBG, 0 occurrences in ARA 1
            if pos_tag == 'JJ' or pos_tag == 'VBN' or pos_tag == 'VBG':
                if token.endswith('ed'):
                    modified_inds.append(token_ind)
                    orig_token = token
                    token = stemmer.stem(token)
                    # print(f'{orig_token} ({pos_tag}) became {token}')

        modified_snt_tokenized.append(token)

        # if token is represented in current amr then add it to the sentence
        # if token was in original amr but is not in current amr then it is either part of a different split amr
        # or was removed during splitting, e.g. 'and', 'before' and should not be added
        # if it was not in the original amr and is not in the current amr then the token is not represented in amr and
        # how to deal with it will be decided in the next steps
        if in_current_amr:
            inds_to_add.add(token_ind)
        elif not in_current_amr and in_orig_amr:
            continue
        else:
            potential_tokens.append(token_ind)

    assert len(modified_inds) < 2

    # add only direct adjacent tokens
    if version == 2:
        unmodified_inds_to_add = inds_to_add.copy()
        for pt_ind in potential_tokens:
            if pt_ind + 1 in unmodified_inds_to_add or pt_ind - 1 in unmodified_inds_to_add:
                # do not add prepositions or determiners back that preceed a now modified action, in order
                # to avoid instructions such as "with shred mozzarella cheese ."
                if pt_ind + 1 not in modified_inds:
                    inds_to_add.add(pt_ind)

    # Add the unaligned tokens if they are adjacent to tokens that get added
    # do this in both directions to be able to add e.g. 'In a' at the beginning of an instruction
    elif version == 3:
        for pt_ind in potential_tokens:
            if pt_ind + 1 in inds_to_add or pt_ind - 1 in inds_to_add:
                if pt_ind + 1 not in modified_inds:
                    inds_to_add.add(pt_ind)
        potential_tokens.reverse()
        for pt_ind in potential_tokens:
            if pt_ind + 1 in inds_to_add or pt_ind - 1 in inds_to_add:
                if pt_ind + 1 not in modified_inds:
                    inds_to_add.add(pt_ind)

    assert len(modified_snt_tokenized) == len(orig_snt_tokenized)   # I only want to convert the one main action to imperative form

    # add all chosen tokens in their original order
    new_gold_sent = []
    for tok_i in range(len(modified_snt_tokenized)):
        if tok_i in inds_to_add:
            new_gold_sent.append(modified_snt_tokenized[tok_i])

    # remove punctuation at sentence beginning, add / replace to get sentence final punctuation and
    # remove lonely 'and' at sentence end
    if new_gold_sent[-1] == 'and':
        new_gold_sent = new_gold_sent[:-1]
    if new_gold_sent[0] in [',', ';', '-', ')', '.']:        # maybe extend
        new_gold_sent = new_gold_sent[1:]
    if new_gold_sent[-1] in [',', '-', '(', ';', ':']:      # maybe extend list
        new_gold_sent = new_gold_sent[:-1]
    if new_gold_sent[-1] not in ['.', '!', '?']:
        new_gold_sent.append('.')

    new_gold_sent = ' '.join(new_gold_sent)
    #print(new_gold_sent)
    return new_gold_sent


def token_has_alignment(token_index: str, graph: nx.Graph) -> (bool, bool):
    """
    Determines whether a specific token is aligned to any node in a graph
    :param token_index: the (document-level) index of the token
    :param graph: an amr graph
    :return: tuple(has_alignment, is action)
            has_alignment: True if the token is aligned to one of the amr nodes, False otherwise
            is_action: True if the token is an action, i.e. belongs to an action-aligned amr node; False otherwise
                        If not action alignment data is available then False is returned
    """
    try:
        action_aligned_nodes = graph.graph['alignments']
    except KeyError:
        action_aligned_nodes = []       # the files of the original graphs do not contain the 'alignments' meta data

    has_alignment = False
    is_action = False
    for node, node_attributes in graph.nodes(data=True):
        if node_attributes['alignment'] == token_index:
            has_alignment = True
            if node in action_aligned_nodes:
                is_action = True
        try:
            amr_node_attr = node_attributes['attr']
            for attr_dict in amr_node_attr:
                if attr_dict['alignment'] == token_index and attr_dict['target'] != 'imperative':
                    has_alignment = True
                    # no need to check for is_action because amr attributes are no individual nodes
        except KeyError:
            continue

    return has_alignment, is_action


def create_gold_recipe(recipe_amrs: List[nx.Graph],
                       orig_amrs: Dict,
                       action_graph: nx.Graph,
                       version: int,
                       order_ac_graph: bool = False):
    """

    :param recipe_amrs: list of the action-level amrs
    :param orig_amrs: dict with the corresponding sentence-level AMRs
                      keys: the original amr/instruction ID; values: the corresponding graph (networkX Graph)
    :param action_graph: the action graph for the recipe
    :param order_ac_graph: if set to True then all action-level AMRs get ordered based on a topological order
                           of the action graph nodes
                           if False, then the main order of the original recipe is kept and only the split amrs
                           derived from the same sentence-level AMR are ordered based on the action graph
    :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
    :return: list of the action-level amr with 'snt' attribute updated to the newly generated gold instructions
             and ordered as specified by order_ac_graph
    """

    orig_instr2amr = defaultdict(list)  # key: n, value: [gr1, gr2] means that gr1 and gr2 are from the original instruction
                                        # at the nth position in the original recipe
    # needed to be able to shift the per-sentence token ids to the document-level token ids
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

        if original_id == post_split_id:    # then the AMR was not split -> keep original instruction
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
    :param action_graph: action graph (networkX Graph) to base ordering on
    :param amrs_to_order: list of AMR graphs (networkX Graph) that should be ordered
    :return: list of the same AMR graphs, ordered based on the action graph
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
                       version: int,
                       text_only: bool = False):
    """

    :param action_amr_corpus: path to parent folder of the action-level AMR files
    :param sentence_amr_corpus: path to parent folder of the sentence-level AMR files
    :param action_graph_corpus: path to parent folder of the action-graph conllu files
    :param gold_corpus_dir: path to directory for the generated gold data set, gets created if not exists yet
    :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
                    1: do not add any unaligned tokens
                    2: add unaligned token if directly adjacent to an aligned token
                    3: add contiguous spans of unaligned tokens if span boundary adjacent to an aligned token
    :param text_only: if set to True then files only containing the generated sentences (but not the AMRs) are created
    :return:
    """

    Path(gold_corpus_dir).mkdir(exist_ok=True, parents=True)

    for dish in os.listdir(action_amr_corpus):
        Path(os.path.join(gold_corpus_dir, dish)).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir(os.path.join(action_amr_corpus, dish)):

            # get action-level AMRs of the recipe
            recipe_path = os.path.join(action_amr_corpus, dish, recipe)
            recipe_amrs_pen = read_aligned_amr_file(recipe_path)
            recipe_amrs = [penman2networkx(amr) for amr in recipe_amrs_pen]

            recipe_name = '_'.join(recipe.split('_')[:-2])

            # get the corresponding original sentence-level amrs
            sentence_amr_file = recipe_name + '_sentences_amr.txt'
            orig_amrs_pen = read_aligned_amr_file(os.path.join(sentence_amr_corpus, dish, sentence_amr_file))
            orig_amrs_list = [penman2networkx(amr) for amr in orig_amrs_pen]
            orig_amrs = dict()
            for orig_amr in orig_amrs_list:
                orig_id = orig_amr.graph['id']
                orig_amrs[orig_id] = orig_amr

            # get the corresponding action graph
            action_graph_file = recipe_name + '.conllu'
            action_graph_path = os.path.join(action_graph_corpus, dish, 'recipes', action_graph_file)
            recipe_action_graph = read_graph_from_conllu(action_graph_path)

            # create the gold instructions for the action-level AMRs of the current recipe
            new_ordered_recipe_amrs = create_gold_recipe(recipe_amrs, orig_amrs, recipe_action_graph, version)

            # Create the new output files
            if text_only:
                with open(os.path.join(gold_corpus_dir, dish, f'{recipe_name}_gold_text.txt'), 'w',
                          encoding='utf-8') as f:
                    for amr_gr in new_ordered_recipe_amrs:
                        new_sent = amr_gr.graph['snt']
                        f.write(f'{new_sent}\n')
            else:
                with open(os.path.join(gold_corpus_dir, dish, f'{recipe_name}_gold.txt'), 'w', encoding='utf-8') as f:
                    for amr_gr in new_ordered_recipe_amrs:
                        penman_amr_gr = networkx2penman(amr_gr)
                        amr_str = penman.encode(penman_amr_gr)
                        f.write(f'{amr_str}\n\n')




if __name__=='__main__':
    create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_sentences_version_3', 3, True)
