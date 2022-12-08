import os
import stanza
import re
from typing import List, Tuple, Dict
import networkx as nx
from pathlib import Path
from nltk.stem import PorterStemmer
from graph_processing.graph_traversal import order_actions_pf_lf_id
from coref_processing.coref_utils import get_coref_clusters_extended, get_new_orig_id_mappings

from graph_processing.recipe_graph import read_graph_from_conllu


def create_dep_baseline_corpus(instruction_dir, ara_dir, output_dir, coref_file=''):
    """

    :param instruction_dir: path to directory with the original sentences,
                            each file should contain the sentences line by line
    :param ara_dir: path to ara directory with the action graphs
    :param coref_file: path to the file with the implicit pronouns added (not explicit mentions!)
    :param output_dir: path to the output directory for the extracted instructions
    :return:
    """
    Path(output_dir).mkdir(exist_ok=True)
    nlp_model = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    coref_data_corpus = get_coref_clusters_extended(coref_file) if coref_file != '' else None

    for dish in os.listdir(ara_dir):
        Path(os.path.join(output_dir, dish)).mkdir(exist_ok=True)
        for recipe in os.listdir(os.path.join(ara_dir, dish, 'recipes')):
            # get the actions from the action graph
            action_graph = read_graph_from_conllu(os.path.join(ara_dir, dish, 'recipes', recipe))
            action_nodes = list(action_graph.nodes())

            # get the sentences
            recipe_name = '.'.join(recipe.split('.')[:-1])
            instructions_file = os.path.join(instruction_dir, dish, recipe_name+'_sentences.txt')
            sentences = []      # one sublist per sentence with the tokens
            sent_ids = []       # one sublist per sentence with the recipe-level token ids for the tokens in sentences list
            sent_actions = []   # one sublist per sentence, each containing the token ids of action tokens in that sentence
            shift_index = 1     # the value by which the list index of a token needs to be shifted to match the recipe-level token ids

            with open(instructions_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    tokens = line.strip().split(' ')
                    sentences.append(tokens)
                    shifted_ids = []
                    corresponding_actions = []
                    for ind, t in enumerate(tokens):
                        shifted_ind = ind + shift_index
                        shifted_ids.append(shifted_ind)
                        if str(shifted_ind) in action_nodes:
                            corresponding_actions.append(str(shifted_ind))
                    sent_ids.append(shifted_ids)
                    sent_actions.append(corresponding_actions)
                    shift_index = shifted_ids[-1] + 1

            recipe_coref_data = coref_data_corpus[recipe_name] if coref_data_corpus else None

            separated_sentences = split_sentences(sentences, sent_ids, sent_actions, nlp_model, action_graph, recipe_coref_data)
            with open(os.path.join(Path(output_dir), dish, f'{recipe_name}_dep_text.txt'), 'w', encoding='utf-8') as out:
                for sent in separated_sentences:
                    out.write(f'{sent}\n')


def split_sentences(sentences: List[List[str]],
                    sentence_ids: List[List[int]],
                    sentence_actions: List[List[str]],
                    nlp_model,
                    ac_graph: nx.DiGraph,
                    coref_dict: dict) -> List[str]:
    """

    :param sentences: one sublist per sentence with the corresponding tokens
    :param sentence_ids: one sublist per sentence with the sentence-level token IDs
                        i.e. the token sentences[0] has the token id sentence_ids[0] and so on
    :param sentence_actions: one sublist per sentence with the ids of the action tokens (strings)
    :param nlp_model: a loaded stanza model pipeline
    :param ac_graph:
    :param coref_dict:
    :return: list of the separated sentences
    """
    separated_sentences = []
    for instr_id, instruction in enumerate(sentences):
        if len(sentence_actions[instr_id]) <= 1:    # only one action -> not split
            separated_sentences.append(' '.join(instruction))
        else:   # more actions -> split
            # first resolve within sentence coreference if present
            if coref_dict:
                extended_sent, extended_ids, extended_acs = add_implicit_coref_mentions(coref_dict, sentence_ids[instr_id], instruction, sentence_actions[instr_id])
                sep_sents, actions = (split_sentence(extended_sent, extended_ids, list(extended_acs.values()), nlp_model))
            else:
                sep_sents, actions = (split_sentence(instruction, sentence_ids[instr_id], sentence_actions[instr_id], nlp_model))
                extended_acs = {ac: ac for ac in actions}
            sorted_ac_nodes = order_actions_pf_lf_id(ac_graph)
            for ac_node in sorted_ac_nodes:
                try:
                    shifted_ac_node = extended_acs[ac_node]
                except KeyError:
                    continue
                if shifted_ac_node in actions:
                    ac_ind = actions.index(shifted_ac_node)
                    sent = sep_sents[ac_ind]
                    separated_sentences.append(sent)
    return separated_sentences


def add_implicit_coref_mentions(coref_data: Dict[str, List], token_ids: List[int], sentence: List[str], actions: List[str]):

    orig_id2new, new_id2orig = get_new_orig_id_mappings(coref_data['original_token_id'], coref_data['token_id'])

    new_sentence = sentence.copy()
    replacements = dict()
    for cluster in coref_data['predicted_clusters']:
        relevant_spans = []
        mask = []
        for span in cluster:
            for token in range(span[0], span[1] + 1):
                orig_id = new_id2orig[token]
                if orig_id == '[MASK]':
                    if new_id2orig[token-1]+1 in token_ids and new_id2orig[token+1]+1 in token_ids:
                        mask.append(new_id2orig[token-1] + 1)   # the index in the original sentence where [MASK] is added
                    elif (new_id2orig[token-1]+1 in token_ids and not new_id2orig[token+1]+1 in token_ids) or \
                            (not new_id2orig[token - 1] + 1 in token_ids and new_id2orig[token + 1] + 1 in token_ids):
                        print("happened")
                    continue
                if orig_id + 1 in token_ids:        # needs to be +1 because coref indices start at 0
                    relevant_spans.append(span)     # gets only added if it was also originally in the sentence
        if relevant_spans and mask:
            # if both are non-empty then the sentence includes an explicit mention and an implicit argument
            explicit_ids = relevant_spans[0]
            explicit_tokens = [coref_data['text'][eid] for eid in range(explicit_ids[0], explicit_ids[1] + 1)]
            for m in mask:
                replacements[m] = explicit_tokens

    shift_value = 0
    sorted_replacements = list(replacements.items())
    sorted_replacements.sort()                          # need to start from sentence beginning on

    original_token_ids_str = [str(tid) for tid in token_ids]        # convert to string because actions are strings

    for repl_ind, repl_tokens in sorted_replacements:
        # IMPORTANT: repl_ind is relative to document!
        sentence_level_repl_ind = token_ids.index(repl_ind) + shift_value
        new_sentence = new_sentence[:sentence_level_repl_ind+1] + repl_tokens + new_sentence[sentence_level_repl_ind+1:]
        original_token_ids_str = original_token_ids_str[:sentence_level_repl_ind + 1] + ['M' for rt in repl_tokens] + original_token_ids_str[sentence_level_repl_ind+1:]
        shift_value += len(repl_tokens)

    new_token_ids = [tid for tid in range(token_ids[0], token_ids[0] + len(new_sentence))]  # need to start at the same index as token_ids
    assert len(new_token_ids) == len(new_sentence)
    new_action_ids = dict()
    for ac in actions:
        new_sentence_level_id = original_token_ids_str.index(ac)
        new_doc_level_ids = new_token_ids[new_sentence_level_id]
        new_action_ids[ac] = (str(new_doc_level_ids))

    return new_sentence, new_token_ids, new_action_ids


def split_sentence(tokens: List[str], token_ids: List[int], actions: List[str], nlp_model):
    """

    :param tokens: list of tokens of a sentence
    :param token_ids: list of the recipe-level token ids of the tokens in tokens
    :param actions: list of the action token ids (strings) of the sentence
    :param nlp_model: a loaded stanza model pipeline
    :return:
    """
    action2tokens = dict()
    processed_sentences = nlp_model(' '.join(tokens))
    proc_snt = []
    sent_shift_value = 0
    # stanza model segments some of the sentences into several sentences -> put them together again for comparability
    for sent in processed_sentences.sentences:
        for token in sent.words:
            token.id = token.id + sent_shift_value
            proc_snt.append(token)
        sent_shift_value += len(sent.words)

    # pos tags of the sentence
    pos_tags = [token.xpos for token in proc_snt]

    shifting_value = token_ids[0]   # first token is usually 0, so first shifted token id is the shift value

    # create a dependency graph
    dep_graph = create_dependency_graph(processed_sentence=proc_snt, shifting_value=shifting_value)

    modified_verb_inds = []
    sentence_level_action_inds = [int(ac) - shifting_value for ac in actions]   # check whether correct

    # for each action, find the tokens that correspond to this action by choosing all tokens that can
    # be reached from the action node without traversing an other action node and the action token itself
    # if action token itself does not have a verb pos then reduce to its lemma
    actions_int = set([int(ac) for ac in actions])
    for ac in actions:
        action_node = int(ac)
        action2tokens[ac] = []
        for token_ind, token, token_tag in zip(token_ids, tokens, pos_tags):
            if action_node == token_ind:
                processed_token = proc_snt[token_ind - shifting_value]       # needs to be relative to sentence
                # fix imperative
                if token_tag in ['JJ', 'VBN', 'VBG', 'VBD', 'NN']:
                    action_lemma = processed_token.lemma
                    if action_lemma == token and token.endswith('ing') or token.endswith('ed'):
                        action_lemma = PorterStemmer().stem(token)
                    action2tokens[ac].append((token_ind, action_lemma, token_tag))
                    modified_verb_inds.append(token_ind - shifting_value)
                    try:
                        parent_node = list(dep_graph.predecessors(token_ind))[0]
                        if parent_node not in actions_int:
                            parent_tag = nx.get_node_attributes(dep_graph, 'tag')[parent_node]
                            parent_token = nx.get_node_attributes(dep_graph, 'token')[parent_node]
                            action2tokens[ac].append((parent_node, parent_token, parent_tag))
                    except IndexError:
                        pass
                else:
                    action2tokens[ac].append((token_ind, token, token_tag))
            else:
                # look at all paths between the action token and the current token
                all_paths = list(nx.all_simple_paths(dep_graph, source=action_node, target=token_ind))
                for path in all_paths:
                    acs_on_path = actions_int.intersection(set(path))
                    # only action node on the path should be the start action node itself
                    if len(acs_on_path) == 1:
                        action2tokens[ac].append((token_ind, token, token_tag))

    separated_sentences = []
    corresponding_actions = []
    for ac, id_token_pairs in action2tokens.items():
        id_token_list = list(set(id_token_pairs))
        id_token_list.sort()        # list of tokens to include in their original order
        extracted_token_ids = [p[0] for p in id_token_list]
        extracted_tokens = [p[1] for p in id_token_list]
        extracted_token_tags = [p[2] for p in id_token_list]
        re_ordered_sentence = fix_ordering(extracted_token_ids,
                                     extracted_tokens,
                                     extracted_token_tags,
                                     modified_verb_inds,
                                     sentence_level_action_inds)
        re_ordered_sentence = fix_sentence_start_and_end(re_ordered_sentence)
        sentence = ' '.join(re_ordered_sentence)
        separated_sentences.append(sentence)
        corresponding_actions.append(ac)

    return separated_sentences, corresponding_actions


def create_dependency_graph(processed_sentence, shifting_value) -> nx.DiGraph:
    """
    create a networkx Graph object from the dependency tree information created by the stanza dependency parser
    :param processed_sentence: the output of running a dependency parsing stanza pipeline on a sentence
    :param shifting_value: the value by which the sentence-level token ids need be shifted to be recipe-level ids
    :return:
    """
    dep_graph = nx.DiGraph()
    nodes = []
    edges = []
    for token in processed_sentence:
        token_id = token.id + shifting_value - 1
        nodes.append((token_id, {'tag': token.xpos, 'token': token.text}))
        if token.head != 0:
            parent_id = token.head + shifting_value - 1
            edges.append((parent_id, token_id))

    dep_graph.add_nodes_from(nodes)
    dep_graph.add_edges_from(edges)

    return dep_graph


def fix_ordering(extracted_token_ids: List[int],
                 extracted_tokens: List[str],
                 extracted_tokens_tags: List[str],
                 modified_verb_inds: List[int],
                 snt_level_action_inds: List[int]) -> List[str]:

    orig_token_ids = extracted_token_ids  # the original indices of the tokens in self.final_tokens
    orig_token_ids.sort()  # sort to get original token order
    extracted_pos_seq = ' '.join(extracted_tokens_tags)  # convert to string for re matching

    # POS pattern for NP-like phrases at the beginning the the POS sequence followed potentially by an adverb
    # and then a verb
    pos_reg_full = r'^(PDT |PDT IN |DT IN )?(DT |PRP$ |CD )?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+?' \
                   r'((, |CC )*?(PDT |PDT IN |DT IN )?(DT |PRP$ |CD )?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+?)*?(, |CC )*?' \
                   r'(RB )*?' \
                   r'(VB|VBP|VBD|VBZ){1}( |$)'
    # If verb was originally a participle (i.e. got stemmed for the extracted sentence) then the original
    # POS tag will not be VB or VBP, so check for NP-like pattern first and then check if the next token was
    # such a stemmed token
    # two versions for this: a greedy and a non-greedy one because I cannot use the POS of the verb as the ending
    # condition
    pos_reg_mod = r'^(PDT |PDT IN |DT IN )?(DT |PRP$ |CD )?(JJ )?(NN |NNS |NNP |NNPS )+?' \
                  r'((, |CC )*?(PDT |PDT IN |DT IN )?(DT |PRP$ |CD )?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+?)*?(, |CC )?' \
                  r'(RB )*?'

    pos_reg_mod_greedy = r'^(PDT |PDT IN |DT IN )?(DT |PRP$ |CD )?(JJ )?(NN |NNS |NNP |NNPS )+' \
                         r'((, |CC )*(PDT |PDT IN |DT IN )?(DT |PRP$ |CD )?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+)*(, |CC )?' \
                         r'(RB )*'

    verb_index = None
    rb_index = None

    search_matching_pos_rb = re.search(pos_reg_full, extracted_pos_seq)
    search_matching_pos_mod_rb = re.search(pos_reg_mod, extracted_pos_seq)
    search_matching_pos_mod_greedy_rb = re.search(pos_reg_mod_greedy, extracted_pos_seq)

    if search_matching_pos_rb:
        matching_pos_list = search_matching_pos_rb.group()  # if emtpy space after last matching POS then this will cause issues
        matching_pos_list = matching_pos_list.strip()  # therefore strip
        matching_pos_list = matching_pos_list.split(' ')

        potential_verb_index = len(matching_pos_list) - 1  # is relative to extracted sentence
        potential_verb_index_original = orig_token_ids[potential_verb_index]  # relative to orig sentence
        if potential_verb_index_original in snt_level_action_inds:
            verb_index = potential_verb_index

            if 'RB' in matching_pos_list:
                potential_rb_index = len(matching_pos_list) - 2  # is relative to extracted sentence
                rb_token = extracted_tokens[potential_rb_index]
                # only re-order if the 'RB' token is a time attribute such as then, now, immediately
                if rb_token == 'then' or rb_token == 'immediately' or rb_token == 'now':
                    rb_index = potential_rb_index

    elif search_matching_pos_mod_rb:
        assert search_matching_pos_mod_greedy_rb
        current_case = "non-greedy"
        matching_pos = search_matching_pos_mod_rb.group()
        while current_case:
            matching_pos = matching_pos.strip()
            matching_pos_list = matching_pos.split(' ')
            # last noun index is len(m_p_l) - 1 -> verb should be next
            potential_verb_index = len(matching_pos_list)  # is relative to extracted sentence
            if len(matching_pos_list) != len(extracted_tokens_tags):  # otherwise potential_verb_index is out of index
                # get corresponding index in the original sentence because modified_action_inds and
                # action_root_inds are relative to orig sent
                potential_verb_index_original = orig_token_ids[potential_verb_index]
                # If it is an action verb but originally had different POS
                if potential_verb_index_original in modified_verb_inds:
                    verb_index = potential_verb_index
                # If the verb got not modified (e.g. not the ending checked for) but is an action verb with POS tag NN or NNS
                elif potential_verb_index_original in snt_level_action_inds and \
                        (extracted_tokens_tags[potential_verb_index] == "NN" or extracted_tokens_tags[
                            potential_verb_index] == "NNS"):
                    verb_index = potential_verb_index

                if verb_index and 'RB' in matching_pos_list:
                    potential_rb_index = len(matching_pos_list) - 1
                    rb_token = extracted_tokens[potential_rb_index]
                    # only re-order if the 'RB' token is a time attribute such as then, now, immediately

                    if rb_token == 'then' or rb_token == 'immediately' or rb_token == 'now':
                        rb_index = potential_rb_index

            # only if non-greedy does not find an appropriate pattern, check with greedy pattern next
            if not verb_index and current_case == "non-greedy":
                matching_pos = search_matching_pos_mod_greedy_rb.group()
                current_case = "greedy"
            else:
                break

    new_extracted_tokens = extracted_tokens.copy()
    if rb_index and verb_index:
        # verb needs to be removed first because it follows rb token and otherwise indices change
        verb_token = new_extracted_tokens.pop(verb_index)
        rb_token = new_extracted_tokens.pop(rb_index)
        new_extracted_tokens.insert(0, verb_token)
        new_extracted_tokens.insert(0, rb_token)
    elif verb_index:
        verb_token = new_extracted_tokens.pop(verb_index)
        new_extracted_tokens.insert(0, verb_token)

    return new_extracted_tokens


def fix_sentence_start_and_end(sentence_tokens):
    punctuation_to_remove = []
    brackets_open = []
    brackets_closing = []
    for t_ind, t in enumerate(sentence_tokens):
        if t == ',':
            try:
                next_token = sentence_tokens[t_ind + 1]
                if next_token == ',':
                    punctuation_to_remove.append(t_ind)
            except IndexError:
                break
        elif t == '(':
            brackets_open.append(t_ind)
        elif t == ')':
            brackets_closing.append(t_ind)
    if brackets_open and not brackets_closing:
        final_str = ' '.join(sentence_tokens)
        if not ')' in final_str:  # some brackets do not get tokenized correctly
            punctuation_to_remove.extend(brackets_open)
    elif not brackets_open and brackets_closing:
        final_str = ' '.join(sentence_tokens)
        if not '(' in final_str:
            punctuation_to_remove.extend(brackets_closing)
    elif brackets_open and brackets_closing:
        for bo in brackets_open:
            if bo + 1 in brackets_closing:
                punctuation_to_remove.append(bo)
                punctuation_to_remove.append(bo + 1)

    sentence_tokens = [ft for ind_ft, ft in enumerate(sentence_tokens) if ind_ft not in punctuation_to_remove]


    while True:
        if sentence_tokens[0] == '(' and sentence_tokens[-1] == ')':
            sentence_tokens = sentence_tokens[1:-1]
            continue
        if sentence_tokens[-1] == 'and':
            sentence_tokens = sentence_tokens[:-1]
            continue
        if sentence_tokens[0] in [',', ';', '-', ')', '.']:
            sentence_tokens = sentence_tokens[1:]
            continue
        if sentence_tokens[-1] in [',', '-', '(', ';', ':']:
            sentence_tokens = sentence_tokens[:-1]
            continue
        if sentence_tokens[-1] not in ['.', '!', '?']:
            sentence_tokens.append('.')
            continue
        break

    if sentence_tokens[0][0].isalpha():
        sentence_tokens[0] = sentence_tokens[0][0].upper() + sentence_tokens[0][1:]

    return sentence_tokens


if __name__=='__main__':

    create_dep_baseline_corpus('../../data/amr_input_data',
                               '../../data/ara1.1',
                               '../tuning_data_sets/dependency_baseline_coref',
                               '../../coref_processing/ara_pronoun_merged_pred.jsonlines'
                               )
