import os
import stanza
import re
from typing import List
import networkx as nx
from pathlib import Path

from graph_processing.recipe_graph import read_graph_from_conllu


# TODO: add re-ordering

def create_dep_baseline_corpus(instruction_dir, ara_dir, output_dir):
    """

    :param instruction_dir: path to directory with the original sentences,
                            each file should contain the sentences line by line
    :param ara_dir: path to ara directory with the action graphs
    :param output_dir: path to the output directory for the extracted instructions
    :return:
    """
    Path(output_dir).mkdir(exist_ok=True)
    nlp_model = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

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

            separated_sentences = split_sentences(sentences, sent_ids, sent_actions, nlp_model)
            with open(os.path.join(Path(output_dir), dish, f'{recipe_name}_dep_text.txt'), 'w', encoding='utf-8') as out:
                for sent in separated_sentences:
                    out.write(f'{sent}\n')


def split_sentences(sentences: List[List[str]],
                    sentence_ids: List[List[int]],
                    sentence_actions: List[List[str]],
                    nlp_model):
    """

    :param sentences: one sublist per sentence with the corresponding tokens
    :param sentence_ids: one sublist per sentence with the sentence-level token IDs
                        i.e. the token sentences[0] has the token id sentence_ids[0] and so on
    :param sentence_actions: one sublist per sentence with the ids of the action tokens (strings)
    :param nlp_model: a loaded stanza model pipeline
    :return: list of the separated sentences
    """
    separated_sentences = []
    for instr_id, instruction in enumerate(sentences):
        if len(sentence_actions[instr_id]) <= 1:    # only one action -> not split
            separated_sentences.append(' '.join(instruction))
        else:   # more actions -> split
            separated_sentences.extend(split_sentence(instruction, sentence_ids[instr_id], sentence_actions[instr_id],
                                       nlp_model))
    return separated_sentences


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
                    action2tokens[ac].append((token_ind, action_lemma, token_tag))
                    modified_verb_inds.append(token_ind - shifting_value)
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
        sentence = ' '.join(re_ordered_sentence)
        separated_sentences.append(sentence)

    return separated_sentences


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
        nodes.append(token_id)
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


if __name__=='__main__':

    create_dep_baseline_corpus('../../data/amr_input_data', '../../data/ara1.1', '../tuning_data_sets/dependency_baseline2')
