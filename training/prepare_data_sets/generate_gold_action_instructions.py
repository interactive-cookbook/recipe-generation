import os
import re
from typing import List, Dict, Set, Tuple
import networkx as nx
import penman
from collections import Counter, defaultdict
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
import spacy

from graph_processing.recipe_graph import read_graph_from_conllu
from graph_processing.read_graphs import read_aligned_amr_file
from graph_processing.graph_traversal import order_actions_df_lf
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman
from utils.paths import ACTION_AMR_DIR, ARA_DIR, SENT_AMR_DIR

"""
Functions to create gold instructions for the split AMRs using a simple string-match and rule-based approach
"""


class InstructionExtractor:

    def __init__(self,
                 split_amr: nx.Graph,
                 sentence_amr: nx.Graph,
                 other_split_amrs: List[nx.Graph],
                 shift_value: int,
                 version: int,
                 spacy_model):
        """

        :param split_amr: the action-level amr
        :param sentence_amr: the corresponding sentence-level amr
        :param other_split_amrs: list of the other split amrs that were obtained from the same sentence-level amr
        :param shift_value: the value for shifting the sentence-level token indices to document-level token indices
        :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
                    1: do not add any unaligned tokens
                    2: add unaligned token if directly adjacent to an aligned token
                    3: add contiguous spans of unaligned tokens if span boundary adjacent to an aligned token
        :param spacy_model: already loaded spacy model for English to use for POS tagging
        """
        self.split_amr = split_amr                  # the action-level AMR
        self.sentence_amr = sentence_amr            # corresponding sentence-level AMR
        self.other_split_amrs = other_split_amrs    # the other action-level AMRs from the same sentence-level AMR
        self.shift_value = shift_value              # value to shift sentence-level to document-level token indices
        self.version = version                      # version for adding back unaligned tokens

        self.pos_tagger = spacy_model
        self.stemmer = PorterStemmer()              # stemmer to use for converting participles to imperative verbs

        self.original_sentence: str = split_amr.graph['snt']                        # original sentence instruction
        self.orig_snt_tokenized: List[str] = self.original_sentence.split(' ')      # tokenized original sentence
        #self.orig_snt_tagged: List[Tuple[str, str]] = nltk.pos_tag(self.orig_snt_tokenized)   # tagged original sentence
        self.orig_snt_tagged = self.tag_sentence(self.original_sentence)

        self.modified_snt_tokenized: List[str] = []     # tokens of original sentence with modified actions replaced with stemmed version
        self.modified_inds: List[int] = []              # sentence-level indices of the tokens that are modified
        self.modified_inds_others: List[int] = []       # sentence-level indices of tokens modified for the other_split_amrs
        self.action_root_inds: List[int] = []           # sentence-level indices of tokens that are actions and aligned to root in current split amr
        self.potential_tokens: List[int] = []           # sentence-level indices of tokens that might be used for the new instructions

        self.inds_to_add: Set[int] = set()              # sentence-level indices of tokens that will make up the extracted instruction
        self.final_tokens: List[str] = []               # tokenized extracted instruction

    def tag_sentence(self, sentence: str):
        processed_sent = self.pos_tagger(sentence)
        token_tag_pairs = [(token.text, token.tag_) for token in processed_sent]
        return token_tag_pairs

    def create_gold_sentence(self):
        """
        Create the gold sentence for split_amr graph
        :return: the substring of the original sentence that was extracted as the gold instruction for the split_amr
        """
        for token_ind, token in enumerate(self.orig_snt_tokenized):
            # Important! token IDs in the alignments were shifted to match the token IDs in the conllu files
            # -> need to be re-shifted here
            shifted_token_ind = token_ind + self.shift_value

            # check whether the token has an alignment to any node in the amr and if yes, whether it corresponds to an action
            in_current_amr, is_action, is_root = self.check_token_alignment(str(shifted_token_ind), self.split_amr)
            if is_action and is_root:
                self.action_root_inds.append(token_ind)

            # check whether the token had an alignment in original AMR or is not represented in AMR in general
            in_orig_amr, _, _ = self.check_token_alignment(str(shifted_token_ind), self.sentence_amr)

            # potentially modify token: Remove -ed ending from actions that are participles, only if it is the root
            #  e.g. "shredded cheese" should become "shred cheese"
            if is_action and is_root:
                token = self.fix_imperative_form(token_ind, True)

            self.modified_snt_tokenized.append(token)

            # decide whether token should be added, should potentially be added or not be added
            self.decide_token_relevance(token_ind, in_current_amr, in_orig_amr)

            # Check whether the token would have been modified for another AMR:
            for other_amr in self.other_split_amrs:
                in_other_amr, is_action_other, is_root_other = self.check_token_alignment(str(shifted_token_ind), other_amr)
                if in_other_amr and is_action_other and is_root_other:
                    _ = self.fix_imperative_form(token_ind, False)

        assert len(self.modified_inds) < 2          # only the main action node should be modified if at all
        assert len(self.modified_snt_tokenized) == len(self.orig_snt_tokenized) # stemming should be the only change

        # run the function that decides about adding back unaligned tokens and that adds their indices to
        # self.inds_to_add if they should be added
        self.add_unaligend_tokens()

        # add the tokens chosen for the new instruction in their original order
        for tok_ind in range(len(self.modified_snt_tokenized)):
            if tok_ind in self.inds_to_add:
                self.final_tokens.append(self.modified_snt_tokenized[tok_ind])

        # Fix the sentence ordering for cases such as "a toothpick insert into the centre"
        self.fix_sentence_order()

        # Modify sentence to start with upper-case character and end with sentence-final punctuation
        self.fix_sentence_start_and_end()

        with open('./all_new_sentences.txt', 'a', encoding='utf-8') as f:
            f.write(f'{self.split_amr.graph["id"]}\t{" ".join(self.final_tokens)}\t{self.orig_snt_tagged}\n')

        return ' '.join(self.final_tokens)

    def fix_imperative_form(self, token_ind: int, current: bool) -> str:
        """
        Stems a token if its original POS tag is JJ, VBN, VBG, VBD or NN and the token ends with 'ed' or 'ing'
        -> stems most of the actions verbs that were originally participles in order to get the imperative form
        :param token_ind: sentence-level index of the token
        :param current: whether the token_ind belongs to the current amr or another amr
                        i.e. whether the token_ind gets added to self.modified_inds or self.modified_others
        :return: the stemmed token if conditions are fulfilled, otherwise the original token
        """
        token, pos_tag = self.orig_snt_tagged[token_ind]
        if pos_tag in ['JJ', 'VBN', 'VBG', 'VBD', 'NN'] and (token.endswith('ed') or token.endswith('ing')):
            token = self.stemmer.stem(token)
            if current:
                self.modified_inds.append(token_ind)
            else:
                self.modified_inds_others.append(token_ind)
        return token

    def decide_token_relevance(self, token_ind, in_current_amr, in_orig_amr):
        """
        If token is represented in current amr then add it to the sentence, i.e. add to self.inds_to_add
        If token was in original amr but is not in current amr then it is either part of a different split amr
        or was removed during splitting, e.g. 'and', 'before' and should not be added
        If it was not in the original amr and is not in the current amr then the token is not represented in amr and
        how to deal with it will be decided in the next steps, i.e. add to self.potential_tokens
        :param token_ind: sentence-level index of token to decide about
        :param in_current_amr: whether token is represented in current action-level amr
        :param in_orig_amr: whether token is represented in original sentence-level amr
        :return: no return value but changes class attributes
        """
        if in_current_amr:
            self.inds_to_add.add(token_ind)
        elif not in_current_amr and in_orig_amr:
            pass
        else:
            self.potential_tokens.append(token_ind)

    def fix_sentence_order(self):
        """
        Reorders self.final_tokens if an NP like phrase is at the beginning of the sentence, followed by a verb that
        is an action and original root
        Then the verb is moved to the sentence beginning because in imperative sentences the verb should
        be before its direct object
        :return:
        """
        orig_token_ids = list(self.inds_to_add)         # the original indices of the tokens in self.final_tokens
        orig_token_ids.sort()                           # sort to get original token order

        extracted_pos_tags = []                         # the POS tags corresponding to the tokens in self.final_tokens
        for orig_id in orig_token_ids:
            extracted_pos_tags.append(self.orig_snt_tagged[orig_id][1])
        extracted_pos_seq = ' '.join(extracted_pos_tags)    # convert to string for re matching

        # Potential POS-tag patterns:
        # DT / PRP$ (0 or 1) JJ (any) NN / NNS (one)
        # above pattern repeated with CC between
        # followed by VB, VBP or a modified token
        pos_reg_full = r'^(PDT )?(DT |PRP$ )?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+((, |CC )*(DT |PRP $)?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+)*(, |CC )*(VB|VBP|VBD|VBZ){1}( |$)'
        # TODO: decide what to do?
        # If including RB then RB should be "then"
        pos_reg_full2 = r'^(PDT )?(DT |PRP$ )?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+((, |CC )*(DT |PRP $)?(JJ |VBD |VBG )?(NN |NNS |NNP |NNPS )+)*(, |CC )*(VB|VBP|VBD|VBZ){1}( |$)'
        # If verb was originally a participle (i.e. got stemmed for the extracted sentence) then the original
        # POS tag will not be VB or VBP, so check for NP-like pattern first and then check if the next token was
        # such a stemmed token
        pos_reg_mod = r'^(PDT )?(DT |PRP$ )?(JJ )?(NN |NNS |NNP |NNPS )+((, |CC ){1}(DT |PRP $)?(JJ )?(NN |NNS |NNP |NNPS )+)*(, |CC )?'

        verb_index = None

        search_matching_pos = re.search(pos_reg_full, extracted_pos_seq)
        search_matching_pos2 = re.search(pos_reg_full2, extracted_pos_seq)
        if search_matching_pos2 and not search_matching_pos:
            matching_pos_list2 = search_matching_pos2.group().split(' ')
            print(f'{search_matching_pos2}\t{self.orig_snt_tagged}\t{extracted_pos_seq}')

        if search_matching_pos:
            matching_pos = search_matching_pos.group()  # if emtpy space after last matching POS then this will cause issues
            matching_pos = matching_pos.strip()
            matching_pos_list = matching_pos.split(' ')

            potential_verb_index = len(matching_pos_list) - 1       # is relative to extracted sentence
            potential_verb_index_original = orig_token_ids[potential_verb_index]    # relative to orig sentence
            if potential_verb_index_original in self.action_root_inds:
                verb_index = potential_verb_index
        else:
            search_matching_pos_mod = re.search(pos_reg_mod, extracted_pos_seq)
            if search_matching_pos_mod:
                matching_pos = search_matching_pos_mod.group()
                matching_pos = matching_pos.strip()
                matching_pos_list = matching_pos.split(' ')
                # last noun index is len(m_p_l) - 1 -> verb should be next
                potential_verb_index = len(matching_pos_list)  # is relative to extracted sentence

                if len(matching_pos_list) != len(extracted_pos_tags):  # otherwise potential_verb_index is out of index
                    # get corresponding index in the original sentence because modified_action_inds and
                    # action_root_inds are relative to orig sent
                    potential_verb_index_original = orig_token_ids[potential_verb_index]

                    if potential_verb_index_original in self.modified_inds:  # then it is an action verb but originally had different POS
                        verb_index = potential_verb_index
                    elif potential_verb_index_original in self.action_root_inds and \
                            (extracted_pos_tags[potential_verb_index] == "NN" or extracted_pos_tags[
                                potential_verb_index] == "NNS"):
                        verb_index = potential_verb_index

        if verb_index:
            verb_token = self.final_tokens.pop(verb_index)
            self.final_tokens.insert(0, verb_token)
            #print(' '.join(self.final_tokens))

    def fix_sentence_start_and_end(self):
        """
        Remove punctuation at the sentence beginning, adds / replaces end of sentence to get sentence final punctuation and
        removes lonely 'and' at the end of the sentence
        Additionally, fix other punctuation issues
        - ( sentence ) -> remove brackets
        - tokens () tokens -> remove brackets
        - "," "," -> remove one comma
        :return: returns nothing but modifies self.final_tokens directly
        """
        punctuation_to_remove = []
        for t_ind, t in enumerate(self.final_tokens):
            if t == ',':
                try:
                    next_token = self.final_tokens[t_ind + 1]
                    if next_token == ',':
                        punctuation_to_remove.append(t_ind)
                except IndexError:
                    break
            elif t == '(':
                try:
                    next_token = self.final_tokens[t_ind + 1]
                    if next_token == ')':
                        punctuation_to_remove.append(t_ind)
                except IndexError:
                    punctuation_to_remove.append(t_ind)
        for punct_ind in punctuation_to_remove:
            self.final_tokens.pop(punct_ind)

        while True:
            if self.final_tokens[0] == '(' and self.final_tokens[-1] == ')':
                self.final_tokens = self.final_tokens[1:-1]
                continue
            if self.final_tokens[-1] == 'and':
                self.final_tokens = self.final_tokens[:-1]
                continue
            if self.final_tokens[0] in [',', ';', '-', ')', '.']:  # maybe extend
                self.final_tokens = self.final_tokens[1:]
                continue
            if self.final_tokens[-1] in [',', '-', '(', ';', ':']:  # maybe extend list
                self.final_tokens = self.final_tokens[:-1]
                continue
            if self.final_tokens[-1] not in ['.', '!', '?']:
                self.final_tokens.append('.')
                continue
            break

        if self.final_tokens[0][0].isalpha():
            self.final_tokens[0] = self.final_tokens[0][0].upper() + self.final_tokens[0][1:]

    def add_unaligend_tokens(self):
        """
        Add back tokens, i.e. add their ids to self.inds_to_add, under specific conditions
        :return: returns nothing but changes self.ind_to_add
        """
        # add only direct adjacent tokens
        if self.version == 2:
            inds_to_add_unchanged = self.inds_to_add.copy()
            for pt_ind in self.potential_tokens:
                _, pt_pos_tag = self.orig_snt_tagged[pt_ind]
                add_pt = self.decide_about_unaligned_token(pt_ind, pt_pos_tag, inds_to_add_unchanged)
                if add_pt:
                    self.inds_to_add.add(pt_ind)

        # Add the unaligned tokens if they are adjacent to tokens that get added
        # do this in both directions to be able to add e.g. 'In a' at the beginning of an instruction
        elif self.version == 3:
            for pt_ind in self.potential_tokens:
                _, pt_pos_tag = self.orig_snt_tagged[pt_ind]
                add_pt = self.decide_about_unaligned_token(pt_ind, pt_pos_tag, self.inds_to_add)
                if add_pt:
                    self.inds_to_add.add(pt_ind)

            potential_tokens_rev = self.potential_tokens.copy()
            potential_tokens_rev.reverse()
            for pt_ind in self.potential_tokens:
                _, pt_pos_tag = self.orig_snt_tagged[pt_ind]
                add_pt = self.decide_about_unaligned_token(pt_ind, pt_pos_tag, self.inds_to_add)
                if add_pt:
                    self.inds_to_add.add(pt_ind)

    def decide_about_unaligned_token(self, pt_ind, pt_pos_tag, inds_to_add) -> bool:
        """
        Decides for unaligned tokens whether to add them or node based on the following rules
        - add only if token is adjacent to an added token
        - do not add prepositions or determiners back that preceed a now modified action, if that action is included
          in order to avoid instructions such as "with shred mozzarella cheese ."
        - but add prepositions or determiners back that preceed a now modified action, if the action is not included
          in order to add e.g. "with" to "Top with [shredded] mozzarella cheese"
        - do not add tokens with tag 'IN' if the next token is not added
        - do not add tokens with tag 'CC' if either previous or next token are not added
        - do not add tokens with tag 'DT' if next token is not added
        :param pt_ind: index of the unaligned token to decide about
        :param pt_pos_tag: pos tag of the unaligned token
        :param inds_to_add: the set of inds to add that should be considered for the decision
        :return: whether to add the token or not
        """
        to_add = False
        if pt_ind + 1 in inds_to_add or pt_ind - 1 in inds_to_add:

            if pt_pos_tag == 'CC' and not (pt_ind + 1 in inds_to_add and pt_ind - 1 in inds_to_add):
                to_add = False
            elif pt_pos_tag == 'IN' and not (pt_ind + 1 in inds_to_add):
                if pt_ind + 1 in self.modified_inds_others and pt_ind + 1 not in inds_to_add:
                    to_add = True
                else:
                    to_add = False
            elif pt_pos_tag == 'DT' and not (pt_ind + 1 in inds_to_add):
                if pt_ind + 1 in self.modified_inds_others and pt_ind + 1 not in inds_to_add:
                    to_add = True
                else:
                    to_add = False
            elif pt_ind + 1 in self.modified_inds and pt_ind + 1 in inds_to_add:
                to_add = False
            elif pt_ind + 1 in self.modified_inds_others and pt_ind + 1 not in inds_to_add:
                to_add = True
            else:
                to_add = True

        # for cases such as [Preposition Determiner ParticipleAction Noun], e.g. "with the shredded mozzarella"
        elif pt_ind + 2 in inds_to_add and pt_ind + 1 in self.modified_inds_others and pt_ind + 1 not in inds_to_add:
            if pt_pos_tag == 'DT' or pt_pos_tag == ',':
                to_add = True

        return to_add

    def check_token_alignment(self, token_index: str, graph: nx.Graph) -> (bool, bool, bool):
        """
        Determines whether a specific token is aligned to any node in the input amr graph
        :param token_index: the (document-level) index of the token
        :param graph: an amr graph
        :return: tuple(has_alignment, is action)
                 has_alignment: True if the token is aligned to one of the amr nodes, False otherwise
                 is_action: True if the token is an action, i.e. belongs to an action-aligned amr node; False otherwise
                            If no action alignment data is available then False is returned
                 is_root: True if the token is the root of the amr; False otherwise
        """
        try:
            action_aligned_nodes = graph.graph['alignments']
        except KeyError:
            action_aligned_nodes = []  # the files of the original graphs do not contain the 'alignments' meta data

        is_root = False
        has_alignment = False
        is_action = False
        for node, node_attributes in graph.nodes(data=True):
            if node_attributes['alignment'] == token_index:
                has_alignment = True
                if node in action_aligned_nodes:
                    is_action = True
                if node == graph.graph['root']:
                    is_root = True
            try:
                amr_node_attr = node_attributes['attr']
                for attr_dict in amr_node_attr:
                    if attr_dict['alignment'] == token_index and attr_dict['target'] != 'imperative':
                        has_alignment = True
                        # no need to check for is_action because amr attributes are no individual nodes
            except KeyError:
                continue

        return has_alignment, is_action, is_root


class RecipeExtractor:

    def __init__(self,
                 recipe_amrs: List[nx.Graph],
                 orig_amrs: Dict,
                 action_graph: nx.DiGraph,
                 spacy_model,
                 version: int = 3,
                 text_only: bool = False,
                 order_ac_graph: bool = False):
        """

        :param recipe_amrs: list of the action-level amrs
        :param orig_amrs: dict with the corresponding sentence-level AMRs
                          keys: the original amr/instruction ID; values: the corresponding graph (networkX Graph)
        :param action_graph: the action graph for the recipe
        :param spacy_model:
        :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
        :param text_only: if only text is relevant then set to true and '\t was changed' gets appended to each extracted
                          sentence
        :param order_ac_graph: if set to True then all action-level AMRs get ordered based on a topological order
                               of the action graph nodes
                               if False, then the main order of the original recipe is kept and only the split amrs
                               derived from the same sentence-level AMR are ordered based on the action graph
        """
        self.recipe_amrs = recipe_amrs
        self.orig_amrs = orig_amrs
        self.action_graph = action_graph
        self.version = version
        self.text_only = text_only
        self.order_ac_graph = order_ac_graph
        self.spacy_model = spacy_model

        self.orig_pos2amr = defaultdict(list)       # key: n, value: [gr1, gr2] means that gr1 and gr2 are from the
                                                    # original instruction at the nth position in the original recipe
        self.group_amrs_by_orig_position()

    def create_gold_recipe(self):

        self.create_gold_instructions()     # extracts the gold instructions for each amr and adds as new 'snt' metadata
        self.create_instruction_order()     # brings the amrs into an appropriate order

        return self.recipe_amrs

    def create_gold_instructions(self):
        """
        Creates for each amr in self.recipe_amrs and instruction based on the original instruction and the node-to-token
        alignments and updates the 'snt' metadata of the amr graph object
        :return:
        """
        shift_value = 0     # needed to be able to shift the per-sentence token ids to the document-level token ids
        prev_sent_len = 1
        already_shifted = []

        for amr in self.recipe_amrs:
            original_id = amr.graph['snt_id']
            post_split_id = amr.graph['id']
            original_sentence = amr.graph['snt']

            # only shift for new sentences, otherwise sentences for which AMRs were split contribute several times to the shift value
            if original_id not in already_shifted:
                shift_value += prev_sent_len
                already_shifted.append(original_id)

            if original_id == post_split_id:  # then the AMR was not split -> keep original instruction
                gold_sentence = original_sentence
            else:
                orig_snt_amr = self.orig_amrs[original_id]
                assert original_sentence == orig_snt_amr.graph['snt']
                # also get other AMRs derived from the current original AMR
                orig_amr_pos = int(original_id.split('instr')[-1])
                others = [gr for gr in self.orig_pos2amr[orig_amr_pos] if gr != amr]

                instruction_creator = InstructionExtractor(split_amr=amr,
                                                           sentence_amr=orig_snt_amr,
                                                           other_split_amrs=others,
                                                           shift_value=shift_value,
                                                           version=self.version,
                                                           spacy_model=self.spacy_model)
                gold_sentence = instruction_creator.create_gold_sentence()

                if self.text_only:
                    gold_sentence += '\t was changed'
            # change sentence metadata
            amr.graph['snt'] = gold_sentence
            prev_sent_len = len(original_sentence.split(' '))

    def create_instruction_order(self):
        """
        Re-orders the self.recipe_amrs
        either according to the action graph or re-orders only the instructiosn that were split
        :return:
        """
        ordered_amrs = []

        instr_amr_pairs = list(self.orig_pos2amr.items())
        # order all amrs according to action graph
        if self.order_ac_graph:
            all_amrs = []
            for pos, amrs in instr_amr_pairs:
                all_amrs.extend(amrs)
            ordered_sep_amrs = self.order_amrs(all_amrs)
            ordered_amrs.extend(ordered_sep_amrs)
        # keep original order of instructions and only order separated amrs for the same instruction relative to each
        # other according to action graph
        else:
            instr_amr_pairs.sort(key=lambda x: x[0])
            for pos, amrs in instr_amr_pairs:
                if len(amrs) == 1:
                    ordered_amrs.extend(amrs)
                else:
                    ordered_sep_amrs = self.order_amrs(amrs)
                    ordered_amrs.extend(ordered_sep_amrs)

        self.recipe_amrs = ordered_amrs

    def order_amrs(self, amrs_to_order):
        """
        Order the amrs based on the corresponding action nodes in the action graph
        Uses the function order_actions_df_lf from graph_processing.graph_traversal.py; could be replaced
        with any other ordering / traversal function
        :param amrs_to_order: list of AMR graphs (networdkX Graph) that should be ordered
        :return: list of the same AMR graphs, ordered based on the action graph
            """
        ordered_amrs = []
        action_node2amr = dict()
        for amr in amrs_to_order:
            action_aligned_amr_nodes = amr.graph['alignments']
            for amr_node in action_aligned_amr_nodes:
                action_node = nx.get_node_attributes(amr, 'alignment')[amr_node]
                action_node2amr[action_node] = amr

        partially_sorted_action_nodes = order_actions_df_lf(self.action_graph)

        for ac_n in partially_sorted_action_nodes:
            if ac_n in action_node2amr.keys():
                amr_to_add = action_node2amr[ac_n]
                if amr_to_add not in ordered_amrs:
                    ordered_amrs.append(amr_to_add)

        return ordered_amrs

    def group_amrs_by_orig_position(self):
        """
        Group all amrs that come from the same original instruction and associate them  with the original position
        of the instruction in the recipe
        :return:
        """
        for amr in self.recipe_amrs:
            original_id = amr.graph['snt_id']
            original_position = original_id.split('instr')[-1]
            original_position = int(original_position)
            self.orig_pos2amr[original_position].append(amr)


def create_gold_corpus(action_amr_corpus: str,
                       sentence_amr_corpus: str,
                       action_graph_corpus: str,
                       gold_corpus_dir: str,
                       version: int = 3,
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
    count = 0
    Path(gold_corpus_dir).mkdir(exist_ok=True, parents=True)
    spacy_model = spacy.load('en_core_web_sm')

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
            gold_recipe_creator = RecipeExtractor(recipe_amrs=recipe_amrs,
                                                  orig_amrs=orig_amrs,
                                                  action_graph=recipe_action_graph,
                                                  spacy_model=spacy_model,
                                                  version=version,
                                                  text_only=text_only)
            new_ordered_recipe_amrs = gold_recipe_creator.create_gold_recipe()

            # Create the new output files
            if text_only:
                with open(os.path.join(gold_corpus_dir, dish, f'{recipe_name}_gold_text.txt'), 'w',
                          encoding='utf-8') as f:
                    for amr_gr in new_ordered_recipe_amrs:
                        new_sent = amr_gr.graph['snt']
                        f.write(f'{new_sent}\n')
                        count += 1
            else:
                with open(os.path.join(gold_corpus_dir, dish, f'{recipe_name}_gold.txt'), 'w', encoding='utf-8') as f:
                    for amr_gr in new_ordered_recipe_amrs:
                        penman_amr_gr = networkx2penman(amr_gr)
                        amr_str = penman.encode(penman_amr_gr)
                        f.write(f'{amr_str}\n\n')
                        count += 1
    print(count)




if __name__=='__main__':
    #create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_sentences_version_2', 2, True)
    #create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_sentences_version_3', 3, True)

    #create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_amr_sentences_version_2', 2)
    #create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/ara1_amr_graphs', 3)
    create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_sentences_ara1_t', 3, True)


