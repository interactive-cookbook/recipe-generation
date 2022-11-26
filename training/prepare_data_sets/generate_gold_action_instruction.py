import re
from typing import List, Set
import networkx as nx

from nltk.stem import PorterStemmer

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
                 nlp_model):
        """

        :param split_amr: the action-level amr
        :param sentence_amr: the corresponding sentence-level amr
        :param other_split_amrs: list of the other split amrs that were obtained from the same sentence-level amr
        :param shift_value: the value for shifting the sentence-level token indices to document-level token indices
        :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
                    1: do not add any unaligned tokens
                    2: add unaligned token if directly adjacent to an aligned token
                    3: add contiguous spans of unaligned tokens if span boundary adjacent to an aligned token
        :param nlp_model: already loaded spacy model for English to use for POS tagging
        """
        self.split_amr = split_amr                  # the action-level AMR
        self.sentence_amr = sentence_amr            # corresponding sentence-level AMR
        self.other_split_amrs = other_split_amrs    # the other action-level AMRs from the same sentence-level AMR
        self.shift_value = shift_value              # value to shift sentence-level to document-level token indices
        self.version = version                      # version for adding back unaligned tokens

        self.pos_tagger = nlp_model
        self.stemmer = PorterStemmer()              # stemmer to use for converting participles to imperative verbs

        self.original_sentence: str = split_amr.graph['snt']                        # original sentence instruction
        self.orig_snt_tokenized: List[str] = self.original_sentence.split(' ')      # tokenized original sentence
        self.orig_snt_tagged = []
        self.orig_snt_lemmas = []
        self.process_sentence(self.original_sentence)
        assert len(self.orig_snt_tokenized) == len(self.orig_snt_tagged)

        self.modified_snt_tokenized: List[str] = []     # tokens of original sentence with modified actions replaced with stemmed version
        self.modified_inds: List[int] = []              # sentence-level indices of the tokens that are modified
        self.modified_inds_others: List[int] = []       # sentence-level indices of tokens modified for the other_split_amrs
        self.action_root_inds: List[int] = []           # sentence-level indices of tokens that are actions and aligned to root in current split amr
        self.potential_tokens: List[int] = []           # sentence-level indices of tokens that might be used for the new instructions

        self.inds_to_add: Set[int] = set()              # sentence-level indices of tokens that will make up the extracted instruction
        self.final_tokens: List[str] = []               # tokenized extracted instruction
        self.final_tokens_tags = []
        self.final_tokens_orig_inds = []


    def process_sentence(self, sentence:str):
        """

        :param sentence:
        :return:
        """
        processed_sentence = self.pos_tagger(sentence)
        token_tag_pairs = []
        lemmas = []
        for sent in processed_sentence.sentences:
            for token in sent.words:
                token_tag_pairs.append((token.text, token.xpos))
                lemmas.append(token.lemma)
        self.orig_snt_tagged = token_tag_pairs
        self.orig_snt_lemmas = lemmas


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

        self.extract_final_tags_and_inds()

        # Switch Determiner and modified verb (extra function for easier readability)
        self.switch_modified_verb()

        # Fix the sentence ordering for cases such as "a toothpick insert into the centre"
        self.fix_sentence_order()

        # Modify sentence to start with upper-case character and end with sentence-final punctuation
        self.fix_sentence_start_and_end()

        with open('./all_new_sentences_tags.txt', 'a', encoding='utf-8') as f:
            f.write(f'{self.split_amr.graph["id"]}\t{" ".join(self.final_tokens)}\t{self.orig_snt_tagged}\n')

        return ' '.join(self.final_tokens)


    def extract_final_tags_and_inds(self):

        orig_token_ids = list(self.inds_to_add)         # the original indices of the tokens in self.final_tokens
        orig_token_ids.sort()                           # sort to get original token order

        extracted_pos_tags = []                         # the POS tags corresponding to the tokens in self.final_tokens
        for orig_id in orig_token_ids:
            extracted_pos_tags.append(self.orig_snt_tagged[orig_id][1])

        self.final_tokens_tags = extracted_pos_tags
        self.final_tokens_orig_inds = orig_token_ids


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
        if pos_tag in ['JJ', 'VBN', 'VBG', 'VBD', 'NN']:
            orig_token = token
            token = self.orig_snt_lemmas[token_ind]
            if orig_token == token and token.endswith('ing') or token.endswith('ed'):
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


    def switch_modified_verb(self):
        """

        :return:
        """
        if self.modified_inds:
            modified_ind = self.modified_inds[0]
            position_in_final_sentence = self.final_tokens_orig_inds.index(modified_ind)
            if self.final_tokens_tags[position_in_final_sentence - 1] == 'DT':# and position_in_final_sentence == 1:
                mod_token = self.final_tokens.pop(position_in_final_sentence)
                self.final_tokens.insert(0, mod_token)
                # need to do the same ordering changes to self.final_tokens_tags and self.final_tokens_orig_inds
                # because otherwise they will not match anymore!
                tmp_tag = self.final_tokens_tags.pop(position_in_final_sentence)
                self.final_tokens_tags.insert(0, tmp_tag)
                tmp_orig_ind = self.final_tokens_orig_inds.pop(position_in_final_sentence)
                self.final_tokens_orig_inds.insert(0, tmp_orig_ind)


    def fix_sentence_order(self):
        """
        Reorders self.final_tokens if an NP like phrase is at the beginning of the sentence, followed by a verb that
        is an action and original root
        Then the verb is moved to the sentence beginning because in imperative sentences the verb should
        be before its direct object
        :return:
        """

        extracted_pos_seq = ' '.join(self.final_tokens_tags)    # convert to string for re matching

        # POS pattern for NP-like phrases at the beginning the POS sequence followed potentially by an adverb
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
            matching_pos_list = matching_pos_list.strip()       # therefore strip
            matching_pos_list = matching_pos_list.split(' ')

            potential_verb_index = len(matching_pos_list) - 1  # is relative to extracted sentence
            potential_verb_index_original = self.final_tokens_orig_inds[potential_verb_index]  # relative to orig sentence
            if potential_verb_index_original in self.action_root_inds:
                verb_index = potential_verb_index

                if 'RB' in matching_pos_list:
                    potential_rb_index = len(matching_pos_list) - 2    # is relative to extracted sentence
                    rb_token = self.final_tokens[potential_rb_index]
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
                if len(matching_pos_list) != len(self.final_tokens_tags):  # otherwise potential_verb_index is out of index
                    # get corresponding index in the original sentence because modified_action_inds and
                    # action_root_inds are relative to orig sent
                    potential_verb_index_original = self.final_tokens_orig_inds[potential_verb_index]
                    # If it is an action verb but originally had different POS
                    if potential_verb_index_original in self.modified_inds:
                        verb_index = potential_verb_index
                    # If the verb got not modified (e.g. not the ending checked for) but is an action verb with POS tag NN or NNS
                    elif potential_verb_index_original in self.action_root_inds and \
                         (self.final_tokens_tags[potential_verb_index] == "NN" or self.final_tokens_tags[
                             potential_verb_index] == "NNS"):
                        verb_index = potential_verb_index

                    if verb_index and 'RB' in matching_pos_list:
                        potential_rb_index = len(matching_pos_list) - 1
                        rb_token = self.final_tokens[potential_rb_index]
                        # only re-order if the 'RB' token is a time attribute such as then, now, immediately

                        if rb_token == 'then' or rb_token == 'immediately' or rb_token == 'now':
                                rb_index = potential_rb_index

                # only if non-greedy does not find an appropriate pattern, check with greedy pattern next
                if not verb_index and current_case == "non-greedy":
                    matching_pos = search_matching_pos_mod_greedy_rb.group()
                    current_case = "greedy"
                else:
                    break

        if rb_index and verb_index:
            # verb needs to be removed first because it follows rb token and otherwise indices change
            verb_token = self.final_tokens.pop(verb_index)
            rb_token = self.final_tokens.pop(rb_index)
            self.final_tokens.insert(0, verb_token)
            self.final_tokens.insert(0, rb_token)
            tmp_tag_verb = self.final_tokens_tags.pop(verb_index)
            tmp_tag_rb = self.final_tokens_tags.pop(rb_index)
            self.final_tokens_tags.insert(0, tmp_tag_verb)
            self.final_tokens_tags.insert(0, tmp_tag_rb)
            tmp_orig_ind_verb = self.final_tokens_orig_inds.pop(verb_index)
            tmp_orig_ind_rb = self.final_tokens_orig_inds.pop(rb_index)
            self.final_tokens_orig_inds.insert(0, tmp_orig_ind_verb)
            self.final_tokens_orig_inds.insert(0, tmp_orig_ind_rb)

        elif verb_index:
            verb_token = self.final_tokens.pop(verb_index)
            self.final_tokens.insert(0, verb_token)
            tmp_tag_verb = self.final_tokens_tags.pop(verb_index)
            self.final_tokens_tags.insert(0, tmp_tag_verb)
            tmp_orig_ind_verb = self.final_tokens_orig_inds.pop(verb_index)
            self.final_tokens_orig_inds.insert(0, tmp_orig_ind_verb)


    def fix_sentence_start_and_end(self):
        """
        Remove punctuation at the sentence beginning, adds / replaces end of sentence to get sentence final punctuation and
        removes lonely 'and' at the end of the sentence
        Additionally, fix other punctuation issues
        - ( sentence ) -> remove brackets
        - tokens () tokens -> remove brackets
        - remove brackets if only opening or only closing brackets in string
        - "," "," -> remove one comma
        :return: returns nothing but modifies self.final_tokens directly
        """
        punctuation_to_remove = []
        brackets_open = []
        brackets_closing = []
        for t_ind, t in enumerate(self.final_tokens):
            if t == ',':
                try:
                    next_token = self.final_tokens[t_ind + 1]
                    if next_token == ',':
                        punctuation_to_remove.append(t_ind)
                except IndexError:
                    break
            elif t == '(':
                brackets_open.append(t_ind)
            elif t == ')':
                brackets_closing.append(t_ind)
        if brackets_open and not brackets_closing:
            final_str = ' '.join(self.final_tokens)
            if not ')' in final_str:    # some brackets do not get tokenized correctly
                punctuation_to_remove.extend(brackets_open)
        elif not brackets_open and brackets_closing:
            final_str = ' '.join(self.final_tokens)
            if not '(' in final_str:
                punctuation_to_remove.extend(brackets_closing)
        elif brackets_open and brackets_closing:
            for bo in brackets_open:
                if bo + 1 in brackets_closing:
                    punctuation_to_remove.append(bo)
                    punctuation_to_remove.append(bo + 1)

        self.final_tokens = [ft for ind_ft, ft in enumerate(self.final_tokens) if ind_ft not in punctuation_to_remove]
        self.final_tokens_tags = [ft for ind_ft, ft in enumerate(self.final_tokens_tags) if ind_ft not in punctuation_to_remove]
        self.final_tokens_orig_inds = [ft for ind_ft, ft in enumerate(self.final_tokens_orig_inds) if ind_ft not in punctuation_to_remove]

        while True:
            if self.final_tokens[0] == '(' and self.final_tokens[-1] == ')':
                self.final_tokens = self.final_tokens[1:-1]
                self.final_tokens_tags = self.final_tokens_tags[1:-1]
                self.final_tokens_orig_inds = self.final_tokens_orig_inds[1:-1]
                continue
            if self.final_tokens[-1] == 'and':
                self.final_tokens = self.final_tokens[:-1]
                self.final_tokens_tags = self.final_tokens_tags[:-1]
                self.final_tokens_orig_inds = self.final_tokens_orig_inds[:-1]
                continue
            if self.final_tokens[0] in [',', ';', '-', ')', '.']:
                self.final_tokens = self.final_tokens[1:]
                self.final_tokens_tags = self.final_tokens_tags[1:]
                self.final_tokens_orig_inds = self.final_tokens_orig_inds[1:]
                continue
            if self.final_tokens[-1] in [',', '-', '(', ';', ':']:
                self.final_tokens = self.final_tokens[:-1]
                self.final_tokens_tags = self.final_tokens_tags[:-1]
                self.final_tokens_orig_inds = self.final_tokens_orig_inds[:-1]
                continue
            if self.final_tokens[-1] not in ['.', '!', '?']:
                self.final_tokens.append('.')
                self.final_tokens_tags.append('.')
                self.final_tokens_orig_inds.append(-1)
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
                add_pt = self.decide_about_unaligned_token(pt_ind, pt_pos_tag, inds_to_add_unchanged, True)
                if add_pt:
                    self.inds_to_add.add(pt_ind)

        # Add the unaligned tokens if they are adjacent to tokens that get added
        # do this in both directions to be able to add e.g. 'In a' at the beginning of an instruction
        elif self.version == 3:
            # start with backward direction
            potential_tokens_rev = self.potential_tokens.copy()
            potential_tokens_rev.reverse()
            for pt_ind in potential_tokens_rev:
                _, pt_pos_tag = self.orig_snt_tagged[pt_ind]
                add_pt = self.decide_about_unaligned_token(pt_ind, pt_pos_tag, self.inds_to_add, False)
                if add_pt:
                    self.inds_to_add.add(pt_ind)

            # then forward direction
            potential_tokens_rev.reverse()
            for pt_ind in potential_tokens_rev:
                _, pt_pos_tag = self.orig_snt_tagged[pt_ind]
                add_pt = self.decide_about_unaligned_token(pt_ind, pt_pos_tag, self.inds_to_add, True)
                if add_pt:
                    self.inds_to_add.add(pt_ind)


    def decide_about_unaligned_token(self, pt_ind, pt_pos_tag, inds_to_add, forward) -> bool:
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
            elif forward and pt_pos_tag == 'DT' and pt_ind + 1 in self.modified_inds and pt_ind + 1 in inds_to_add:
                to_add = True
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
        # for cases such as [Determiner Adverb ParticipleAction Noun], e.g. "the slowly cooked mixture"
        elif pt_ind + 3 in inds_to_add and pt_ind + 2 in self.modified_inds_others and pt_ind + 2 not in inds_to_add:
            if pt_pos_tag == 'DT' or pt_pos_tag == ',':
                to_add = True
                #print(self.split_amr.graph['id'])
                #print(self.orig_snt_tagged[pt_ind:pt_ind+4])

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



