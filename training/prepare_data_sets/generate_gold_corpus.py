import os
from typing import List, Dict, Union
import networkx as nx
import penman
from collections import defaultdict
from pathlib import Path
import stanza
import json
import argparse

from graph_processing.recipe_graph import read_graph_from_conllu
from graph_processing.read_graphs import read_aligned_amr_file
from graph_processing.graph_traversal import order_actions_pf_lf_id
from amr_processing.penman_networkx_conversions import penman2networkx, networkx2penman
from coref_processing.coref_utils import read_joined_coref
from coref_processing.create_ident_clusters import create_ident_clusters
from utils.paths import ACTION_AMR_DIR, ARA_DIR, SENT_AMR_DIR, JOINED_COREF_DIR
from generate_gold_action_instruction import InstructionExtractor

"""
Functions to create a corpus of gold instructions for the split AMRs using a simple string-match and rule-based approach
"""


class RecipeExtractor:

    def __init__(self,
                 recipe_amrs: List[nx.Graph],
                 orig_amrs: Dict,
                 action_graph: nx.DiGraph,
                 nlp_model,
                 ident_clusters: Union[None, List],
                 coref_clusters: Union[None, List],
                 version: int = 3,
                 text_only: bool = False,
                 order_ac_graph: bool = False):
        """

        :param recipe_amrs: list of the action-level amrs
        :param orig_amrs: dict with the corresponding sentence-level AMRs
                          keys: the original amr/instruction ID; values: the corresponding graph (networkX Graph)
        :param action_graph: the action graph for the recipe
        :param nlp_model: already loaded stanza model for English to use for POS tagging
        :param ident_clusters: None or the information about which nodes in the different A-AMRs were a single node in the
                                original S-AMR
        :param coref_clusters: None or the coreference information
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
        self.nlp_model = nlp_model
        self.ident_clusters = ident_clusters
        self.coref_clusters = coref_clusters
        self.instr_extractors: Dict[str, InstructionExtractor] = dict()

        self.all_new_sentences = dict()

        self.orig_pos2amr = defaultdict(list)       # key: n, value: [gr1, gr2] means that gr1 and gr2 are from the
                                                    # original instruction at the nth position in the original recipe
        self.group_amrs_by_orig_position()

    def create_gold_recipe(self):

        self.create_gold_instructions()     # extracts the gold instructions for each amr and adds as new 'snt' metadata
        self.create_instruction_order()     # brings the amrs into an appropriate order
        if self.ident_clusters:
            self.create_coherent_dets()
        if self.coref_clusters:
            self.remove_redundant_mentions()

        self.log_new_sentences()

        return self.recipe_amrs

    def log_new_sentences(self):
        with open('./all_new_sentences_final.txt', 'a', encoding='utf-8') as f:
            for sent_id, sentence in self.all_new_sentences.items():
                f.write(f'{sent_id}\t{sentence}\n')

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

                instruction_creator = InstructionExtractor(split_amr=amr, sentence_amr=orig_snt_amr,
                                                           other_split_amrs=others, shift_value=shift_value,
                                                           version=self.version, nlp_model=self.nlp_model)
                gold_sentence = instruction_creator.create_gold_sentence()
                self.instr_extractors[post_split_id] = instruction_creator

                self.all_new_sentences[post_split_id] = gold_sentence
            # change sentence metadata
            amr.graph['snt'] = gold_sentence
            prev_sent_len = len(original_sentence.split(' '))

    def create_instruction_order(self):
        """
        Re-orders the self.recipe_amrs
        either according to the action graph or re-orders only the instructions that were split
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
                    try:
                        ordered_sep_amrs = self.order_amrs(amrs)
                    except:
                        ordered_sep_amrs = amrs
                        print(f'Warning: found action graph with disconnected components or without end node. '
                              f'Correct ordering cannot be ensured! {self.recipe_amrs[0].graph["id"]}')
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

        partially_sorted_action_nodes = order_actions_pf_lf_id(self.action_graph)

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

    def create_coherent_dets(self):
        """
        If self.ident_clusters is None, this functions has no effect
        Otherwise, it checks whether in two action-level sentences that stem from the same original sentence there are tokens
        which are aligned to AMR nodes that were the same AMR node in the original S-AMR and that are preceded by an indefinite determiner
        If this is the case, the indefinite determiner in the second sentence (according to the determined order of A-AMRs
        is replaced with a definite determiner)
        :return:
        """
        ordered_amr_ids = [gr.graph['id'] for gr in self.recipe_amrs]
        for cluster in self.ident_clusters:
            involved_amr_ids = cluster['amr_ids']
            relevant_ids_ordered = [amr_id for amr_id in ordered_amr_ids if amr_id in involved_amr_ids]
            relevant_extractors = []
            for amr_id in relevant_ids_ordered:
                relevant_extractors.append(self.instr_extractors[amr_id])
            orig_shared_token_id = cluster['token_id']       # is relative to document

            found_indef = False

            for extractor in relevant_extractors:
                shared_token_id = orig_shared_token_id - extractor.shift_value   # needs to be shifted to match extractor ids
                if shared_token_id in extractor.final_tokens_orig_inds:
                    new_index = extractor.final_tokens_orig_inds.index(shared_token_id)
                    potential_determiner_index = new_index - 1
                    pos_tag = extractor.final_tokens_tags[potential_determiner_index]

                    if pos_tag == 'DT':
                        potential_determiner_token = extractor.final_tokens[potential_determiner_index]
                        if potential_determiner_token.lower() in ['a', 'an'] and not found_indef:
                            found_indef = True
                        elif potential_determiner_token.lower() in ['a', 'an'] and found_indef:
                            # in this case there was already a mention of the same entity in a previous sentence
                            # with an indefinite determiner and now the definite one needs to be used
                            if potential_determiner_index == 0: # should probably never happen, but still check ...
                                def_det = 'The'
                            else:
                                def_det = 'the'

                            new_sentence_tokens = extractor.final_tokens
                            new_sentence_tokens[potential_determiner_index] = def_det   # changes also the extractor.final_tokens list!

                            new_instruction = ' '.join(new_sentence_tokens)
                            current_amr_id = extractor.split_amr.graph['id']
                            current_amr_position = ordered_amr_ids.index(current_amr_id)
                            current_graph = self.recipe_amrs[current_amr_position]
                            current_graph.graph['snt'] = new_instruction
                            self.all_new_sentences[current_amr_id] = new_instruction

    def remove_redundant_mentions(self):
        """
        Makes use of the coreference clusters in order to remove duplicated entity mentions such as in
        "Knead the dough it until smooth"
        If two token spans of the same coreference cluster occur directly after each other, the second span gets removed
        Directly changes the snt attribute of the graphs for which this is the case
        :return:
        """
        ordered_amr_ids = [gr.graph['id'] for gr in self.recipe_amrs]
        tokens_to_remove = defaultdict(set)
        for cluster in self.coref_clusters:
            token_spans = cluster['token_ids']

            involved_amr_ids = set()
            for node_span in cluster['amr_nodes']:
                for node_occurrence in node_span:
                    for node in node_occurrence:
                        amr_id = node[2]
                        involved_amr_ids.add(amr_id)

            relevant_ids_ordered = [amr_id for amr_id in ordered_amr_ids if amr_id in involved_amr_ids]
            # only change sentences that were split, otherwise there is also no InstructionExtractor object
            relevant_ids_ordered = [amr_id for amr_id in relevant_ids_ordered if amr_id in self.all_new_sentences.keys()]
            relevant_extractors = []
            for amr_id in relevant_ids_ordered:
                relevant_extractors.append(self.instr_extractors[amr_id])

            # for each sentence which potentially includes redundant coreferences
            for extractor in relevant_extractors:
                extracted_sent_ids = [sent_id + extractor.shift_value for sent_id in extractor.final_tokens_orig_inds]

                successive_coref_token_spans = []
                prev_span = []
                for extr_token_id in extracted_sent_ids:
                    flag = False
                    for span in token_spans:
                        if extr_token_id in span:
                            if prev_span and not prev_span == span:
                                # then the current token and the previous one are in the same coref cluster but belong
                                # to different spans -> are two redundant mentions of acutally the same entitiy
                                successive_coref_token_spans.append((prev_span, span))
                                prev_span = span
                            else:
                                prev_span = span
                            flag = True
                            break
                    if not flag:
                        prev_span = []

                tokens_to_remove_current = set()
                for successive_pair in successive_coref_token_spans:
                    span2 = [t_id for t_id in successive_pair[1] if t_id in extracted_sent_ids]
                    # need to know at which index these token_ids are so the tokens can get removed
                    span2_token_ids = [extracted_sent_ids.index(t_id) for t_id in span2]
                    tokens_to_remove_current = tokens_to_remove_current.union(set(span2_token_ids))

                tokens_to_remove[extractor.split_amr.graph['id']] = tokens_to_remove[extractor.split_amr.graph['id']].union(tokens_to_remove_current)


        for instr_id, remove_tokens in tokens_to_remove.items():

            if not remove_tokens:
                continue

            current_amr_position = ordered_amr_ids.index(instr_id)
            current_graph = self.recipe_amrs[current_amr_position]
            current_extractor = self.instr_extractors[instr_id]

            old_instruction = current_graph.graph['snt'].split(' ')

            new_instruction = old_instruction.copy()
            remove_tokens = list(remove_tokens)
            remove_tokens.sort()
            counter = 0
            # remove tokens and update corresponding recipe extractor
            for rt in remove_tokens:
                new_instruction.pop(rt - counter)
                current_extractor.final_tokens.pop(rt - counter)
                current_extractor.final_tokens_orig_inds.pop(rt - counter)
                current_extractor.final_tokens_tags.pop(rt - counter)
                counter += 1

            new_instruction = ' '.join(new_instruction)
            current_graph.graph['snt'] = new_instruction
            self.all_new_sentences[instr_id] = new_instruction


def create_gold_corpus(action_amr_corpus: str,
                       sentence_amr_corpus: str,
                       action_graph_corpus: str,
                       gold_corpus_dir: str,
                       use_coref: bool,
                       version: int = 3,
                       text_only: bool = False):
    """

    :param action_amr_corpus: path to parent folder of the action-level AMR files
    :param sentence_amr_corpus: path to parent folder of the sentence-level AMR files
    :param action_graph_corpus: path to parent folder of the action-graph conllu files
    :param gold_corpus_dir: path to directory for the generated gold data set, gets created if not exists yet
    :param use_coref: whether to use the coreference information to remove redundant mentions
    :param version: algorithm version to use for deciding about adding unaligned tokens; i.e. the "sliding window"
                    1: do not add any unaligned tokens
                    2: add unaligned token if directly adjacent to an aligned token
                    3: add contiguous spans of unaligned tokens if span boundary adjacent to an aligned token
                    Version 3 works best
    :param text_only: if set to True then files only containing the generated sentences (but not the AMRs) are created
    :return:
    """
    count = 0
    Path(gold_corpus_dir).mkdir(exist_ok=True, parents=True)
    nlp_model_config = {'processors': 'tokenize,mwt,pos,lemma',
                        'lang': 'en',
                        'tokenize_pretokenized': True}
    nlp_model = stanza.Pipeline(**nlp_model_config)

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

            # get the corresponding coreference file if specified
            if use_coref:
                coref_file = recipe_name + '_joined.jsonlines'
                coref_file_path = os.path.join(JOINED_COREF_DIR, dish, coref_file)
                identity_clusters, coref_clusters = read_joined_coref(coref_file_path)
            else:
                # extract the information about previously identical nodes
                identity_clusters = create_ident_clusters(recipe_path, action_graph_path)
                # set coref to None
                coref_clusters = None

            # create the gold instructions for the action-level AMRs of the current recipe
            gold_recipe_creator = RecipeExtractor(recipe_amrs=recipe_amrs, orig_amrs=orig_amrs,
                                                  action_graph=recipe_action_graph, nlp_model=nlp_model,
                                                  ident_clusters=identity_clusters, coref_clusters= coref_clusters,
                                                  version=version, text_only=text_only)
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

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--sep_dir', required=False)
    arg_parser.add_argument('--orig_dir', required=False)
    arg_parser.add_argument('--ara_dir', required=False)
    arg_parser.add_argument('--out_dir', required=True)
    arg_parser.add_argument('--coref', required=False)
    arg_parser.add_argument('--text', required=False, action='store_true')

    args = arg_parser.parse_args()

    action_amr_dir = args.sep_dir if args.sep_dir else ACTION_AMR_DIR
    sent_amr_dir = args.orig_dir if args.orig_dir else SENT_AMR_DIR
    ara_dir = args.ara_dir if args.ara_dir else ARA_DIR
    out_dir = args.out_dir
    coref = args.coref if args.coref else False
    only_text = args.text
    create_gold_corpus(action_amr_dir, sent_amr_dir, ara_dir, out_dir, coref, 3, only_text)

    #create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_amrs_ara1', True, 3, False)
    #create_gold_corpus(ACTION_AMR_DIR, SENT_AMR_DIR, ARA_DIR, '../tuning_data_sets/gold_sentences_test', True, 3, True)


