import os
import json
import torch
from argparse import ArgumentParser
from typing import List, Tuple
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from .dataset_reader import read_data_set, remove_token_alignments
from torch.utils.data import DataLoader
from tqdm import tqdm


# Based on amrlib code: https://github.com/bjascob/amrlib/blob/master/amrlib/models/generate_t5/inference.py
class RecipeGenerator:
    """
    A class to hold the fine-tuned generation model and all parameters relevant for running the
    generation as well as the method to run the generation for contextualized AMR graphs
    """
    def __init__(self, configuration: dict):
        self.task = 'translation_cond_amr_to_text'
        self.model = T5ForConditionalGeneration.from_pretrained(configuration['model_name_or_path'])
        self.tokenizer = T5Tokenizer.from_pretrained(configuration['tokenizer_name_or_path'])
        self.max_in_len = configuration.get('max_in_len', 1024)
        self.max_out_len = configuration.get('max_out_len', 1024)

        self.device = configuration['device'] if torch.cuda.is_available() else 'cpu'
        self.batch_size = configuration.get('batch_size', 1)
        self.num_beams = configuration.get('num_beams', 1)
        self.num_ret_seq = configuration.get('num_ret_seq', 1)
        self.linearization = configuration.get('linearization',
                                               self.model.config.task_specific_params[self.task].get('linearization', 'penman'))
        self.sep_token = self.model.config.task_specific_params[self.task].get('sep_token', '')

    def generate(self, contexts: List[str], graphs: List[str]) -> Tuple[List[str], List[bool]]:
        """
        Generates sentences for context-graph input sequences using self.model and self.tokenizer
        graphs should be a list of penman graph strings without metadata, line breaks or indentation
        contexts should contain the corresponding contexts for the graphs in the matching order
        each context at position i in contexts will be prepended to the graph at position i in graphs
        separated by the self.sep_token
        If no context should be preprended, the corresponding context in contexts should be the empty string
        :param contexts: a list of contexts
        :param graphs: a list of graphs
        :return: the list of sentences generated from the graphs and previous context
                 list of bools, indicating for each sentence at same index in the list of generated sentences,
                 whether the input got truncated
        """
        # some assertions to identify problem with the input beforehand
        assert isinstance(graphs, list)
        assert isinstance(contexts, list)
        assert len(graphs) == len(contexts)
        for g in graphs:
            assert not g.startswith('# ::'), "AMR graphs include metadata. Please remove the metadata to get the correct" \
                                             "behavior of the model. "

        if self.linearization == 'penman_wo_alignments':
            converted_graphs = [remove_token_alignments(gr_str) for gr_str in graphs]
            graphs = converted_graphs
        elif self.linearization != 'penman':
            raise NotImplemented

        contextualized_graphs = [f'{c} {self.sep_token} {g}' for (c, g) in zip(contexts, graphs)]
        generated_sentences = []

        dataloader = DataLoader(contextualized_graphs, batch_size=self.batch_size)
        clipped = []
        for batch in tqdm(dataloader):
            input_str = ['%s' % c_gr for c_gr in batch]
            # encode input
            if self.max_in_len:
                input_encodings = self.tokenizer.batch_encode_plus(input_str,
                                                                   padding=True,
                                                                   truncation=True,
                                                                   max_length=self.max_in_len,
                                                                   return_overflowing_tokens=True)

                for ind, trunc in enumerate(input_encodings['num_truncated_tokens']):
                    if trunc > 0:
                        clipped.append(True)
                    else:
                        clipped.append(False)

            else:
                input_encodings = self.tokenizer.batch_encode_plus(input_str, padding=True)

            input_ids = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)

            # run inference
            if self.max_out_len:
                output = self.model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             max_length=self.max_out_len,
                                             early_stopping=True,
                                             num_beams=self.num_beams,
                                             num_return_sequences=self.num_ret_seq)
            else:
                output = self.model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             early_stopping=True,
                                             num_beams=self.num_beams,
                                             num_return_sequences=self.num_ret_seq)

            # decode the output
            decoded_output = [self.tokenizer.decode(out_ids, skip_special_tokens=True) for out_ids in output]
            generated_sentences.extend(decoded_output)

        return generated_sentences, clipped


if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the configuration file for generation")
    args = parser.parse_args()
    config_file = args.config
    generate_data_set(config_file)



