import os
import re

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Taken from amrlib code: https://github.com/bjascob/amrlib/blob/master/amrlib/models/generate_t5/trainer.py
class AMRDataset(Dataset):
    def __init__(self, encodings, sents):
        self.encodings = encodings
        self.sents = sents

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


# based on amrlib code: https://github.com/bjascob/amrlib/blob/master/amrlib/models/generate_t5/trainer.py
def build_dataset(tokenizer: PreTrainedTokenizer, data_path, context_len: int,
                  max_in_length: int, max_out_length,
                  sep_token: str, linearization: str = 'penman') -> Dataset:
    """
    Reads data from file(s) and creates a data set such that for the input sequences
    to each amr graph the context_len previous sentences from the same document are prepended,
    separated by the specified sep_token
    target sequence is the sentence corresponding to the amr graph
    each file is treated as one document if context_len > 0
    Returned data set contains the input and target sequences encoded by the tokenizer
    :param tokenizer: a pretrained T5 tokenizer
    :param data_path: path to the file with the data instances or to a folder containing the file(s) with the instances
    :param context_len: the number of previous sentences to add to the current amr
    :param max_in_length: max input length, longer ones will be truncated
    :param max_out_length: max target / output length, longer ones will be truncated
    :param sep_token: the separator token to add between context and amr graph
    :param linearization: the linearization type to use for amr graph
    :return: an AMRDataset object containing the encoded inputs and the sentences
    """

    data_set_entries = read_data_set(data_path, context_len)
    # Convert graph strings if linearization is different from
    if linearization == 'penman_wo_alignments':
        converted_graphs = [remove_token_alignments(gr_str) for gr_str in data_set_entries['graph']]
        data_set_entries['graph'] = converted_graphs
    elif linearization != 'penman':
        raise NotImplemented

    contextualized_input = [f'{c} {sep_token} {g}' for (c, g) in zip(data_set_entries['context'], data_set_entries['graph'])]
    input_seqs = ['%s' % cont_gr for cont_gr in contextualized_input]
    target_seqs = ['%s' % sent for sent in data_set_entries['sent']]

    # Encode input and target using tokenizer
    # original code had also truncation = True and max_length = X
    # TODO: decide what to do -> does not work to have padding and return_overflowing_tokens but not max_length and truncation

    if max_in_length:
        input_encodings = tokenizer.batch_encode_plus(input_seqs,
                                                      padding=True,
                                                      max_length=max_in_length,
                                                      truncation=True,
                                                      return_overflowing_tokens=True)
    else:
        input_encodings = tokenizer.batch_encode_plus(input_seqs, padding=True)

    if max_out_length:
        target_encodings = tokenizer.batch_encode_plus(target_seqs,
                                                       padding=True,
                                                       max_length=max_out_length,
                                                       truncation=True,
                                                       return_overflowing_tokens=True)
    else:
        target_encodings = tokenizer.batch_encode_plus(target_seqs, padding=True)

    indices_truncated = set()
    if max_in_length:
        for ind, trunc in enumerate(input_encodings['num_truncated_tokens']):
            if trunc > 0:
                indices_truncated.add(ind)
    if max_out_length:
        for ind, trunc in enumerate(target_encodings['num_truncated_tokens']):
            if trunc > 0:
                indices_truncated.add(ind)

    #input_encodings['input_ids'] = [ie for ind, ie in enumerate(input_encodings['input_ids']) if ind not in indices_truncated]
    #target_encodings['input_ids'] = [te for ind, te in enumerate(target_encodings['input_ids']) if ind not in indices_truncated]
    #input_encodings['attention_mask'] = [ie for ind, ie in enumerate(input_encodings['attention_mask']) if ind not in indices_truncated]
    #target_encodings['attention_mask'] = [te for ind, te in enumerate(target_encodings['attention_mask']) if ind not in indices_truncated]

    print(f'{len(indices_truncated)} truncated data instances.')

    encodings = {'input_ids': torch.LongTensor(input_encodings['input_ids']),
                 'attention_mask': torch.LongTensor(input_encodings['attention_mask']),
                 'target_ids': torch.LongTensor(target_encodings['input_ids']),
                 'target_attention_mask': torch.LongTensor(target_encodings['attention_mask'])}

    data_set = AMRDataset(encodings, data_set_entries['sent'])
    return data_set


def read_data_set(data_path, context_len: int):
    """
    Reads the graphs, sentences and previous context of length context_len from the file
    data_path if it is a file or from all files in data_path if data_path is a directory
    each file is treated as one document, i.e. first data instance in a file has no previous context
    :param data_path: path to the file with the data instances or to a folder containing the file(s) with the instances
    :param context_len: the number of previous sentences to add to the current amr
    :return:
    """
    if os.path.isdir(data_path):
        data_set_entries = {'sent': [], 'graph': [], 'context': []}
        for document in os.listdir(data_path):
            document_entries = read_document(os.path.join(data_path, document), context_len)
            data_set_entries['sent'].extend(document_entries['sent'])
            data_set_entries['graph'].extend(document_entries['graph'])
            data_set_entries['context'].extend(document_entries['context'])
    # enable also reading in only one train / val / test file, e.g. with context_len = 0 to get same behavior as
    # for original model
    else:
        data_set_entries = read_document(data_path, context_len)

    return data_set_entries


# based on amr lib code: https://github.com/bjascob/amrlib/blob/master/amrlib/graph_processing/amr_loading.py
def read_document(document_file, context_len):
    """
    Reads the graphs, sentences and previous context of length context_len from a file
    File needs to be in penman format, individual amrs separated by one empty line
    The sentence corresponding to an AMR graph needs to be available as metadata with the name "::snt"
    If a sentence/graph is at position i, with i >= context_len then the previous context will be i-1 long
    An empty context is the empty string ''
    :param document_file: path to one file with amr graphs and their metadata, i.e. the corresponding sentence
    :param context_len: the number of previous sentences to add to the current amr
    :return:
    """
    document_dict = {'sent': [], 'graph': [], 'context': []}
    ordered_sentences = []

    with open(document_file, 'r', encoding='utf-8') as doc_f:
        document = doc_f.read()

    for amr_data in document.split('\n\n'):
        graph_str_lines = []
        sentence = None
        if not amr_data:
            continue
        elif amr_data.startswith('# AMR release'):
            continue
        for line in amr_data.split('\n'):
            if line.startswith('# ::snt') and not line.startswith('# ::snt_'):
                sentence = line[len('# ::snt'):].strip()
            elif line.startswith('# '):
                continue
            else:
                graph_str_lines.append(line.strip())

        graph_str = ' '.join(graph_str_lines)
        if sentence and graph_str_lines:

            document_dict['sent'].append(sentence)
            document_dict['graph'].append(graph_str)

            if context_len > 0:
                context = ordered_sentences[-context_len:]
            else:
                context = []
            # concatenate the context sentences; if first sentence then context will be empty string
            document_dict['context'].append(' '.join(context))
            ordered_sentences.append(sentence)

    return document_dict


def remove_token_alignments(graph_str: str):
    """
    Remove alignment information from graph string
    :param graph_str:
    :return:
    """
    reg = r'~e.[1-9]+'
    stripped_graph_str = re.sub(reg, '', graph_str)

    return stripped_graph_str

