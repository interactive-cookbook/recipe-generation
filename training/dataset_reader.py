import os

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Taken from amrlib code
class AMRDataset(Dataset):
    def __init__(self, encodings, sents):
        self.encodings = encodings
        self.sents = sents

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def build_dataset(tokenizer: PreTrainedTokenizer, data_path, context_len: int, linearization: str = 'penman') -> Dataset:

    data_set_entries = read_data_set(data_path, context_len, linearization)
    # TODO maybe change special token
    contextualized_input = [f'{c} <GRAPH> {g}' for (c, g) in zip(data_set_entries['context'], data_set_entries['graph'])]
    input_seqs = ['%s' % cont_gr for cont_gr in contextualized_input]
    target_seqs = ['%s' % sent for sent in data_set_entries['sent']]

    # Encode input and target using tokenizer
    # original code had also truncation = True and max_length = X TODO: decide what to do
    input_encodings = tokenizer.batch_encode_plus(input_seqs,
                                                  padding=True,
                                                  return_overflowing_tokens=True)
    target_encodings = tokenizer.batch_encode_plus(target_seqs,
                                                   padding=True,
                                                   return_overflowing_tokens=True)
    encodings = {'input_ids': torch.LongTensor(input_encodings['input_ids']),
                 'attention_mask': torch.LongTensor(input_encodings['attention_mask']),
                 'target_ids': torch.LongTensor(target_encodings['input_ids']),
                 'target_attention_mask': torch.LongTensor(target_encodings['attention_mask'])}

    data_set = AMRDataset(encodings, data_set_entries['sent'])
    return data_set


def read_data_set(data_path, context_len: int, linearization:str = 'penman'):

    if os.path.isdir(data_path):
        data_set_entries = {'sent': [], 'graph': [], 'context': []}
        for document in os.listdir(data_path):
            document_entries = read_document(os.path.join(data_path, document), context_len, linearization)
            data_set_entries['sent'].extend(document_entries['sent'])
            data_set_entries['graph'].extend(document_entries['graph'])
            data_set_entries['context'].extend(document_entries['context'])
    # enable also reading in only one train / val / test file, e.g. with context_len = 0 to get same behavior as
    # for original model
    else:
        data_set_entries = read_document(data_path, context_len, linearization)

    return data_set_entries


# based on amr_loading.py from amrlib but modified
def read_document(document_file, context_len, linearization:str = 'penman'):

    document_dict = {'sent': [], 'graph': [], 'context': []}
    ordered_sentences = []

    with open(document_file, 'r', encoding='utf-8') as doc_f:
        document = doc_f.read()

    for amr_data in document.split('\n\n'):
        graph_str = []
        sentence = None
        if not amr_data:
            continue
        elif amr_data.startswith('# AMR release'):
            continue
        for line in amr_data.split('\n'):
            if line.startswith('# ::snt'):
                sentence = line[len('# ::snt'):].strip()
            elif line.startswith('# '):
                continue
            else:
                graph_str.append(line.strip())

        if sentence and graph_str:
            if linearization != 'penman':
                raise NotImplemented
            ordered_sentences.append(sentence)
            document_dict['sent'].append(sentence)
            document_dict['graph'].append(' '.join(graph_str))

            # l[:n] returns first n elements if len(l) >= n, else returns l
            context = ordered_sentences[:context_len]
            # concatenate the context sentences; if first sentence then context will be empty string
            document_dict['context'].append(' '.join(context))

    return document_dict




