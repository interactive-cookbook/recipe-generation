import os

import torch
from torch.utils.data import Dataset


# Taken from amrlib code
class AMRDataset(Dataset):
    def __init__(self, encodings, sents):
        self.encodings = encodings
        self.sents = sents

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


# Taken from amrlib code
class T2TDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['tagret_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        collated_data = {'input_ids': input_ids, 'attention_mask': attention_mask,
                         'labels': lm_labels, 'decoder_attention_mask': decoder_attention_mask}

        return collated_data


def build_dataset(tokenizer, data_path, context_len: int, linearization: str = 'penman'):

    data_set_entries = read_data_set(data_path, context_len, linearization)
    # TODO concatenate 'graph' and 'context' elements and add special token between them


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




