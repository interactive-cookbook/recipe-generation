import os
from bs4 import BeautifulSoup
"""
Code to 
* extract all amr graphs from the AMR 3.0 corpus for which there exist multisentence-amr annotations
* create a corpus with one file per document, each containing the AMRs making up this document as specified in 
the multisentence-amr xml files 
"""


def get_doc_amrs(doc_amr_path) -> dict:
    """
    Get the AMR ids and their order for each document
    :param doc_amr_path: path to the folder with the ms-amr xml files
                        i.e. should be [some_path]/amr_annotation_3.0/data/multisentence/ms-amr-unsplit
    :return: a dictionary with a sub dictionary for each file in the doc_amr_path folder
             key: name of the file
             value: dictionary with
                     'ids': list of the amr/sentence ids that are part of the document
                     'position': ordering information, such that int value x at position y means that the
                                 sentence with the id at position y in 'ids' is the xth sentence in the document
    """
    documents = dict()

    for file in os.listdir(doc_amr_path):
        with open('/'.join([doc_amr_path, file]), 'r', encoding='utf-8') as f:
            data = f.read()

        documents[file] = dict()
        documents[file]['ids'] = []
        documents[file]['position'] = []

        Bs_data = BeautifulSoup(data, "xml")

        amr_tag = Bs_data.find_all('amr')
        for sent_amr in amr_tag:
            sent_id = sent_amr.get('id')
            sent_position = sent_amr.get('order')
            sent_position = int(sent_position)
            documents[file]['ids'].append(sent_id)
            documents[file]['position'].append(sent_position)

    return documents


def write_corpus_files(ms_amr_mappings: dict, id_amr_mappings: dict):
    """
    Creates a file for each document for which the multi-sentence amr annotations exist and
    writes the amrs in the correct order to it
    :param ms_amr_mappings: a dictionary with a sub dictionary for each document of the multi-sentence amr corpus
                            key: name of the file
                            value: dictionary with
                                    'ids': list of the amr/sentence ids that are part of the document
                                    'position': ordering information, such that int value x at position y means that the
                                            sentence with the id at position y in 'ids' is the xth sentence in the document
    :param id_amr_mappings: a dictionary with all amr ids as keys and the complete string for the corresponding amr
                            (including metadata) as value
    :return:
    """
    for document, doc_amrs in ms_amr_mappings.items():
        document_name = '.'.join(document.split('.')[:-1])
        document_name += '.txt'
        id_pos_pairs = list(zip(doc_amrs['ids'], doc_amrs['position']))
        id_pos_pairs.sort(key = lambda x: x[1])
        # NOTE: some xml ms-amr files start counting at 1 others start counting at 0
        prev_position = -1
        with open('/'.join(["./ms_amr_graphs", document_name]), 'w', encoding='utf-8') as file:
            for amr_id, snt_pos in id_pos_pairs:
                if prev_position != -1:
                    assert snt_pos == prev_position + 1
                prev_position = snt_pos
                amr_str = id_amr_mappings[amr_id]
                file.write(f'{amr_str}\n')


def create_doc_amr_corpus(ms_amr_dir, all_amr_dir):
    """
    Creates a corpus of all amrs of the AMR 3.0 corpus for which multisentence annotations exist

    :param ms_amr_dir: path to the folder with the ms-amr xml files
                        i.e. should be [some_path]/amr_annotation_3.0/data/multisentence/ms-amr-unsplit
    :param all_amr_dir: path to the folder with the complete AMR 3.0 graphs
                        i.e. should be [some_path]/amr_annotation_3.0/data/amrs/unsplit
    :return:
    """
    ms_amr_files = get_doc_amrs(ms_amr_dir)
    id2amrs = dict()

    for file in os.listdir(all_amr_dir):
        amr_file_path = '/'.join([all_amr_dir, file])

        with open(amr_file_path, 'r', encoding='utf-8') as amrs:
            current_amr_str = ""
            current_amr_id = ""

            for line in amrs:

                if line.startswith("# AMR release"):
                    continue

                elif line == '\n':
                    if current_amr_str:
                        id2amrs[current_amr_id] = current_amr_str
                        current_amr_str = ""
                        current_amr_id = ""

                elif line[0] == '#':
                    meta_type = line.strip().split(' ')[1]
                    meta_values = line.strip().split()[2:]
                    meta_key = meta_type[2:]
                    if meta_key == 'id':
                        assert current_amr_str == ""
                        sentence_id = meta_values[0]
                        current_amr_id = sentence_id
                    current_amr_str += line

                else:
                    current_amr_str += line

            if current_amr_str:
                id2amrs[current_amr_id] = current_amr_str

    write_corpus_files(ms_amr_files, id2amrs)


if __name__=='__main__':

    create_doc_amr_corpus("./amr_annotation_3.0/data/multisentence/ms-amr-unsplit",
                          "./amr_annotation_3.0/data/amrs/unsplit")