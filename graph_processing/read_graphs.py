import penman
from typing import List, Union
from pathlib import Path

"""
Function to read the AMRs produced by the Struct-BART parser, i.e. including node-token alignments
Creates a list of penman graph objects 
"""


def read_aligned_amr_file(recipe_amr_file: Union[str, Path]) -> List[penman.Graph]:
    """
    Reads a file with the sentence or action level AMRs for one recipe
    Node to token alignments included
    :param recipe_amr_file: .txt file with the instruction AMRs
    :return: a list of all AMRs as penman Graph objects with
             complete metadata
    """
    graphs = []

    with open(recipe_amr_file, "r", encoding="utf-8") as amrs:
        current_amr_str = ""
        current_meta_data = dict()

        for line in amrs:
            if line == '\n':
                current_amr = penman.decode(current_amr_str)
                for meta_k, meta_v in current_meta_data.items():
                    current_amr.metadata[meta_k] = meta_v
                graphs.append(current_amr)
                current_amr_str = ""
                current_meta_data = dict()

            # add the meta data
            elif line[0] == '#':
                meta_type = line.strip().split(' ')[1]
                meta_values = line.strip().split()[2:]
                meta_key = meta_type[2:]
                current_meta_data[meta_key] = ' '.join(meta_values)
            else:
                current_amr_str += line.strip() + ' '

        if current_amr_str != "":
            current_amr = penman.decode(current_amr_str)
            for meta_k, meta_v in current_meta_data.items():
                current_amr.metadata[meta_k] = meta_v
            graphs.append(current_amr)

    return graphs

