import penman
from penman import surface
import os
from pathlib import Path


"""
Script to create a new version of the parsed AMRs, namely one that matches the token ids
of the ARA recipes
If input AMRs include already alignments those alignments get shifted
If the input AMRs do not include alignments yet, the alignments get computed using
the FAA aligner from amrlib and those alignments get shifted
"""


def read_amr_file(recipe_amrs):
    """
    Read a file containing all instructions of a recipe with their corresponding AMRs in Penman notation
    :param recipe_amrs:
    :return: list of the sentences, list of the corresponding AMRs as penman string
            list of the unique AMR IDs
    """
    sentences = []
    amrs = []
    ids = []
    with open(recipe_amrs, "r", encoding="utf-8") as amr_file:
        current_amr_str = ""
        for line in amr_file:
            if line == '\n':
                amrs.append(current_amr_str)
                current_amr_str = ""
            elif line[0] == '#':
                splitted_line = line.strip().split(' ')
                tokens = splitted_line[2:]
                if splitted_line[1] == '::snt':
                    sentence = ' '.join(tokens)
                    sentences.append(sentence)
                elif splitted_line[1] == '::id':
                    id = ' '.join(tokens)
                    ids.append(id)
            else:
                current_amr_str += line.strip()
                current_amr_str += ' '

        if current_amr_str != "":
            amrs.append(current_amr_str)

    assert len(sentences) == len(amrs) == len(ids)

    return sentences, amrs, ids


def create_recipe_amr(recipe_amrs, output_file):
    """
    Take all amrs corresponding to a specific recipe, shift the alignments to reflect
    the sequential numbering of tokens in the complete recipe and not per sentence,
    and write all modified amrs into the output file
    :param recipe_amrs:
    :param output_file:
    :return:
    """
    instructions, amrs, amr_ids = read_amr_file(recipe_amrs)

    amrs_with_alignments = amrs

    shifted_amrs = []

    token_off_set = 1       # needs to start at 1 because ARA token IDs start at 1
    for instr_id in range(len(instructions)):

        instruction = instructions[instr_id]

        # shift the token ids in the amrs
        amr = amrs_with_alignments[instr_id]
        penman_amr = penman.decode(amr)         # convert to penman GRAPH for easier handling of alignments

        shifted_penman_amr = shift_alignments_amr(penman_amr, token_off_set)
        shifted_amrs.append(shifted_penman_amr)

        token_off_set += len(instruction.split(' '))

    with open(output_file, "w", encoding="utf-8") as out:
        for instr_id in range(len(instructions)):
            out.write(f'# ::id {amr_ids[instr_id]}\n')
            out.write(f'# ::snt {instructions[instr_id]}\n')
            amr = shifted_amrs[instr_id]
            out.write(f'{penman.encode(amr)}\n\n')


def shift_alignments_amr(penman_amr, offset):
    """
    Shift all alignments in the Penman AMR Graph by +offset
    :param penman_amr:
    :param offset:
    :return:
    """

    graph_epidata = penman_amr.epidata
    for instance in graph_epidata.keys():
        if not graph_epidata[instance]:
            continue

        for ep_data_ind, ep_data in enumerate(graph_epidata[instance]):
            if isinstance(ep_data, penman.surface.Alignment):  # or isinstance(ep_data, penman.surface.RoleAlignment):
                old_alignment = graph_epidata[instance].pop(ep_data_ind)
                old_id = old_alignment.indices[0]
                new_id = old_id + offset
                new_alignment = surface.Alignment.from_string(f'e.{new_id}')
                graph_epidata[instance].insert(ep_data_ind, new_alignment)

    return penman_amr


def create_alignments_corpus(corpus_dir, new_corpus_dir):
    """
    Adjusts all alignments, i.e. token ids, within one recipe file to match the token ids relative to the
    complete recipe text
    If the amrs in the corpus_dir do not yet include alignments, the alignments get first computed
    using the FAA aligner from amrlib
    :param corpus_dir: directory with the recipe sentence-level AMRs, one subdirectory per dish
    :param new_corpus_dir: parent directory for the created AMR files
                            will contain one subdirectory for each dish, each containing a folder 'amrs'
                            in these folders the amr.txt files will be placed
    :param aligned: whether the AMRs include already token IDs or not
    :return:
    """
    Path(new_corpus_dir).mkdir(exist_ok=True, parents=True)
    for dish in os.listdir(corpus_dir):
        Path('/'.join([new_corpus_dir, dish])).mkdir(exist_ok=True, parents=True)
        Path('/'.join([new_corpus_dir, dish, 'amrs'])).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir('/'.join([corpus_dir, dish])):

            create_recipe_amr('/'.join([corpus_dir, dish, recipe]), '/'.join([new_corpus_dir, dish, 'amrs', recipe]))


if __name__=="__main__":

    non_shifted_dir = "./aligned_cyclic_recipe_amrs_ara2"
    output_dir = "./aligned_cyclic_shifted_recipe_amrs_ara2"

    create_alignments_corpus(non_shifted_dir, output_dir)
