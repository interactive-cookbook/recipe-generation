import penman
from penman import surface
import re
import os
from pathlib import Path
from align_spring_faa import get_alignments_recipe


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


def create_recipe_amr(recipe_amrs, output_file, aligned):
    """

    :param recipe_amrs:
    :param output_file:
    :param aligned:
    :return:
    """
    instructions, amrs, amr_ids = read_amr_file(recipe_amrs)

    if aligned:
        amrs_with_alignments = amrs
        alignment_strings = ['' for a in amrs_with_alignments]
    else:
        amrs_with_alignments, alignment_strings = get_alignments_recipe(instructions, amrs)

    shifted_amrs = []
    shifted_alignments = []

    token_off_set = 1       # needs to start at 1 because ARA token IDs start at 1
    for instr_id in range(len(instructions)):

        instruction = instructions[instr_id]

        # shift the token ids in the amrs
        amr = amrs_with_alignments[instr_id]
        penman_amr = penman.decode(amr)         # convert to penman GRAPH for easier handling of alignments

        shifted_penman_amr = shift_alignments_amr(penman_amr, token_off_set)
        shifted_amrs.append(shifted_penman_amr)

        # shift the token ids in the alignment strings
        # only if the alignments were created with FAA
        align_str = alignment_strings[instr_id]
        if not aligned:
            shifted_align_str = shift_alignments_str(align_str, token_off_set)
            shifted_alignments.append(shifted_align_str)
        else:
            shifted_alignments.append(align_str)

        token_off_set += len(instruction.split(' '))

    with open(output_file, "w", encoding="utf-8") as out:
        for instr_id in range(len(instructions)):
            out.write(f'# ::id {amr_ids[instr_id]}\n')
            out.write(f'# ::snt {instructions[instr_id]}\n')
            out.write(f'# ::alignments {shifted_alignments[instr_id]}\n')
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
        elif isinstance(graph_epidata[instance][0], penman.surface.Alignment):
            old_alignment = graph_epidata[instance].pop(0)
            old_id = old_alignment.indices[0]
            new_id = old_id + offset
            new_alignment = surface.Alignment.from_string(f'e.{new_id}')
            graph_epidata[instance].insert(0, new_alignment)

    return penman_amr


def shift_alignments_str(align_str, offset):
    """
    Shift all token ids in the alignments by +offset
    :param align_str:
    :param offset:
    :return:
    """
    shifted = []
    alignments = align_str.strip().split(' ')
    for al in alignments:
        token_id, node_id = al.split('-')
        int_token_id = int(token_id)
        shifted_token_id = int_token_id + offset
        shifted_alignment = f'{shifted_token_id}-{node_id}'
        shifted.append(shifted_alignment)

    new_alignments = ' '.join(shifted)
    return new_alignments


def create_alignments_corpus(corpus_dir, new_corpus_dir, aligned=True):
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
            create_recipe_amr('/'.join([corpus_dir, dish, recipe]), '/'.join([new_corpus_dir, dish, 'amrs', recipe]),
                              aligned)


if __name__=="__main__":

    create_alignments_corpus('../aligned_recipe_amrs_ibm_not_shifted',
                             '../aligned_recipe_amrs_ibm')
