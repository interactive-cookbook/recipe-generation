from amrlib.alignments.faa_aligner import FAA_Aligner
import penman
from penman import surface
import re
import os
from pathlib import Path


def read_amr_file(recipe_amrs):
    """
    Read a file containing all instructions of a recipe with their corresponding AMRs in Penman notation
    :param recipe_amrs:
    :return: list of the sentences, list of the corresponding AMRs as penman string
    """
    sentences = []
    amrs = []
    with open(recipe_amrs, "r", encoding="utf-8") as amr_file:
        current_amr_str = ""
        for line in amr_file:
            if line == '\n':
                amrs.append(current_amr_str)
                current_amr_str = ""
            elif line[0] == '#':
                splitted_line = line.strip().split(' ')
                tokens = splitted_line[2:]
                sentence = ' '.join(tokens)
                sentences.append(sentence)
            else:
                current_amr_str += line.strip()
                current_amr_str += ' '

        if current_amr_str != "":
            amrs.append(current_amr_str)

    assert len(sentences) == len(amrs)

    return sentences, amrs


def get_alignments_recipe(recipe_amrs):
    """
    Reads all recipe AMRs from recipe_amrs and computes the token-node alignments using FAA aligner
    Alignmnents are based on individual sentences -> enumeration of tokens starts at 0 for each sentence
    :param recipe_amrs:
    :return: list of the sentences, list of the AMRs with the ISI alignment information, list of the alignment strings
    """
    aligner = FAA_Aligner()
    sentences, amrs = read_amr_file(recipe_amrs)
    lowered_sentences = [sent.lower() for sent in sentences]
    amrs_w_alignments, alignment_strings = aligner.align_sents(lowered_sentences, amrs)

    assert len(sentences) == len(amrs_w_alignments) == len(alignment_strings)

    return sentences, amrs_w_alignments, alignment_strings


def create_recipe_amr(recipe_amrs, output_file):
    """

    :param recipe_amrs:
    :param output_file:
    :return:
    """
    instructions, amrs_with_alignments, alignment_strings = get_alignments_recipe(recipe_amrs)
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
        align_str = alignment_strings[instr_id]
        shifted_align_str = shift_alignments_str(align_str, token_off_set)
        shifted_alignments.append(shifted_align_str)

        token_off_set += len(instruction.split(' '))

    with open(output_file, "w", encoding="utf-8") as out:
        for instr_id in range(len(instructions)):
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


def create_alignments_corpus(corpus_dir, new_corpus_dir):
    """
    Creates the node-token alignments for all the AMRs for all recipes in the subdirectories of corpus_dir
    Alignments, i.e. token ids, within one recipe file are adjusted to match the token ids relative to the
    complete recipe text
    :param corpus_dir:
    :param new_corpus_dir:
    :return:
    """
    Path(new_corpus_dir).mkdir(exist_ok=True, parents=True)
    for dish in os.listdir(corpus_dir):
        Path('/'.join([new_corpus_dir, dish])).mkdir(exist_ok=True, parents=True)
        Path('/'.join([new_corpus_dir, dish, 'amrs'])).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir('/'.join([corpus_dir, dish])):

            create_recipe_amr('/'.join([corpus_dir, dish, recipe]),
                              '/'.join([new_corpus_dir, dish, 'amrs', recipe]))


if __name__=="__main__":

    create_alignments_corpus('./recipe_sentence_amrs',
                             './aligned_recipe_amrs')
