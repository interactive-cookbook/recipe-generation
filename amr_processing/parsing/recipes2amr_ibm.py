import os
from pathlib import Path
from transition_amr_parser.parse import AMRParser
import penman
from penman.surface import Alignment


checkpoint_path = "../../transition-amr-parser-master/transition-amr-parser-master/DATA/AMR3.0/models/amr3.0-structured-bart-large-neur-al-sampling5/seed42/checkpoint_wiki.smatch_top5-avg.pt"


def get_sentences(recipe_text_file):
    """
    Reads the new-line separated sentences from recipe file
    :param recipe_text_file: the recipe .txt file, one instruction per line
    :return: a list of all sentences / instructions (strings)
    """
    sentences = []
    with open(recipe_text_file, "r", encoding="utf-8") as text:
        for line in text:
            sentences.append(line.strip())
    return sentences


def parse_recipe_ibm(parsing_model, recipe_text_file):
    """
    Parses all sentences in a recipe text using a pretrained checkpoint from the Struct-BART AMR parser
    :param parsing_model: a loaded pretrained parsing model
    :param recipe_text_file: the recipe file, one instruction per line
    :return:
    """

    amr_graphs = []
    recipe_instructions = get_sentences(recipe_text_file)

    for instruction in recipe_instructions:
        tokens = instruction.split(' ')
        annotations, decoding_data = parsing_model.parse_sentence(tokens)

        # get the part of the output string that are the tokens and remove '# ::tok'
        toks = annotations.split('\n')[0]
        toks  = toks.split(' ')
        toks = toks[2:]
        toks = ' '.join(toks)

        # get the part of the output string that is the amr graph
        amr_str = annotations.split('\n')[1:]
        amr_str = ' '.join(amr_str)

        # convert into penman Graph and add the 'e.' prefix for the alignments
        penman_gr = penman.decode(amr_str)
        epi_dat = penman_gr.epidata
        for trip, ep in epi_dat.items():
            try:
                if isinstance(ep[0], Alignment):
                    ep[0].prefix = 'e.'
            except IndexError:
                continue

        amr_to_write = f'# ::snt {toks}\n{penman.encode(penman_gr)}'
        amr_graphs.append(amr_to_write)

    return amr_graphs


def parse_recipe_corpus_ibm(corpus_dir, graph_dir):
    """
    Run the Struct-BART parsing model to parse all recipes in the subdirectories of corpus_dir sentence by sentence
    into AMRs
    :param corpus_dir: parent directory of the recipe corpus, one subfolder per dish, recipes files have one
                        instruction per line
    :param graph_dir: directory where the files with amrs should be written to
                      directory will have one subdirectory for each dish, each containing a folder 'amrs' with the
                      created files
                      file naming: if original file is named 'baked_ziti_0_sentences.txt' then the file with the amrs
                                    will be named 'baked_ziti_0_sentences_amrs.txt'
    :return:
    """

    # get parser
    parse_model = AMRParser.from_checkpoint(checkpoint_path)

    # create directories and run parser on each recipe file
    Path(graph_dir).mkdir(exist_ok=True, parents=True)
    for dish in os.listdir(corpus_dir):
        Path('/'.join([graph_dir, dish])).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir('/'.join([corpus_dir, dish])):

            # parse one recipe file
            instruction_amrs = parse_recipe_ibm(parse_model, '/'.join([corpus_dir, dish, recipe]))

            # get name for file with recipe amrs
            recipe_name = recipe.split('.')[:-1]
            recipe_name = '.'.join(recipe_name)
            new_file_name = recipe_name + '_amr.txt'
            # and name for creating a unique AMR ID
            recipe_name_wo_suffix = recipe_name.split('_')[:-1]
            recipe_name_wo_suffix = '_'.join(recipe_name_wo_suffix)

            # write file
            with open('/'.join([graph_dir, dish, new_file_name]), "w", encoding="utf-8") as gr_file:
                for instr_id, amr in enumerate(instruction_amrs):
                    gr_file.write(f'# ::id {recipe_name_wo_suffix}_instr{instr_id}\n')
                    gr_file.write(f'{amr}\n\n')


if __name__=="__main__":

    parse_recipe_corpus_ibm("../../Corpora/Mapped_Ara/amr_input_data",
                            "../aligned_recipe_amrs_ibm_not_shifted")


