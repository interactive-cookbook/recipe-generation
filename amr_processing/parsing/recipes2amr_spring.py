import os
from pathlib import Path
import amrlib
import torch.cuda


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


def parse_recipe_spring(parsing_model, recipe_text_file):
    """

    :param parsing_model: a pretrained amr parsing model from amrlib
    :param recipe_text_file: the recipe .txt file to parse, one instruction per line
    :return: a list of the AMR graphs for each of the sentences
    """
    recipe_instructions = get_sentences(recipe_text_file)
    amr_graphs = parsing_model.parse_sents(recipe_instructions)

    return amr_graphs


def parse_recipe_corpus_spring(corpus_dir, graph_dir):
    """
    Run the amrlib parsing model to parse all recipes in the subdirectories of corpus_dir sentence by sentence
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
    torch.cuda.empty_cache()
    parse_model = amrlib.load_stog_model(device='cpu')

    # create directories and run parser on each recipe file
    Path(graph_dir).mkdir(exist_ok=True, parents=True)
    for dish in os.listdir(corpus_dir):
        Path('/'.join([graph_dir, dish])).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir('/'.join([corpus_dir, dish])):
            # parse one recipe file
            instruction_amrs = parse_recipe_spring(parse_model, '/'.join([corpus_dir, dish, recipe]))

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

    parse_recipe_corpus_spring("../Corpora/Mapped_Ara/amr_input_data", "./recipe_sentence_amrs_test")

