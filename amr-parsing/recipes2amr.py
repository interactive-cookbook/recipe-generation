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


def parse_recipe(parsing_model, recipe_text_file):
    """

    :param parsing_model: a pretrained amr parsing model from amrlib
    :param recipe_text_file: the recipe .txt file to parse, one instruction per line
    :return: a list of the AMR graphs for each of the sentences
    """
    recipe_instructions = get_sentences(recipe_text_file)
    amr_graphs = parsing_model.parse_sents(recipe_instructions)

    return amr_graphs


def parse_recipe_corpus(corpus_dir, graph_dir):
    """
    Run the amrlib parsing model to parse all recipes in the subdirectories of corpus_dir sentence by sentence
    into AMRs
    :param corpus_dir: parent directory of the recipe corpus, one subfolder per dish, recipes files have one
                        instruction per line
    :param graph_dir:
    :return:
    """
    torch.cuda.empty_cache()
    parse_model = amrlib.load_stog_model(device='cpu')
    Path(graph_dir).mkdir(exist_ok=True, parents=True)
    for dish in os.listdir(corpus_dir):
        Path('/'.join([graph_dir, dish])).mkdir(exist_ok=True, parents=True)
        for recipe in os.listdir('/'.join([corpus_dir, dish])):
            # parse one recipe file
            instruction_amrs = parse_recipe(parse_model, '/'.join([corpus_dir, dish, recipe]))
            # get name for file with recipe amrs
            new_file_name = recipe.split('.')[:-1]
            new_file_name = '.'.join(new_file_name)
            new_file_name = new_file_name + '_amr.txt'
            # write file
            with open('/'.join([graph_dir, dish, new_file_name]), "w", encoding="utf-8") as gr_file:
                for amr in instruction_amrs:
                    gr_file.write(f'{amr}\n\n')



if __name__=="__main__":

    parse_recipe_corpus("../Corpora/Mapped_Ara/amr_input_data", "./recipe_sentence_amrs")

