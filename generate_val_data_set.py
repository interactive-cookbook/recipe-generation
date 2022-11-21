import os
from pathlib import Path
from utils.paths import DATA_DIR, ARA_DIR, MODEL_DIR, PROJ_DIR
from generate_recipe import generate_different_orderings
from graph_processing.recipe_graph import read_graph_from_conllu


def generate_ara1():
    assert os.path.join(DATA_DIR).split(os.sep)[-1] == 'data'
    recipe_names = []
    with open('training/ara1_val_split.tsv', 'r', encoding='utf-8') as val_f:
        for line in val_f.readlines():
            recipe_names.append(line.strip())

    generate_from_conllu_file(recipe_names)


def generate_ara2():
    assert os.path.join(DATA_DIR).split(os.sep)[-1] == 'data_ara2'
    recipe_names = []
    with open('training/ara2_val_split.tsv', 'r', encoding='utf-8') as val_f:
        for line in val_f.readlines():
            recipe_names.append(line.strip())

    generate_from_conllu_file(recipe_names)


def generate_from_conllu_file(recipe_names: list):

    for recipe_name in recipe_names:
        dish = recipe_name.split('_')[:-1]
        dish = '_'.join(dish)
        ac_graph_path = os.path.join(ARA_DIR, dish, 'recipes', recipe_name + '.conllu')

        action_graph = read_graph_from_conllu(ac_graph_path)
        configuration_file = MODEL_DIR / Path('recipe_gen_config.json')
        ordering_list = ['top', 'ids', 'pf', 'pf-lf', 'pf-lf-id']
        ordering_names = {'top': 'NetworkX Topological Order', 'ids': 'Token ID Ordering', 'pf': 'Path-First Ordering',
                          'pf-lf': 'Path-First Longest-First Ordering', 'pf-lf-id': 'Path-First Longest-First IDs Ordering'}
        context_len = 1
        output_file = os.path.join(PROJ_DIR, 'output', recipe_name+'_generated.txt')

        generate_different_orderings(ac_graph=action_graph, configuration_file=configuration_file,
                                     ordering_list=ordering_list, ordering_names=ordering_names,
                                     context_len=context_len, output_file= output_file)

if __name__=='__main__':

    generate_ara1()
