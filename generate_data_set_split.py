import os
from pathlib import Path
from utils.paths import DATA_DIR, ARA_DIR, MODEL_DIR, PROJ_DIR
from generate_recipe import generate_recipe_different_orderings, generate_recipe_one_ordering
from graph_processing.recipe_graph import read_graph_from_conllu
from typing import List, Union


def generate_ara1(split_file, split_type, config_file, context_len, orderings=None, output_dir=None):
    """

    :param split_file:
    :param config_file:
    :param context_len:
    :param split_type:
    :param orderings:
    :param output_dir:
    :return:
    """
    assert os.path.join(DATA_DIR).split(os.sep)[-1] == 'data'
    recipe_names = read_split_file_names(split_file, split_type)

    generate_from_conllu_file(recipe_names=recipe_names,
                              configuration_file=config_file,
                              context_len=context_len,
                              orderings=orderings,
                              output_dir=output_dir)


def generate_ara2(split_file, split_type, config_file, context_len, orderings=None, output_dir=None):
    """

    :param split_file:
    :param config_file:
    :param split_type:
    :param orderings:
    :param output_dir:
    :return:
    """
    assert os.path.join(DATA_DIR).split(os.sep)[-1] == 'data_ara2'
    recipe_names = read_split_file_names(split_file, split_type)

    generate_from_conllu_file(recipe_names=recipe_names,
                              configuration_file=config_file,
                              context_len=context_len,
                              orderings=orderings,
                              output_dir=output_dir)


def generate_ara1_explicit(split_file, split_type, config_file, context_len, orderings=None, output_dir=None):
    """

    :param split_file:
    :param config_file:
    :param split_type:
    :param orderings:
    :param output_dir:
    :return:
    """
    assert os.path.join(DATA_DIR).split(os.sep)[-1] == 'data_ara1_explicit'
    recipe_names = read_split_file_names(split_file, split_type)

    generate_from_conllu_file(recipe_names=recipe_names,
                              configuration_file=config_file,
                              context_len=context_len,
                              orderings=orderings,
                              output_dir=output_dir)


def read_split_file_names(split_file, split_type) -> List[str]:
    """

    :param split_file:
    :param split_type:
    :return:
    """
    assert split_type in ['train', 'val', 'test']
    split_recipes = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            s_type, recipe_gold_file_path = line.strip().split('\t')
            if s_type == split_type:
                recipe_gold_file = recipe_gold_file_path.split(os.sep)[-1]
                recipe_name = recipe_gold_file.replace('_gold.txt', '')
                split_recipes.append(recipe_name)
    return split_recipes


def generate_from_conllu_file(recipe_names: List[str],
                              configuration_file: Union[str, Path],
                              context_len: int,
                              orderings: Union[None, List] = None,
                              output_dir: Union[str, Path] = None):
    """

    :param recipe_names:
    :param configuration_file:
    :param context_len:
    :param orderings:
    :param output_dir:
    :return:
    """
    configuration_file = MODEL_DIR / Path(configuration_file)
    all_possible_orderings = ['top', 'ids', 'pf', 'pf-lf', 'pf-lf-id']
    if not orderings:
        ordering_list = all_possible_orderings
    else:
        ordering_list = orderings
        assert not [ord for ord in orderings if not ord in all_possible_orderings]
    ordering_names = {'top': 'NetworkX Topological Order', 'ids': 'Token ID Ordering', 'pf': 'Path-First Ordering',
                      'pf-lf': 'Path-First Longest-First Ordering', 'pf-lf-id': 'Path-First Longest-First IDs Ordering'}



    for recipe_name in recipe_names:
        dish = recipe_name.split('_')[:-1]
        dish = '_'.join(dish)
        ac_graph_path = os.path.join(ARA_DIR, dish, 'recipes', recipe_name + '.conllu')

        action_graph = read_graph_from_conllu(ac_graph_path)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = os.path.join(PROJ_DIR, output_dir, recipe_name+'_generated.txt')
        else:
            output_file = os.path.join(PROJ_DIR, 'output', recipe_name+'_generated.txt')

        if len(ordering_list) > 1:
            generate_recipe_different_orderings(ac_graph=action_graph, configuration_file=configuration_file,
                                                ordering_list=ordering_list, ordering_names=ordering_names,
                                                context_len=context_len, output_file=output_file)
        else:
            generate_recipe_one_ordering(ac_graph=action_graph, configuration_file=configuration_file,
                                         ordering=ordering_list[0], context_len=context_len,
                                         output_file=output_file)


if __name__=='__main__':

    """
    generate_ara1_explicit('./final_ara1_split.tsv',
                            'test',
                            'model_explicit_config.json',
                            'context_len'=1
                            ['pf-lf-id'],
                            Path('output/output_explicit'))
    """

    generate_ara1(split_file='./final_ara1_split.tsv',
                  split_type='test',
                  config_file='model_no_context_config.json',
                  context_len=0,
                  orderings=['pf-lf-id'],
                  output_dir=Path('output/output_no_context'))

    generate_ara1(split_file='./final_ara1_split.tsv',
                  split_type='test',
                  config_file='model_context_config.json',
                  context_len=1,
                  orderings=['pf-lf-id'],
                  output_dir=Path('output/output_context'))