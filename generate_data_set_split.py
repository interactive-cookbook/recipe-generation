import os
from pathlib import Path
import argparse
from typing import List, Union

from utils.paths import DATA_DIR, ARA_DIR, MODEL_DIR, PROJ_DIR
from generate_recipe import generate_recipe_different_orderings, generate_recipe_one_ordering
from graph_processing.recipe_graph import read_graph_from_conllu

# TODO: document

def generate_specific_split(split_file, split_type, config_file, context_len, orderings, output_dir=None):
    """

    :param split_file:
    :param config_file:
    :param context_len:
    :param split_type:
    :param orderings:
    :param output_dir:
    :return:
    """
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
                              orderings: List,
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

        if len(orderings) > 1:
            generate_recipe_different_orderings(ac_graph=action_graph, configuration_file=configuration_file,
                                                ordering_list=orderings, ordering_names=ordering_names,
                                                context_len=context_len, output_file=output_file)
        else:
            generate_recipe_one_ordering(ac_graph=action_graph, configuration_file=configuration_file,
                                         ordering=orderings[0], context_len=context_len,
                                         output_file=output_file)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--type', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cont', required=True)
    parser.add_argument('--order', required=False)
    parser.add_argument('--out', required=False)

    args = parser.parse_args()

    # needed for the headers if a csv file with recipes for different traversals gets created
    ordering_names = {'top': 'NetworkX Topological Order', 'ids': 'Token ID Ordering', 'pf': 'Path-First Ordering',
                      'pf-lf': 'Path-First Longest-First Ordering', 'pf-lf-id': 'Path-First Longest-First IDs Ordering'}

    ordering = args.order if args.order else 'pf-lf-id'
    possible_orderings = ['top', 'ids', 'pf', 'pf-lf', 'pf-lf-id']
    output = args.out if args.out else None

    if ordering == 'all':
        generate_specific_split(split_file=args.split,
                                split_type=args.type,
                                config_file=args.config,
                                context_len=args.cont,
                                orderings=possible_orderings,
                                output_dir=output)
    else:
        assert ordering in possible_orderings
        generate_specific_split(split_file=args.split,
                                split_type=args.type,
                                config_file=args.config,
                                context_len=args.cont,
                                orderings=[ordering],
                                output_dir=output)
