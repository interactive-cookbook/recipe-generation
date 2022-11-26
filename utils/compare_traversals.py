import os
from typing import List, Tuple
from graph_processing.recipe_graph import read_graph_from_conllu
import graph_processing.graph_traversal as grt
from utils.paths import ARA_DIR


def compare_traversals(action_dir):

    top_vs_id = {'equal': 0, 'non_equal': 0}
    top_vs_pf = {'equal': 0, 'non_equal': 0}
    top_vs_pf_lf = {'equal': 0, 'non_equal': 0}
    top_vs_pf_lf_id = {'equal': 0, 'non_equal': 0}
    id_vs_pf = {'equal': 0, 'non_equal': 0}
    id_vs_pf_lf = {'equal': 0, 'non_equal': 0}
    id_vs_pf_lf_id = {'equal': 0, 'non_equal': 0}
    pf_vs_pf_lf = {'equal': 0, 'non_equal': 0}
    pf_vs_pf_lf_id = {'equal': 0, 'non_equal': 0}
    pf_lf_vs_pf_lf_id = {'equal': 0, 'non_equal': 0}

    for dish in os.listdir(action_dir):
        dish_recipe_dir = os.path.join(action_dir, dish, 'recipes')
        for recipe in os.listdir(dish_recipe_dir):
            recipe_path = os.path.join(dish_recipe_dir, recipe)
            action_graph = read_graph_from_conllu(recipe_path)

            top_order = grt.order_actions_topological(action_graph)
            id_order = grt.order_actions_token_ids(action_graph)
            pf_order = grt.order_actions_pf(action_graph)
            pf_lf_order = grt.order_actions_pf_lf(action_graph)
            pf_lf_id_order = grt.order_actions_pf_lf_id(action_graph)

            eq, neq = compare_traversals_pairwise(top_order, id_order)
            top_vs_id['equal'] += eq
            top_vs_id['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(top_order, pf_order)
            top_vs_pf['equal'] += eq
            top_vs_pf['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(top_order, pf_lf_order)
            top_vs_pf_lf['equal'] += eq
            top_vs_pf_lf['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(top_order, pf_lf_id_order)
            top_vs_pf_lf_id['equal'] += eq
            top_vs_pf_lf_id['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(id_order, pf_order)
            id_vs_pf['equal'] += eq
            id_vs_pf['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(id_order, pf_lf_order)
            id_vs_pf_lf['equal'] += eq
            id_vs_pf_lf['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(id_order, pf_lf_id_order)
            id_vs_pf_lf_id['equal'] += eq
            id_vs_pf_lf_id['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(pf_order, pf_lf_order)
            pf_vs_pf_lf['equal'] += eq
            pf_vs_pf_lf['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(pf_order, pf_lf_id_order)
            pf_vs_pf_lf_id['equal'] += eq
            pf_vs_pf_lf_id['non_equal'] += neq
            eq, neq = compare_traversals_pairwise(pf_lf_order, pf_lf_id_order)
            pf_lf_vs_pf_lf_id['equal'] += eq
            pf_lf_vs_pf_lf_id['non_equal'] += neq

    print(f'top vs. id traversal: {top_vs_id}\n')
    print(f'top vs. pf traversal: {top_vs_pf}\n')
    print(f'top vs. pf-lf traversal: {top_vs_pf_lf}\n')
    print(f'top vs. pf-lf-id traversal:{top_vs_pf_lf_id}\n')
    print(f'id vs. pf traversal: {id_vs_pf}\n')
    print(f'id vs. pf-lf traversal: {id_vs_pf_lf}\n')
    print(f'id vs. pf-lf-id traversal: {id_vs_pf_lf_id}\n')
    print(f'pf vs. pf-lf traversal: {pf_vs_pf_lf}\n')
    print(f'pf vs. pf-lf-id traversal: {pf_vs_pf_lf_id}\n')
    print(f'pf-lf vs. pf-lf-id traversal: {pf_lf_vs_pf_lf_id}')


def compare_traversals_pairwise(trav_1: List, trav_2: List) -> Tuple[int, int]:

    equal = 0
    non_equal = 0

    preds1 = dict()
    preds2 = dict()

    for ind, ac in enumerate(trav_1):
        try:
            prev_ac = trav_1[ind-1]
            preds1[ac] = prev_ac
        except IndexError:
            preds1[ac] = -1

    for ind, ac in enumerate(trav_2):
        try:
            prev_ac = trav_2[ind-1]
            preds2[ac] = prev_ac
        except IndexError:
            preds2[ac] = -1

    acs1 = list(preds1.keys())
    acs2 = list(preds2.keys())
    acs1.sort()
    acs2.sort()
    assert acs1 == acs2

    for ac in preds1.keys():
        if preds1[ac] == preds2[ac]:
            equal += 1
        else:
            non_equal += 1

    return equal, non_equal


if __name__=='__main__':

    compare_traversals(ARA_DIR)
