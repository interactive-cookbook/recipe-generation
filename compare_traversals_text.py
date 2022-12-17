import os
from pathlib import Path
from itertools import combinations

from graph_processing.recipe_graph import read_graph_from_conllu
from generate_recipe import generate_recipe_different_orderings
from utils.paths import MODEL_DIR, ARA_DIR


def compare_texts_different_traversals(action_dir, config_file):
    """

    :param action_dir:
    :return:
    """
    configuration_file = MODEL_DIR / Path(config_file)
    ordering_list = ['top', 'ids', 'pf', 'pf-lf', 'pf-lf-id']
    ordering_names = {'top': 'NetworkX Topological Order', 'ids': 'Token ID Ordering', 'pf': 'Path-First Ordering',
                      'pf-lf': 'Path-First Longest-First Ordering', 'pf-lf-id': 'Path-First Longest-First IDs Ordering'}
    context_len = 1

    traversal_combinations = list(combinations(ordering_list, 2))
    results = dict()
    for comb in traversal_combinations:
        results[comb] = {'diff_text_diff_order': 0,
                         'diff_text_eq_order': 0,
                         'eq_text_diff_order': 0,
                         'eq_text_eq_order': 0}

    for dish in os.listdir(action_dir):
        dish_dir = os.path.join(action_dir, dish, 'recipes')
        for recipe in os.listdir(dish_dir):
            recipe_file = os.path.join(dish_dir, recipe)
            action_graph = read_graph_from_conllu(recipe_file)

            generated_sentences, action_orderings = generate_recipe_different_orderings(ac_graph=action_graph,
                                                                                        configuration_file=configuration_file,
                                                                                        ordering_list=ordering_list,
                                                                                        ordering_names=ordering_names,
                                                                                        context_len=context_len,
                                                                                        output_file=None)

            for (trav1, trav2) in traversal_combinations:
                sentences_trav1 = generated_sentences[trav1]
                sentences_trav2 = generated_sentences[trav2]
                action_order_trav1 = action_orderings[trav1]
                action_order_trav2 = action_orderings[trav2]
                action2sent_trav1, action2prev_trav1 = get_action_mappings(sentences_trav1, action_order_trav1)
                action2sent_trav2, action2prev_trav2 = get_action_mappings(sentences_trav2, action_order_trav2)

                for action in action2sent_trav1.keys():
                    sent_eq = action2sent_trav1[action] == action2sent_trav2[action]
                    prev_ac_eq = action2prev_trav1[action] == action2prev_trav2[action]
                    if sent_eq and prev_ac_eq:
                        results[(trav1, trav2)]['eq_text_eq_order'] += 1
                    elif sent_eq and not prev_ac_eq:
                        results[(trav1, trav2)]['eq_text_diff_order'] += 1
                    elif not sent_eq and prev_ac_eq:
                        results[(trav1, trav2)]['diff_text_eq_order'] += 1
                    else:
                        results[(trav1, trav2)]['diff_text_diff_order'] += 1

    return results


def get_action_mappings(sentence_list, action_list):
    """

    :param sentence_list:
    :param action_list:
    :return:
    """
    action2sentence = dict()
    action2prev = dict()

    prev_action = -1
    for sentence, action in zip(sentence_list, action_list):
        action_tup = tuple(sorted(action))
        action2sentence[action_tup] = sentence
        action2prev[action_tup] = prev_action
        prev_action = action

    return action2sentence, action2prev


def save_comparison_results(results: dict, output_file):
    """

    :param results:
    :param output_file:
    :return:
    """
    with open(output_file, 'w', encoding='utf-8') as out:
        for traversal_pair, traversal_counts in results.items():
            out.write(f'Comparing {traversal_pair[0]} vs. {traversal_pair[1]}:\n')
            for key, count in traversal_counts.items():
                out.write(f'{key}: {count}\n')
            out.write('\n')


if __name__=='__main__':

    results = compare_texts_different_traversals(ARA_DIR, 'model_context_config.json')
    save_comparison_results(results, './traversal_comp_ara1.txt')
