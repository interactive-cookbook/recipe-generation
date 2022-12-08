import os
from pathlib import Path
from coref_utils import get_coref_clusters_extended, get_new_orig_id_mappings


def create_explicit_action_corpus(original_ara_dir, shifted_ara_dir, coref_file):

    coref_data_complete = get_coref_clusters_extended(coref_file)
    original_ara_dir = Path(original_ara_dir)
    shifted_ara_dir = Path(shifted_ara_dir)

    Path(shifted_ara_dir).mkdir(exist_ok=True, parents=True)
    for dish in os.listdir(original_ara_dir):
        dish_dir = os.path.join(original_ara_dir, dish, 'recipes')
        shifted_dish_dir = os.path.join(shifted_ara_dir, dish, 'recipes')
        Path(shifted_dish_dir).mkdir(exist_ok=True, parents=True)

        for recipe in os.listdir(dish_dir):
            recipe_name = '.'.join(recipe.split('.')[:-1])
            coref_data = coref_data_complete[recipe_name]
            # need to shift the ids to start at 1
            new_ids = [t_id + 1 for t_id in coref_data['token_id']]
            orig_ids = [t_id + 1 if isinstance(t_id, int) else t_id for t_id in coref_data['original_token_id']]
            new_tokens = coref_data['text']
            orig2new, new2orig = get_new_orig_id_mappings(orig_ids, new_ids)

            original_lines = dict()
            with open(os.path.join(dish_dir, recipe), 'r', encoding='utf-8') as orig_ac:
                for line in orig_ac:
                    columns = line.strip().split('\t')
                    orig_token_id = int(columns[0])
                    original_lines[orig_token_id] = columns

            new_lines = []
            for t_id, t in zip(new_ids, new_tokens):
                orig_t_id = new2orig[t_id]
                if orig_t_id != '[MASK]':
                    orig_l = original_lines[orig_t_id]
                    orig_edge_id = int(orig_l[6])
                    if orig_edge_id == 0:   # keep 0 as "no parent"
                        new_edge_id = 0
                    else:
                        new_edge_id = orig2new[orig_edge_id]
                    assert t == orig_l[1]
                    new_l = [str(t_id), t, orig_l[2], orig_l[3], orig_l[4], orig_l[5],
                             str(new_edge_id), orig_l[7], orig_l[8], orig_l[9]]
                    new_lines.append(new_l)

                else:
                    new_l = [str(t_id), t, '_', '_', 'O', '_', '0', 'root', '_', '_']
                    new_lines.append(new_l)

            with open(os.path.join(shifted_dish_dir, recipe), 'w', encoding='utf-8') as new_ac:
                for line in new_lines:
                    line_str = '\t'.join(line)
                    new_ac.write(f'{line_str}\n')


if __name__=='__main__':

    create_explicit_action_corpus('../data/ara1.1', '../data_ara1_explicit/ara1_exp', './ara_explicit_merged_pred.jsonlines')
