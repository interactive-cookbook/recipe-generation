import os
from pathlib import Path
from coref_utils import read_coref_file
from utils.paths import DATA_DIR


def create_explicit_recipe_texts(coreference_file_path):

    corpus_coref_data: dict = read_coref_file(coreference_file_path)

    output_dir = os.path.join('../data_ara1_explicit', 'recipe_texts_explicit')
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for recipe_name in corpus_coref_data.keys():
        dish_name = '_'.join(recipe_name.split('_')[:-1])
        dish_dir = os.path.join(output_dir, dish_name)
        Path(dish_dir).mkdir(exist_ok=True, parents=True)

        recipe_coref_data = corpus_coref_data[recipe_name]
        text = []
        with open(os.path.join(dish_dir, recipe_name + '.txt'), 'w', encoding='utf-8') as rf:
            for sentence in recipe_coref_data['sentences']:
                sent = ' '.join(sentence)
                rf.write(f'{sent}\n')
                text.extend(sentence)
        assert len(text) == len(recipe_coref_data['token_id']) == len(recipe_coref_data['original_token_id'])


if __name__=='__main__':

    create_explicit_recipe_texts('./ara_pronoun_merged_pred.jsonlines')
