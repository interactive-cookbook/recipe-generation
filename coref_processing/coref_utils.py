import json
import os.path
from typing import Dict, List, Tuple


def read_coref_file(coref_file: str) -> Dict[str, Dict]:
    """
    Read a coreference .jsonlines file containing the information for all recipes of the corpus
    :param coref_file: path to the coref file
    :return: dictionary with one dictionary per recipe with the information for the recipe
    """
    coref_dicts = dict()
    with open(coref_file, "r", encoding="utf-8") as cf:
        for line in cf:
            coref_dict = json.loads(line)
            recipe_name = coref_dict['recipe_id']
            coref_dicts[recipe_name] = coref_dict
    return coref_dicts


def read_joined_coref(recipe_coref_path) -> Tuple[List, List]:
    """
    Reads the split-rel clusters from the coref file
    :return: a list of the split-rel cluster dicts
            [{"token_id": id, "variable": v, "concept": c, "token": t, "amr_ids": [amrid1, amrid2, ...]},
            ...]
    """
    split_clusters = []
    coref_clusters = []
    with open(recipe_coref_path, 'r', encoding='utf-8') as cf:
        for line in cf.readlines():
            cluster = json.loads(line)
            cluster_type = list(cluster.keys())[0]
            if cluster_type.startswith('split-rel'):
                split_clusters.append(cluster[cluster_type])
            else:
                coref_clusters.append(cluster[cluster_type])

    return split_clusters, coref_clusters



def get_coref_clusters_extended(coref_file_path) -> Dict[str, Dict]:
    """
    Reads a coref_file and returns a dictionary with one sub dictionary per recipe_id
    each sub_dictionary is based on the corresponding data in the coref_file with the following
    changes:
    - the data with keys 'clusters', 'doc_key' and 'speakers' are removed
    - a new key-value pair is added: 'text' : [list of all tokens of the text]
    - if a PRO token is part of a cluster individually as well as as part of another span then
      it gets removed from that span if it is the first or last token and otherwise the compmlete
      mention gets removed
    :param coref_file_path:
    :return:
    """
    coref_data: dict = read_coref_file(coref_file_path)
    cleaned_coref_data = dict()
    for recipe_name in coref_data.keys():
        recipe_data = coref_data[recipe_name]
        cleaned_recipe_data = get_recipe_clusters_extended(recipe_data)
        cleaned_coref_data[recipe_name] = cleaned_recipe_data

    return cleaned_coref_data


def get_recipe_clusters_extended(coref_data: Dict) -> Dict:
    """

    :param coref_data:
    :return:
    """
    relevant_data = dict()
    for key, value in coref_data.items():
        if key in ['clusters', 'doc_key', 'speakers']:  # remove not needed data
            continue
        elif key == 'sentences':
            text = []
            for sent in value:
                text.extend(sent)
            relevant_data['text'] = text        # add the list of all tokens

            fixed_sents = fix_sentence_segmentation(coref_data['recipe_id'], text, coref_data['original_token_id'])
            value = value if not fixed_sents else fixed_sents
            sanity_check = []           # make sure the tokens of the text are still the same
            for sent in value:
                sanity_check.extend(sent)
            assert sanity_check == text

        elif key == 'predicted_clusters':
            value = remove_nested_clusters(coref_data)
        relevant_data[key] = value              # keep all other data unchanged

    return relevant_data


def remove_nested_clusters(coref_data: Dict) -> List[List[List[int]]]:
    """
    if a PRO token is part of a cluster individually as well as as part of another span then
      it gets removed from that span if it is the first or last token and otherwise the compmlete
      mention gets removed
    all other clusters are not changed
    :param coref_data:
    :return:
    """
    original_ids = coref_data['original_token_id']
    new_ids = coref_data['token_id']
    orig2new, new2orig = get_new_orig_id_mappings(original_ids, new_ids)

    cleaned_clusters = []
    mask_ids = [newid for newid in new2orig.keys() if new2orig[newid] == '[MASK]']

    for cluster in coref_data['predicted_clusters']:
        cleaned_cluster = []
        for span in cluster:
            tokens = [tid for tid in range(span[0], span[1] + 1)]
            if tokens[0] == tokens[-1]:
                cleaned_cluster.append(span)
            elif tokens[0] in mask_ids and not tokens[-1] in mask_ids:
                cleaned_cluster.append([span[0]+1, span[-1]])
            elif tokens[-1] in mask_ids and not tokens[0] in mask_ids:
                cleaned_cluster.append([span[0], span[-1]-1])
            elif tokens in mask_ids:        # then it cannot be removed that easily so remove span from cluster
                continue
            else:
                cleaned_cluster.append(span)
        cleaned_cluster = set([tuple(mention) for mention in cleaned_cluster])
        cleaned_cluster = [list(mention) for mention in cleaned_cluster]
        cleaned_cluster.sort()
        cleaned_clusters.append(cleaned_cluster)

    return cleaned_clusters


def get_coref_clusters_original(coref_file_path):
    """

    :param coref_file_path:
    :return:
    """
    coref_data: dict = read_coref_file(coref_file_path)
    cleaned_coref_data = dict()
    for recipe_name in coref_data.keys():
        recipe_data = coref_data[recipe_name]
        cleaned_recipe_data = get_recipe_clusters_original(recipe_data)
        cleaned_coref_data[recipe_name] = cleaned_recipe_data

    return cleaned_coref_data


def get_recipe_clusters_original(coref_data: dict):
    """

    :param coref_data:
    :return:
    """
    original_coref_dict = dict()
    original_ids = coref_data['original_token_id']
    new_ids = coref_data['token_id']
    orig2new, new2orig = get_new_orig_id_mappings(original_ids, new_ids)

    tokens = []
    sent_map = []
    for sent_id, sentence in enumerate(coref_data['sentences']):
        tokens.extend(sentence)
        sent_ids = [sent_id for i in range(len(sentence))]
        sent_map.extend(sent_ids)

    # keep only those tokens that were in the original text
    cleaned_tokens = []
    cleaned_sentences = [[] for i in range(len(coref_data['sentences']))]
    for t_id, t in enumerate(tokens):
        assert t_id == new_ids[t_id]
        orig_id = new2orig[t_id]
        if orig_id != '[MASK]':
            cleaned_tokens.append(t)
            sent_id = sent_map[t_id]    # get id of corresponding sentence
            cleaned_sentences[sent_id].append(t)    # add token to appropriate sentence

    cleaned_token_ids = [t_id for t_id in original_ids if t_id != '[MASK]']

    # fix sentence segmentation if comparison file available
    fixed_sentences = fix_sentence_segmentation(recipe_id=coref_data['recipe_id'],
                                                  coref_tokens=cleaned_tokens,
                                                  orig_token_ids=cleaned_token_ids)
    cleaned_sentences = cleaned_sentences if not fixed_sentences else fixed_sentences

    original_coref_dict['sentences'] = cleaned_sentences
    original_coref_dict['text'] = cleaned_tokens
    original_coref_dict['token_id'] = cleaned_token_ids

    # need to shift the ids in the predicted clusters
    pred_clusters = coref_data['predicted_clusters']
    cleaned_clusters = []
    for cluster in pred_clusters:
        cleaned_cl = []
        for mention in cluster:
            new_start_id = mention[0]
            new_end_id = mention[1]
            orig_start_id = new2orig[new_start_id]
            orig_end_id = new2orig[new_end_id]
            # need to remove the token IDs of the masked tokens, and hopefully, they will never be included within
            # a mention
            while True:
                if orig_start_id == '[MASK]' and not orig_end_id == '[MASK]':
                    orig_start_id = new2orig[new_start_id + 1]
                elif orig_end_id == '[MASK]' and not orig_start_id == '[MASK]':
                    orig_end_id = new2orig[new_end_id - 1]
                elif orig_start_id == '[MASK]' and orig_end_id == '[MASK]':
                    break       # then the complete mention was newly added
                else:
                    cleaned_cl.append([orig_start_id, orig_end_id])
                    break

        if len(cleaned_cl) > 1:         # remove clusters that are now singletaries
            cleaned_clusters.append(cleaned_cl)
    #if len(cleaned_clusters):
        #print(coref_data['recipe_id'])
    original_coref_dict['predicted_clusters'] = cleaned_clusters

    return original_coref_dict


def get_new_orig_id_mappings(original_ids, new_ids):
    """

    :param original_ids: list of the original token IDs where the original ID is '[MASK]' in case the token was added
    :param new_ids: list of the new token ids, so essentially new_ids[x] = x
    :return: two dictionaries with the mappings between the new and the original token IDs
    """
    orig2new = dict()
    new2orig = dict()
    for orig_id, new_id in zip(original_ids, new_ids):
        if orig_id != '[MASK]':
            orig2new[orig_id] = new_id
        new2orig[new_id] = orig_id

    return orig2new, new2orig


# TODO: enable to use the ara2 corpus as well
def fix_sentence_segmentation(recipe_id: str, coref_tokens: List[str], orig_token_ids: List[int]):
    """
    Adjust the sentence segmentation from the coreference data files such that it matches with the
    segmentation from the microsoft corpus
    :param recipe_id:
    :param coref_tokens:
    :param orig_token_ids:
    :return:
    """
    dish_name = '_'.join(recipe_id.split('_')[:-1])
    gold_segmentation_file = os.path.join('..', 'data', 'amr_input_data', dish_name, recipe_id + '_sentences.txt')
    try:
        f = open(gold_segmentation_file, 'r', encoding='utf-8')
        f.close()
    except FileNotFoundError:
        return None

    sentence_end_ids = []
    shift_value = 0
    with open(gold_segmentation_file, 'r', encoding='utf-8') as gsf:
        for g_sentence in gsf.readlines():
            tokens = g_sentence.strip().split()
            last_token_id = shift_value + len(tokens) - 1
            shift_value += len(tokens)
            sentence_end_ids.append(last_token_id)

    new_segmented_sentences = []
    current_start = 0
    for current_end in sentence_end_ids:
        start_index = orig_token_ids.index(current_start)
        end_index = orig_token_ids.index(current_end)
        # check whether next token would be an added pronoun token, if yes need to add it because
        # otherwise it would get lost
        try:
            if orig_token_ids[end_index + 1] == '[MASK]':
                tokens_sentence = coref_tokens[start_index:end_index + 2]
            else:
                tokens_sentence = coref_tokens[start_index:end_index + 1]
        except IndexError:      # if we are at the sentence end
            tokens_sentence = coref_tokens[start_index:end_index + 1]
        current_start = current_end + 1
        new_segmented_sentences.append(tokens_sentence)

    return new_segmented_sentences

