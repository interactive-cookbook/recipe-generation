import json


def read_coref_file(coref_file: str):
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


def get_coref_clusters_extended(coref_file_path):
    """

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


def get_recipe_clusters_extended(coref_data: dict):
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
        elif key == 'predicted_clusters':
            value = remove_nested_clusters(coref_data)
        relevant_data[key] = value              # keep all other data unchanged

    return relevant_data


def remove_nested_clusters(coref_data):
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
            elif tokens[0] in mask_ids:
                cleaned_cluster.append([span[0]+1, span[-1]])
            elif tokens[-1] in mask_ids:
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

    original_coref_dict['sentences'] = cleaned_sentences
    original_coref_dict['text'] = cleaned_tokens
    cleaned_token_ids = [t_id for t_id in original_ids if t_id != '[MASK]']
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

    :param original_ids:
    :param new_ids:
    :return:
    """
    orig2new = dict()
    new2orig = dict()
    for orig_id, new_id in zip(original_ids, new_ids):
        if orig_id != '[MASK]':
            orig2new[orig_id] = new_id
        new2orig[new_id] = orig_id

    return orig2new, new2orig

