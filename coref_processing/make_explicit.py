from coref_utils import get_coref_clusters_extended


def make_mentions_explicit(coref_file, new_coref_file):

    coref_data = get_coref_clusters_extended(coref_file)
    new_coref_data = dict()

    for recipe_name, data in coref_data.items():
        original_token_ids = data['original_token_id']
        pronoun_sentences = data['sentences']
        pronoun_tokens = data['text']
        pronoun_token_ids = data['token_ids']
        predicted_clusters = data['predicted_clusters']

        cluster2mention = get_first_mention(predicted_clusters, pronoun_sentences)
        token2cluster = get_cluster_for_token(predicted_clusters)

        orig2new = dict()
        pronoun2new = dict()
        current_shift_value = 0
        for list_ind, orig_token_ind in enumerate(original_token_ids):
            pronoun_token_ind = pronoun_token_ids[list_ind]
            if orig_token_ind == '[MASK]':

            else:


def get_first_mention(predicted_clusters, pronoun_sentences):
    cluster2mention = dict()
    for cluster in predicted_clusters:
        first_mention = cluster[0]
        token_span = []
        for token_id in first_mention:
            token_span.append(pronoun_sentences[token_id])
        cluster2mention[str(cluster)] = (first_mention, token_span, len(token_span))

    return cluster2mention

def get_cluster_for_token(predicted_clusters):
    token2cluster = dict()
    for cluster in predicted_clusters:
        for span in cluster:
            for token in span:
                token2cluster[token] = cluster

    return token2cluster

