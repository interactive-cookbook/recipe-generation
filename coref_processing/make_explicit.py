import json
from typing import Dict, List

from coref_utils import get_coref_clusters_extended


def make_mentions_explicit(coref_file, new_coref_file):
    """
    Replaces each occurrence of 'it' or 'them' which was implicit in the original data with an explicit mention
    Removes nested clusters
    :param coref_file:
    :param new_coref_file:
    :return:
    """
    coref_data: Dict[str, Dict] = get_coref_clusters_extended(coref_file)
    new_coref_data = coref_data.copy()

    for recipe_name, data in coref_data.items():
        original_token_ids: List = data['original_token_id']
        pronoun_sentences: List[List[str]] = data['sentences']
        token_ind2sent = get_token_ind_sent(pronoun_sentences)
        pronoun_tokens: List[str] = data['text']
        pronoun_token_ids: [List[int]] = data['token_id']
        predicted_clusters: List[List[List[int]]] = data['predicted_clusters']

        token2cluster: Dict[int, List] = get_cluster_for_token(predicted_clusters)

        new_sentences = [[] for i in range(len(pronoun_sentences))]
        new_tokens = []
        new_orig_ids = []
        new_token_spans = dict()      # needed to shift the coreference clusters in the end
        sanity_check = []

        for list_ind, orig_token_ind in enumerate(original_token_ids):
            pronoun_token_ind = pronoun_token_ids[list_ind]             # the new token ID in the pronoun extended data
            target_sent_id = token_ind2sent[pronoun_token_ind]          # the ID of the sentence to which the token belongs
            if orig_token_ind == '[MASK]':
                try:
                    coref_clusters = token2cluster[pronoun_token_ind]
                    assert len(coref_clusters) == 1                     # a pronoun should never be part of 2 coref clusters!
                    coref_cluster = coref_clusters[0]

                    new_mention = get_previous_mention(current_cluster=coref_cluster, current_token_id=pronoun_token_ind,
                                                       pronoun_tokens=pronoun_tokens, original_token_ids=original_token_ids)

                    new_token_spans[pronoun_token_ind] = [len(new_tokens), len(new_tokens) + len(new_mention) - 1]
                    sanity_check.extend([i for i in range(len(new_tokens), len(new_tokens) + len(new_mention))])
                    new_sentences[target_sent_id].extend(new_mention)
                    new_tokens.extend(new_mention)
                    new_orig_ids.extend(['[MASK]' for i in range(len(new_mention))])

                # then not part of a coreference cluster and cannot be replaced
                except KeyError:
                    new_token_spans[pronoun_token_ind] = [len(new_tokens), len(new_tokens)]
                    sanity_check.append(len(new_tokens))
                    new_sentences[target_sent_id].append(pronoun_tokens[pronoun_token_ind])
                    new_tokens.append(pronoun_tokens[pronoun_token_ind])
                    new_orig_ids.append(orig_token_ind)

            # then it is an original token and should not be replaced but added as it is
            else:
                new_token_spans[pronoun_token_ind] = [len(new_tokens), len(new_tokens)]
                sanity_check.append(len(new_tokens))
                new_sentences[target_sent_id].append(pronoun_tokens[pronoun_token_ind])
                new_tokens.append(pronoun_tokens[pronoun_token_ind])
                new_orig_ids.append(orig_token_ind)

        new_token_ids = [tid for tid in range(len(new_tokens))]
        assert new_token_ids == sanity_check

        new_coref_data[recipe_name]['sentences'] = new_sentences
        new_coref_data[recipe_name]['text'] = new_tokens
        new_coref_data[recipe_name]['original_token_id'] = new_orig_ids
        new_coref_data[recipe_name]['token_id'] = new_token_ids

        new_coref_data[recipe_name]['predicted_clusters'] = shift_coref_clusters(predicted_clusters, new_token_spans)

    with open(new_coref_file, 'w', encoding='utf-8') as f:
        for recipe_name in new_coref_data.keys():
            json.dump(new_coref_data[recipe_name], f)
            f.write('\n')


def shift_coref_clusters(original_clusters, new_spans):

    shifted_clusters = []

    for cluster in original_clusters:
        shifted_cl = []
        for span in cluster:

            if span[0] == span[1]:           # original span was only one token long, so the new span can be used directly
                token = span[0]
                new_token_span = new_spans[token]
                shifted_span = new_token_span
            else:
                new_span_start = new_spans[span[0]][0]
                new_span_end = new_spans[span[1]][0]
                shifted_span = [new_span_start, new_span_end]

            shifted_cl.append(shifted_span)
        shifted_clusters.append(shifted_cl)

    return shifted_clusters


def get_token_ind_sent(sentences: List[List[str]]) -> Dict[int, int]:
    """

    :param sentences: a list of the tokenized sentences
    :return: dictionary, specifying for each document-level token index the index of the sentence to which it belongs
            e.g. if the 10th token belongs to the second sentence then token_id2sent[10] = 2
    """
    token_id2sent = dict()
    running_token_ind = 0
    for sent_id, sentence in enumerate(sentences):
        for token_ind, token in enumerate(sentence):
            token_id2sent[running_token_ind] = sent_id
            running_token_ind += 1

    return token_id2sent


def get_previous_mention(current_cluster: List[List[int]],
                         current_token_id: int,
                         pronoun_tokens: List[str],
                         original_token_ids: List[int]) -> List[str]:
    """
    Finds for an implicit PRO token the closest previous explicit mention from the same coreference cluster if there
    is one and returns the corresponding tokens
    If there is not such explicit mention then the PRO token itself is returned
    :param current_cluster:
    :param current_token_id:
    :param pronoun_tokens:
    :param original_token_ids:
    :return:
    """

    current_cluster.sort()              # should already be sorted but just to make sure

    current_mention = 0
    for mention_ind, mention in enumerate(current_cluster):
        if current_token_id in mention:                     # function gets only called for PRO tokens -> should always be one token
            current_mention = mention_ind
            break

    new_mention_ids = [current_token_id, current_token_id]  # the original span for the current token
    # go through all preceding spans in the cluster until finding an explicit mention for the current token
    # or if none is found make no changes
    prev_ind = current_mention - 1
    while prev_ind > -1:
        prev_mention = current_cluster[prev_ind]
        if prev_mention[0] != prev_mention[1]:                   # mentions with more than 1 token are explicit
            new_mention_ids = prev_mention
            break
        elif original_token_ids[prev_mention[0]] != '[MASK]':   # then it is a one token explicit mention
            new_mention_ids = prev_mention
            break
        prev_ind -= 1

    new_mention_tokens = [pronoun_tokens[t_id] for t_id in range(new_mention_ids[0], new_mention_ids[1] + 1)]

    return new_mention_tokens


def get_first_mention(current_cluster, pronoun_tokens):

    new_mention_ids = current_cluster[0]

    new_mention_tokens = [pronoun_tokens[t_id] for t_id in range(new_mention_ids[0], new_mention_ids[1] + 1)]
    return new_mention_tokens


def get_cluster_for_token(predicted_clusters: List[List[List[int]]]) -> Dict[int, List]:
    """

    :param predicted_clusters: the list of predicted clusters
    :return: a dictionary with the IDs of tokens occurring in coref clusters as keys and the clusters in which
             the token occurs as value
    """
    token2cluster = dict()
    for cluster in predicted_clusters:
        for span in cluster:
            for token in range(span[0], span[1] + 1):
                try:
                    token2cluster[token].append(cluster)
                except KeyError:
                    token2cluster[token] = [cluster]

    return token2cluster



if __name__=='__main__':

    make_mentions_explicit('./ara_pronoun_merged_pred.jsonlines', './ara_explicit_merged_pred.jsonlines')

