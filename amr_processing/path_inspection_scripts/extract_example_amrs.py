from ..prelim_inspection_amr_action_alignments import get_graph_pairs, get_one2many_amrs_and_alignments
import random
import penman
from ..penman_networkx_conversions import networkx2penman


def sample_multi2one_amrs(action_graph_dir, amr_graph_dir, output_file, n=None):
    """

    :param action_graph_dir:
    :param amr_graph_dir:
    :param output_file:
    :param n:
    :return:
    """
    amr_ac_graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)

    all_one2many_amrs = []

    for recipe_name in amr_ac_graph_pairs.keys():
        one2many_current_recipe = get_one2many_amrs_and_alignments(amr_ac_graph_pairs[recipe_name]['action'],
                                                                   amr_ac_graph_pairs[recipe_name]['amrs'])
        all_one2many_amrs.extend(one2many_current_recipe)

    if not n:
        sampled_amrs = all_one2many_amrs
    else:
        sampled_amrs = random.sample(all_one2many_amrs, n)

    with open(output_file, 'w', encoding='utf-8') as out:
        for (amr, alignments) in sampled_amrs:

            pen_amr = networkx2penman(amr)
            aligned_amr_nodes = [al[0] for al in alignments]
            pen_amr.metadata['alignments'] = ', '.join(aligned_amr_nodes)

            pen_str = penman.encode(pen_amr)

            out.write(f'{pen_str}\n\n')


if __name__=="__main__":

    sample_multi2one_amrs('../../Corpora/Ara_Punctuation/new_ara_data_new_action_graphs',
                          './recipe_amrs_sentences',
                          './many2one_amrs.txt')
