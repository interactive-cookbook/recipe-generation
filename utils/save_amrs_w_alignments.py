from utils.paths import ARA_DIR, SENT_AMR_DIR, get_new_dish_dir
from graph_processing.action_amr_graph_mappings import get_graph_pairs
from pathlib import Path
from amr_processing.penman_networkx_conversions import networkx2penman
import penman


def amrs_with_alignments(output_dir):
    """
    Reads the S-AMRs and corresponding action graphs, extracts the action alignment information from the pairs
    and saves the S-AMRs with the information which nodes are action-aligned as part of the metadata (similar as
    is done for the A-AMRs created by the splitting algorithm)
    :param output_dir: name of directory where the modified amrs get saved
    """
    new_dir = Path(f'data/{output_dir}')
    new_dir.mkdir(exist_ok=True, parents=True)

    # read amrs and action graphs
    ara_corpus = ARA_DIR
    amr_corpus = SENT_AMR_DIR
    graph_pairs = get_graph_pairs(ara_corpus, amr_corpus)

    for recipe in graph_pairs.keys():
        dish_name = recipe.split('_')[:-1]
        dish_name = '_'.join(dish_name)
        dish_path = new_dir / dish_name
        dish_path.mkdir(exist_ok=True, parents=True)

        amrs = graph_pairs[recipe]['amrs']
        with open(f'{dish_path}/{recipe}_sentences_amr.txt', 'w', encoding='utf-8') as new_file:
            for amr in amrs:
                penman_amr = networkx2penman(amr)
                amr_string = penman.encode(penman_amr)
                new_file.write(f'{amr_string}\n\n')


if __name__=='__main__':
    amrs_with_alignments('recipe_amrs_sentences_alignments_ara1')
