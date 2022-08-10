from read_graphs import read_aligned_amr_file, read_action_graph
import os
import penman
from penman import surface
from collections import defaultdict


def get_graph_pairs(action_graph_dir, amr_graph_dir):

    graph_pairs = dict()

    for dish in os.listdir(action_graph_dir):
        for recipe in os.listdir('/'.join([action_graph_dir, dish, 'recipes'])):

            ac_graph = read_action_graph('/'.join([action_graph_dir, dish, 'recipes', recipe]))

            recipe_name = recipe.split('.')[:-1]
            recipe_name = '.'.join(recipe_name)
            corresponding_amr = recipe_name + '_sentences_amr.txt'
            amr_graph = read_aligned_amr_file('/'.join([amr_graph_dir, dish, 'amrs', corresponding_amr]))

            graph_pairs[recipe_name] = dict()
            graph_pairs[recipe_name]['action'] = ac_graph
            graph_pairs[recipe_name]['amr'] = amr_graph

    return graph_pairs


def inspect_amr_action_alignments(action_graph_dir, amr_graph_dir):

    graph_pairs = get_graph_pairs(action_graph_dir, amr_graph_dir)
    all_one2many_amrs = []
    graphs_with_missing_alignments = dict()
    count_missed_action_nodes = 0
    count_action_nodes = 0
    count_instructions = 0          # should correspond to the number of AMRs
    count_alignments = defaultdict(int)

    for recipe_name in graph_pairs.keys():

        action_graph = graph_pairs[recipe_name]['action']
        amr_graphs = graph_pairs[recipe_name]['amr']
        count_instructions += len(amr_graphs)
        count_action_nodes += len(list(action_graph['nodes'].keys()))

        # find all AMRs that are aligned to more than one action node
        one2many_amrs_recipe, alignment_counts = get_one2many_amrs(action_graph, amr_graphs)
        all_one2many_amrs.extend(one2many_amrs_recipe)

        # update how often an AMR is aligned to X action nodes
        for key, value in alignment_counts.items():
            count_alignments[key] += value

        # find all action nodes that do not have an aligned AMR node
        non_aligned_actions = get_unaligned_action_nodes(action_graph, amr_graphs)
        if non_aligned_actions:
            graphs_with_missing_alignments[recipe_name] = non_aligned_actions
            count_missed_action_nodes += len(non_aligned_actions)

    print(f'Number of AMRs aligned to more than one action node: {len(all_one2many_amrs)}\n')
    print(f'Number of action nodes without aligned AMR: {count_missed_action_nodes}\n')
    print(f'Number of total action nodes: {count_action_nodes}')
    print(f'Number of instructions: {count_instructions}')
    print(count_alignments)

    with open('./action_nodes_without_alignments.txt', "w", encoding="utf-8") as file:
        for re_name, nodes in graphs_with_missing_alignments.items():
            for n in nodes:
                file.write(f'{re_name}\t{n}\n')


def get_one2many_amrs(action_graph, amr_graphs):

    one_to_many_amrs = []
    action_nodes = list(action_graph['nodes'].keys())
    count_alignments = defaultdict(int)

    for amr_graph in amr_graphs:
        count = 0

        token_aligned_nodes = surface.alignments(amr_graph)     # alignments for all aligned nodes
        for instance, alignment in token_aligned_nodes.items():
            token_id = alignment.indices[0]                     # get the token id of the aligned token

            # IMPORTANT! token ids in the recipe graphs are strings but surface.alignments give ints
            if str(token_id) in action_nodes:                   # check if aligned token is an action node
                count += 1

        if count > 1:
            one_to_many_amrs.append(amr_graph)
        count_alignments[count] += 1

    return one_to_many_amrs, count_alignments


def get_unaligned_action_nodes(action_graph, amr_graphs):

    amr_node_aligned_tokens = []

    for amr_graph in amr_graphs:
        token_aligned_nodes = surface.alignments(amr_graph)
        for instance, alignment in token_aligned_nodes.items():
            token_id = alignment.indices[0]
            amr_node_aligned_tokens.append(str(token_id))

    non_aligned_action_nodes = []

    for action_node in action_graph['nodes'].keys():
        if action_node not in amr_node_aligned_tokens:
            multi_token_ids = action_graph['nodes'][action_node]['ids']
            found_alignment = False
            for mt_id in multi_token_ids:
                if mt_id in amr_node_aligned_tokens:
                    found_alignment = True

            if not found_alignment:
                non_aligned_action_nodes.append(action_node)

    return non_aligned_action_nodes


if __name__=="__main__":

    inspect_amr_action_alignments('../Corpora/Mapped_Ara/new_ara_data_new_action_graphs',
                                  './aligned_recipe_amrs')
