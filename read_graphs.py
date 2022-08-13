import penman
from typing import List, Dict


def read_aligned_amr_file(recipe_amr_file: str) -> List[penman.Graph]:
    """
    Reads a file with the sentence level AMRs for one recipe
    Node to token alignments included
    :param recipe_amr_file: .txt file with the instruction AMRs
    :return: a list of all AMRs as penman Graph objects with
             complete metadata
    """
    graphs = []

    with open(recipe_amr_file, "r", encoding="utf-8") as amrs:
        current_amr_str = ""
        current_meta_data = dict()

        for line in amrs:
            if line == '\n':
                current_amr = penman.decode(current_amr_str)
                for meta_k, meta_v in current_meta_data.items():
                    current_amr.metadata[meta_k] = meta_v
                graphs.append(current_amr)
                current_amr_str = ""
                current_meta_data = dict()

            # add the meta data
            elif line[0] == '#':
                meta_type = line.strip().split(' ')[1]
                meta_values = line.strip().split()[2:]
                meta_key = meta_type[2:]
                current_meta_data[meta_key] = ' '.join(meta_values)
            else:
                current_amr_str += line.strip() + ' '

        if current_amr_str != "":
            current_amr = penman.decode(current_amr_str)
            for meta_k, meta_v in current_meta_data.items():
                current_amr.metadata[meta_k] = meta_v
            graphs.append(current_amr)

    return graphs


def read_action_graph(recipe_action_file: str) -> Dict[str, Dict]:
    """
    Read in a file with the action graph of a recipe
    :param recipe_action_file: the .conllu file for the recipe
    :return: dictionary with the nodes and edges of the action graph
            'nodes': a dictionary with an item for each node in the graph
                        key: the id of the first token of this node (i.e. with B-A label)
                            'label': all tokens belonging to this node
                            'ids': the token ids of all tokens
            'edges: a list of tuples, (source_node_id, target_node_id) for all edges
    """
    node_dict = dict()
    edge_list = []

    with open(recipe_action_file, "r", encoding="utf-8") as grf:
        complete_token = ""
        complete_ids = []
        prev_id = 0
        for line in grf:
            columns = line.strip().split()
            id = columns[0]
            token = columns[1]
            label = columns[4]
            edge = columns[6]
            edge_label = columns[7]

            if label == "O":
                if complete_token != "":
                    node_dict[prev_id] = {"label": complete_token, "ids": complete_ids}
                    complete_token = ""
                    complete_ids = []

            elif label[0] == "B":
                if complete_token != "":
                    node_dict[prev_id] = {"label": complete_token, "ids": complete_ids}

                complete_token = token
                complete_ids = [id]
                prev_id = id
                if edge != "0":
                    edge_list.append((id, edge))

            elif label[0] == "I":
                complete_token += " " + token
                complete_ids.append(id)

        if complete_token != "":
            node_dict[prev_id] = {"label": complete_token, "ids": complete_ids}

    return {'nodes': node_dict, 'edges': edge_list}


if __name__=="__main__":
    graph = read_action_graph('./test.conllu')

    for g in graph['nodes']:
        print(g)
