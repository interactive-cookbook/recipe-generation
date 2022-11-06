import networkx as nx
from graph_processing.action_amr_graph_mappings import add_semantic_representations

"""
Functions I need:
- read the action graph for a dish
- check for each node from which recipe it stems and add corresponding amr
- deal with action nodes that belong together / same amr
- function to traverse the graph
- function to generate a sentence

"""


def generate(amr_graph: nx.DiGraph, context, model):
    pass


def generate_wo_context(amr_graph: nx.DiGraph, model):
    pass


# TODO: replace model dir and context len with the configuration file
def generate_recipe_ac_graph(action_graph: nx.DiGraph, model_dir, context_len: int = 1):

    generated_sentences = []
    already_realized_amrs = []

    # load the model and instantiate inference class object
    model = None

    # add the corresponding amr graphs
    sem_action_graph = add_semantic_representations(action_graph)

    # create an order of the action graphs
    action_ordering = order_actions_df(sem_action_graph)

    # loop through the ordered action nodes
    for action_node in action_ordering:
        sem_repr = nx.get_node_attributes(sem_action_graph, 'amr')[action_node]
        amr_id = sem_repr.graph['id']
        # skip already covered amr graphs
        if amr_id in already_realized_amrs:
            continue

        if context_len == 0:
            gen_snt = generate_wo_context(sem_repr, model)
        else:
            context_sentences = generated_sentences[-context_len:]
            gen_snt = generate(sem_repr, context_sentences, model)

        generated_sentences.append(gen_snt)
        already_realized_amrs.append(amr_id)

    return generated_sentences


def order_actions_topological(action_graph: nx.Graph):
    """

    :param action_graph:
    :return:
    """
    ordered_actions = list(nx.topological_sort(action_graph))
    return ordered_actions


def order_actions_df(action_graph: nx.DiGraph):
    """

    :param action_graph:
    :return:
    """
    all_nodes = set(action_graph.nodes)
    covered_nodes = set()
    potential_starts = []
    ordered_actions = []

    # find potential start nodes
    for n in action_graph.nodes():
        parent_nodes = list(action_graph.predecessors(n))
        if not parent_nodes:
            potential_starts.append(n)
    potential_starts.sort()

    current_node = potential_starts.pop(0)
    ordered_actions.append(current_node)
    covered_nodes.add(current_node)
    while len(covered_nodes) < len(all_nodes):
        child_node = list(action_graph.successors(current_node))
        assert len(child_node) == 1

        child_parents = list(action_graph.predecessors(current_node))
        all_parents_covered = True
        for cp in child_parents:
            if cp not in covered_nodes:
                all_parents_covered = False
                next_start_node = None
                for left_start_node in potential_starts:
                    if list(nx.all_simple_paths(action_graph, left_start_node, child_node)):
                        next_start_node = left_start_node
                        break
                assert next_start_node
                current_node = next_start_node
                ordered_actions.append(next_start_node)
                covered_nodes.add(next_start_node)
                break

        if all_parents_covered:
            ordered_actions.append(child_node)
            covered_nodes.add(child_node)
            current_node = child_node

    return ordered_actions


