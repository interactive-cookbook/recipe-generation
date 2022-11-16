import random

import networkx as nx


def order_actions_topological(action_graph: nx.Graph):
    """
    Orders all nodes of the input graph topologically
    :param action_graph:
    :return: List of ordered nodes of action_graph
    """
    try:
        ordered_actions = list(nx.topological_sort(action_graph))
    except nx.NetworkXError:
        print(f'Warning: found action graph with disconnected components. Correct ordering of the disconnected'
              f'components cannot be ensured!')
        ordered_actions = list(action_graph.nodes)
        ordered_actions.sort()
    return ordered_actions


def order_actions_token_ids(action_graph: nx.DiGraph):
    """
    Order the nodes of an action graph by following the original order of actions
    as far as possible,
    if the actions a1, a2, a3, a4 occur in this order in the original recipe
    then
    :param action_graph:
    :return:
    """
    all_nodes = list(action_graph.nodes)
    remaining_nodes = all_nodes.copy()
    covered_nodes = []
    all_nodes = [int(n) for n in all_nodes if n != 'end']
    all_nodes.sort()
    all_nodes = [str(n) for n in all_nodes]
    all_nodes.append('end')

    while len(covered_nodes) < len(all_nodes):
        for ac_ind, ac_node in enumerate(remaining_nodes):
            parent_nodes = list(action_graph.predecessors(ac_node))
            if not parent_nodes:
                covered_nodes.append(ac_node)
                remaining_nodes.pop(ac_ind)
                break
            else:
                all_parents_covered = True
                for p_node in parent_nodes:
                    if p_node not in covered_nodes:
                        all_parents_covered = False
                if all_parents_covered:
                    covered_nodes.append(ac_node)
                    remaining_nodes.pop(ac_ind)
                    break

    return covered_nodes


def find_longest_path_start(candidate_nodes: list, end_node, graph: nx.DiGraph) -> list:
    """
    Computes the path from each node in candidate_nodes to the end_node and returns the nodes from candidate_nodes
    that have the longest path to the end_node
    :param candidate_nodes:
    :param end_node:
    :param graph:
    :return:
    """
    current_max_len = 0
    current_starts = []
    for cn in candidate_nodes:
        paths = list(nx.all_simple_paths(graph, cn, end_node))
        for p in paths:
            if len(p) > current_max_len:
                current_max_len = len(p)
                current_starts = [cn]
            elif len(p) == current_max_len:
                current_starts.append(cn)

    current_starts.sort()

    return current_starts


def order_actions_df_lf(action_graph: nx.DiGraph, tie_strategy=None):
    """
    Orders all nodes from the input graph in a the following way
    1. start with the node without parents that has the longest path to the 'end' node
    2. continue with the child node if a) child has no other parent nodes or b) all parent nodes are already covered
    3. If child node has other not yet covered parents, choose the node without parents which has the longest path to an
       uncovered parent node;
    4. Continue 1. - 3. until all nodes are covered
    :param action_graph:
    :param tie_strategy: strategy to use if two nodes have same longest path
                        if tie_strategy is None then choose any node
                        if tie_strategy = 'id' then choose the node with the smallest token ID
    :return: ordered list of all nodes of action_graph
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

    # find longest path
    longest_starts = find_longest_path_start(potential_starts, 'end', action_graph)
    if tie_strategy == 'id':
        current_node = longest_starts[0]
    elif not tie_strategy:
        current_node = random.choice(longest_starts)
    else:
        raise NotImplementedError

    potential_starts.remove(current_node)
    ordered_actions.append(current_node)
    covered_nodes.add(current_node)
    while len(covered_nodes) < len(all_nodes):
        # parser only produces trees, so each node has exactly one successor
        # except the 'end' node which has no successor but should be added last so loop is not entered for it
        child_nodes = list(action_graph.successors(current_node))

        if not len(child_nodes) == 1:       # then the graph has disconnected components
            print(f'Warning: found action graph with disconnected components. Correct ordering of the disconnected'
                  f'components cannot be ensured!')
            current_node = potential_starts.pop(0)
            ordered_actions.append(current_node)
            covered_nodes.add(current_node)
            continue

        child_node = child_nodes[0]

        child_parents = list(action_graph.predecessors(child_node))
        all_parents_covered = True
        for cp in child_parents:
            if cp not in covered_nodes:  # found a not yet covered parent
                all_parents_covered = False
                potential_next_starts = find_longest_path_start(potential_starts, child_node, action_graph)
                if tie_strategy == 'id':
                    next_start_node = potential_next_starts[0]
                elif not tie_strategy:
                    next_start_node = random.choice(potential_next_starts)
                else:
                    raise NotImplementedError
                assert next_start_node
                potential_starts.remove(next_start_node)
                current_node = next_start_node
                ordered_actions.append(next_start_node)
                covered_nodes.add(next_start_node)
                break

        if all_parents_covered:
            ordered_actions.append(child_node)
            covered_nodes.add(child_node)
            current_node = child_node

    return ordered_actions


def order_actions_df_lf_id(action_graph: nx.DiGraph):
    """
    Orders all nodes from the input graph in a the following way
    1. start with the node without parents that has the longest path to the 'end' node
    2. continue with the child node if a) child has no other parent nodes or b) all parent nodes are already covered
    3. If child node has other not yet covered parents, choose the node without parents which has the longest path to an
       uncovered parent node; if two nodes have same longest path then choose the one with the smaller token ID
    4. Continue 1. - 3. until all nodes are covered
    :param action_graph:
    :return: ordered list of all nodes of action_graph
    """

    return order_actions_df_lf(action_graph=action_graph, tie_strategy='id')


def order_actions_df(action_graph: nx.DiGraph):
    """
    Orders all nodes from the input graph in a the following way
    1. start with a node without parents
    2. continue with the child node if a) child has no other parent nodes or b) all parent nodes are already covered
    3. If child node has other not yet covered parents, choose a node without parents which has a path to an uncovered parent node
    4. Continue 1. - 3. until all nodes are covered
    :param action_graph:
    :return: ordered list of all nodes of action_graph
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
        # parser only produces trees, so each node has exactly one successor
        # except the 'end' node which has no successor but should be added last so loop is not entered for it
        child_nodes = list(action_graph.successors(current_node))

        if not len(child_nodes) == 1:  # then the graph has disconnected components
            print(f'Warning: found action graph with disconnected components. Correct ordering of the disconnected'
                  f'components cannot be ensured!')
            current_node = potential_starts.pop(0)
            ordered_actions.append(current_node)
            covered_nodes.add(current_node)
            continue

        child_node = child_nodes[0]

        child_parents = list(action_graph.predecessors(child_node))
        all_parents_covered = True
        for cp in child_parents:
            if cp not in covered_nodes:  # found a not yet covered parent
                all_parents_covered = False
                next_start_node = None
                for node_ind, left_start_node in enumerate(potential_starts):
                    if list(nx.all_simple_paths(action_graph, left_start_node, child_node)):
                        next_start_node = left_start_node
                        potential_starts.pop(node_ind)
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
