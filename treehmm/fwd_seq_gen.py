#!/usr/bin/env python3

# This file calculates the order in which nodes in the tree should be
# traversed during the forward
# pass(roots to leaves)

# Tree is a complex graphical model where we can have multiple parents and multiple children for a node. Hence the
# order in which the tree should be traversed becomes significant. Forward algorithm is a dynamic programming
# problem where to calculate the values at a node,we need the values of the parent nodes beforehand, which need to be
# traversed before this node.

# Importing the libraries

import numpy as np

# Defining the forward_sequence_generator function

def forward_sequence_generator(adj_mat):
    """
      Args:
          adj_mat: "adj_mat" is the value of 'adjacent_symmetry_matrix' key from the dictionary 'hmm'.
      Returns:
          forward_tree_sequence: A list of size "D", where "D" is the number of
          nodes in the tree
      """

    # "adj_mat" is the value of 'adjacent_symmetry_matrix' key from the dictionary 'hmm'.
    #adj_mat = hmm["adjacent_symmetry_matrix"]
    pair_of_nonzero_indices = np.transpose(np.nonzero(adj_mat))

    # Use this for pair_of_nonzero_indices when "adj_mat" is sparse matrix and comment the above line.

    # temp = adj_mat.tocoo()
    # rows = temp.row
    # cols = temp.col
    # pair_of_nonzero_indices = np.array([[r,c] for r,c in zip(rows,cols)])

    adj_mat_row_sums = np.squeeze(np.asarray(np.sum(adj_mat, axis=1)))
    adj_mat_col_sums = np.squeeze(np.asarray(np.sum(adj_mat, axis=0)))
    indices_of_root_nodes = np.where(np.logical_and(adj_mat_row_sums != 0,adj_mat_col_sums == 0))[0]  # np.array()

    # order is a list of arrays with the array consisting of indices of the root nodes in the adjacent_symmetry_matrix

    order = list()
    order.append(indices_of_root_nodes)  # [array]

    # this for loop appends the unique next level in order
    for o in order:
        previous_level = o
        next_level = np.array([pair_of_nonzero_indices[list(np.where(pair_of_nonzero_indices[:, 0] == i)[0]), 1] for i in previous_level], dtype=object)
        next_level = np.concatenate(next_level)
        if len(next_level) == 0:
            break
        order.append(np.unique(next_level))
    order.append(np.array([]))
    length_of_order = len(order)

    for i in range(1, length_of_order - 1):
        shift = []
        for j in order[i]:
            indices_of_nonzero_adj_mat_row = np.nonzero(adj_mat[:, j])[0]
            boolean_check = set(indices_of_nonzero_adj_mat_row).issubset(
                np.hstack(order[:i]))
            if not boolean_check:
                shift.append(j)
        element_to_update = [i for i in order[i] if i not in shift]
        order[i] = np.unique(element_to_update)
        order[i + 1] = np.unique(list(order[i + 1]) + shift)

    # 'forward_order' is an empty list which will hold the forward tree sequence
    forward_order = []

    # this for loop appends the final order list to forward order
    for i in order:
        forward_order = forward_order + list(i)

    return forward_order


def run_an_example():
    """
    sample run for forward_sequence_generator function
    """
    from scipy.sparse import csr_matrix
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)

    forward_tree_sequence = forward_sequence_generator(csr_matrix(sample_tree))
    print(forward_tree_sequence)

if __name__ == "__main__":
    run_an_example()
