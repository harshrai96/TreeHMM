#!/usr/bin/env python3

# This file calculates the order in which nodes in the tree should be
# traversed during the backward pass(leaves to roots)

# Tree is a complex graphical model where we can have multiple parents and multiple children for a node. Hence the
# order in which the tree should be traversed becomes significant. Backward algorithm is a dynamic programming
# problem where to calculate the values at a node,we need the values of the children nodes beforehand, which need to
# be traversed before this node. This algorithm outputs a possible(not unique) order of the traversal of nodes
# ensuring that the children are traversed first before the parents

# Importing the libraries

import numpy as np

# Defining the backward_sequence_generator function

def backward_sequence_generator(adj_mat):
    """
    Args:
        adj_mat: is the value of 'adjacent_symmetry_matrix' key from the dictionary 'hmm'.
    Returns:
        backward_tree_sequence: A list of size "D", where "D" is the number of
        nodes in the tree
    """
    
    pair_of_nonzero_indices = np.transpose(np.nonzero(adj_mat))

    # Use this for pair_of_nonzero_indices when "adj_mat" is a sparse matrix and comment the above line.

    # temp = adj_mat.tocoo()
    # rows = temp.row
    # cols = temp.col
    # pair_of_nonzero_indices = np.array([[r, c] for r, c in zip(rows, cols)])

    adj_mat_row_sums = np.squeeze(np.asarray(np.sum(adj_mat, axis=1)))
    adj_mat_col_sums = np.squeeze(np.asarray(np.sum(adj_mat, axis=0)))
    indices_of_root_nodes = np.where(np.logical_and(adj_mat_row_sums == 0, adj_mat_col_sums !=0))[0]  # np.array()

    # order is a list of arrays with the array consisting of indices of the root nodes in the adjacent_symmetry_matrix
    order = list()
    order.append(indices_of_root_nodes)  # [array]

    # this for loop appends the unique next level in order
    for o in order:
        previous_level = o
        next_level = np.array([pair_of_nonzero_indices[list(np.where(pair_of_nonzero_indices[:, 1] == i)[0]), 0] for i in previous_level], dtype=object)
        next_level = np.concatenate(next_level)
        if len(next_level) == 0:
            break
        order.append(np.unique(next_level))

    order.append(np.array([]))
    length_of_order = len(order)

    #
    for i in range(1, length_of_order - 1):
        shift = []
        for j in order[i]:
            indices_of_nonzero_adj_mat_column = np.nonzero(adj_mat[j,:])[1]
            boolean_check = set(indices_of_nonzero_adj_mat_column).issubset(np.hstack(order[:i]))
            if not boolean_check:
                shift.append(j)
        element_to_update = [i for i in order[i] if i not in shift]
        order[i] = np.unique(element_to_update)
        order[i + 1] = np.unique(list(order[i + 1]) + shift)

    # 'backward_order' is an empty list which will hold the backward tree sequence
    backward_order = []

    # this for loop appends the final order list to backward order
    for i in order:
        backward_order = backward_order + list(i)

    return backward_order


def run_an_example():
    """
    sample run for backward_sequence_generator function
    """
    from scipy.sparse import csr_matrix
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)

    backward_tree_sequence = backward_sequence_generator(csr_matrix(sample_tree))
    print(backward_tree_sequence)

if __name__ == "__main__":
    run_an_example()
