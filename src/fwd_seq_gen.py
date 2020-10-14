#!/usr/bin/env python3

# This file calculates the order in which nodes in the tree should be
# traversed during the forward
# pass(indices_of_zero_rowsums_and_nonzero_column_sums to leaves)

# Tree is a complex graphical model where we can have multiple parents and multiple children for a node. Hence the
# order in which the tree should be traversed becomes significant. Forward algorithm is a dynamic programming
# problem where to calculate the values at a node,we need the values of the parent nodes beforehand, which need to be
# traversed before this node.

# Importing the libraries

import numpy as np
import pandas as pd
import initHMM

# Defining the fwd_seq_gen function

def fwd_seq_gen(hmm, number_of_levels=100):
    """
      Args:
          hmm: It is a dictionary given as output by initHMM.py file
          number_of_levels: No. of levels in the tree, if known. Default is 100

      Returns:
          forward_tree_sequence: A list of size "D", where "D" is the number of
          nodes in the tree
      """

    adj_mat = hmm["adjacent_symmetry_matrix"]
    # pair_of_nonzero_indices = np.transpose(np.nonzero(adj_mat))

    # Use this when "adj_mat" is sparse matrix and comment the above line.

    temp = adj_mat.tocoo()
    rows = temp.row
    cols = temp.col
    pair_of_nonzero_indices = np.array([[r,c] for r,c in zip(rows,cols)])

    adj_mat_row_sums = np.squeeze(np.asarray(np.sum(adj_mat, axis=1)))
    adj_mat_col_sums = np.squeeze(np.asarray(np.sum(adj_mat, axis=0)))

    indices_of_zero_rowsums_and_nonzero_column_sums = np.where(
        np.logical_and(
            adj_mat_row_sums != 0,
            adj_mat_col_sums == 0))[0] # np.array()
    order = list()
    order.append(indices_of_zero_rowsums_and_nonzero_column_sums)  # [array]

    for o in order:
        previous_level = o
        next_level = np.array([pair_of_nonzero_indices[list(np.where(pair_of_nonzero_indices[:, 0] == i)[0]), 1] for i in previous_level])
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
    forward_order = []
    for i in order:
        forward_order = forward_order + list(i)

    return forward_order


def run_an_example():
    """
    sample run for fwd_seq_gen function
    """
    import initHMM
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)
    states = ['P', 'N']
    symbols = [['L', 'R']]
    hmm = initHMM.initHMM(states, symbols, sample_tree)

    forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)

if __name__ == "__main__":
    run_an_example()
