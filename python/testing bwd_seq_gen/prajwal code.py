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
    pair_of_nonzero_indices = np.transpose(np.nonzero(adj_mat))
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
        next_level = np.array([pair_of_nonzero_indices[list(
            np.where(pair_of_nonzero_indices[:, 0] == i)[0]), 1] for i in previous_level])
        
        next_level = np.unique(next_level)
        if len(next_level) == 0:
            break
        order.append(next_level)

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
