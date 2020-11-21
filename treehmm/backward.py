# This file infers the backward probabilities for all the nodes of the treeHMM

# The backward probability for state X and observation at node k is defined as the probability of observing the
# sequence of observations e_k+1, ... ,e_n under the condition that the state at node k is X.

# That is backward_probabilities[X,k] := Prob(E_k+1 = e_k+1, ... , E_n = e_n | X_k = X)
# E_1...E_n = e_1...e_n is the sequence of observed emissions and X_k is a random variable that represents
# the state at node k

# Importing the libraries

import numpy as np
import pandas as pd
import math
import itertools

# Defining the backward function

def backward(hmm, adjacent_matrix, emission_observation, backward_tree_sequence, observed_states_training_nodes=None, verbose = False):
    """
    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        emission_observation: emission_observation is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number
        backward_tree_sequence: It is a list denoting the order of nodes in
            which the tree should be traversed in backward direction(from leaves
            to roots).It's the output of backward_sequence_generator function.
        observed_states_training_nodes: It is a (L * 2) dataframe where L is the number of training
            nodes where state values are known. First column should be the node
            number and the second column being the corresponding known state
            values of the nodes

    Returns:
        backward_probs: A dataframe of size (N * D) denoting the backward
        probabilities at each node of the tree, where "N" is possible no. of
        states and "D" is the total number of nodes in the tree
    """

    if observed_states_training_nodes is None:
        observed_states_training_nodes = pd.DataFrame(columns=["node", "state"])

    #adjacent_matrix = hmm["adjacent_matrix"]
    hmm["state_transition_probabilities"].fillna(0, inplace=True)
    number_of_levels = len(emission_observation)

    for m in range(number_of_levels):
        hmm["emission_probabilities"][m].fillna(0, inplace=True)

    number_of_observations = len(emission_observation[0])

    number_of_states = len(hmm["states"])

    # We define a variable 'backward_probabilities' which is a numpy array which will be denoting the backward probabilities

    backward_probabilities = np.zeros(
        shape=(number_of_states, number_of_observations))

    # We transform the numpy array 'backward_probabilities' into a pandas dataframe
    backward_probabilities = pd.DataFrame(
        data=backward_probabilities,
        index=hmm["states"],
        columns=range(number_of_observations))
    if verbose:
        print("Backward loop running")

    # main for loop to calulate the backward_probabilities
    for k in backward_tree_sequence:
        boolean_value = set([k]).issubset(list(observed_states_training_nodes["node"])) if  observed_states_training_nodes is not None else False
        desired_state = list(observed_states_training_nodes["state"][observed_states_training_nodes["node"] == k])[0] if boolean_value else None

        next_state = np.nonzero(adjacent_matrix[k, :] != 0)[1]
        length_of_next_state = len(next_state)

        if length_of_next_state == 0:
            for state in hmm["states"]:
                backward_probabilities.loc[state, k] = 0
            if boolean_value:
                desired_state_index = np.where(
                    desired_state != np.array(hmm["states"]))[0]

                # 'mapdf' is a dataframe which maps from old state to new state
                mapdf = np.array([[i, j] for i, j in zip(
                    range(number_of_states), hmm["states"])])
                mapdf = pd.DataFrame(
                    data=mapdf, columns=[
                        "old_states", "new_states"])
                mapdf["old_states"] = pd.to_numeric(mapdf["old_states"])
                tozero = list(
                    mapdf["new_states"][mapdf["old_states"].isin(desired_state_index)])[0]
                backward_probabilities.loc[tozero, k] = -math.inf
            continue

        next_array = np.array(list(itertools.product(hmm["states"],repeat=length_of_next_state)))
        # 'inter' is a list to find which next_state is present in the observed_states_training_nodes
        inter = list(set(next_state) & set(observed_states_training_nodes.iloc[:, 0]))
        len_inter = len(inter)
        # 'true_boolean_array' is a boolean array with only True boolean elements
        true_boolean_array = np.repeat(True, next_array.shape[0], axis=0)
        if len_inter != 0:
            for i in range(len_inter):
                index_variable_1 = np.where(
                    observed_states_training_nodes.iloc[:, 0] == inter[i])[0][0]
                index_variable_2 = np.where(inter[i] == next_state)[0][0]
                desired_state = observed_states_training_nodes.iloc[index_variable_1, 1]
                true_boolean_array = np.logical_and(len(np.where(
                    next_array[:, index_variable_2] == desired_state)[0]), true_boolean_array)

        # 'index_array' is a numpy array comprising the index of all the True values of true_boolean_array
        index_array = np.where(true_boolean_array)[0]

        for state in hmm["states"]:
            logsum = []
            for i in index_array:
                temp = 0
                for j in range(next_array.shape[1]):
                    emit = 0
                    for m in range(number_of_levels):
                        if emission_observation[m][k] is not None:
                            try:
                                emit = math.log(hmm["emission_probabilities"][m].loc[next_array[i, j], emission_observation[m][next_state[j]]]) + emit
                            except ValueError:
                                emit = -math.inf
                                break
                    try:
                        temp = temp + (backward_probabilities.loc[next_array[i, j], next_state[j]] + math.log(hmm["state_transition_probabilities"].loc[state, next_array[i, j]]) + emit)
                    except ValueError:
                        temp = -math.inf
                        break

                if -math.inf < temp < 0:
                    logsum.append(temp)

            backward_probabilities.loc[state, k] = np.log(np.sum(np.exp(logsum)))

        if boolean_value:
            old_states = range(number_of_states)
            new_states = hmm["states"]
            desired_state_index = np.where(
                desired_state != np.array(
                    hmm["states"]))[0]
            mapdf = np.array([[i, j] for i, j in zip(old_states, new_states)])
            mapdf = pd.DataFrame(
                data=mapdf, columns=[
                    "old_states", "new_states"])
            mapdf["old_states"] = pd.to_numeric(mapdf["old_states"])
            tozero = mapdf["new_states"][mapdf["old_states"].isin(
                desired_state_index)].tolist()[0]
            backward_probabilities.loc[tozero, k] = -math.inf

    return backward_probabilities

def run_an_example():
    """sample run for backward function"""
    from treehmm.initHMM import initHMM
    from treehmm.bwd_seq_gen import backward_sequence_generator
    from scipy.sparse import csr_matrix

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    sparse_sample_tree = csr_matrix(sample_tree)

    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    emissions = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM(states, emissions)
    emission_observation = [["L", "L", "R", "R", "L"]]
    backward_tree_sequence = backward_sequence_generator(sparse_sample_tree)
    data = {'node': [1], 'state': ['P']}
    observed_states_training_nodes = pd.DataFrame(data=data, columns=["node", "state"])
    backward_probs = backward(hmm, sparse_sample_tree, emission_observation, backward_tree_sequence, observed_states_training_nodes, True)
    print(backward_probs)

# sample call to the function
if __name__ == "__main__":
    run_an_example()
