# Importing the libraries

import numpy as np
import pandas as pd
import itertools
import math
from scipy.special import logsumexp


# This function calculates the probability of transition from multiple
# nodes to given node in the tree

# Defining the noisy_or function

def noisy_or(hmm, previous_state, current_state):
    """
    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        previous_state: It is a numpy array containing state variable values for
            the previous nodes
        current_state: It is a string denoting the state variable value for
            current node

    Returns:
        transition_prob: The Noisy_OR probability for the transition
    """

    l = len(
        np.where(
            np.array(previous_state) == np.array(
                hmm["states"][0]))[0])
    fin = math.pow(
        hmm["state_transition_probabilities"].loc[hmm["states"][0], hmm["states"][1]], l)

    if current_state == hmm["states"][1]:
        return fin
    else:
        return 1 - fin


# This function infers the forward probabilities for all the nodes of the treeHMM

# The forward probability for state X up to observation at node k is defined as the probability of observing the
# sequence of observations e_1,..,e_k given that the state at node k is X. That is forward_probabilities[X,
# k] := Prob( X_k = X | E_1 = e_1,.., E_k = e_k) where E_1...E_n = e_1...e_n is the sequence of observed
# emissions and  X_k is a random variable that represents the state
# at node k

# Defining the forward function

def forward(hmm, adjacent_matrix, emission_observation, forward_tree_sequence, observed_states_training_nodes=None, verbose = False):
    """
    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        emission_observation: emission_observation is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number
        forward_tree_sequence: It is a list denoting the order of nodes in which
            the tree should be traversed in forward direction(from roots to
            leaves).
        observed_states_training_nodes: It is a (L * 2) dataframe where L is the number of training
            nodes where state values are known. First column should be the node
            number and the second column being the corresponding known state
            values of the nodes

    Returns:
        forward_probs: A dataframe of size (N * D) denoting the forward
        probabilites at each node of the tree, where "N" is possible no. of
        states and "D" is the total number of nodes in the tree
    """

    hmm["state_transition_probabilities"].fillna(0, inplace=True)
    number_of_levels = len(emission_observation)
    if verbose:
        print("Forward loop running")

    for m in range(number_of_levels):
        hmm["emission_probabilities"][m].fillna(0, inplace=True)
    number_of_observations = len(emission_observation[0])
    number_of_states = len(hmm["states"])

    # We define a variable 'forward_probabilities' which is a numpy array which will be denoting
    # the forward probabilities
    forward_probabilities = np.zeros(
        shape=(number_of_states, number_of_observations))
    # We transform the numpy array 'forward_probabilities' into a pandas DataFrame
    forward_probabilities = pd.DataFrame(
        data=forward_probabilities,
        index=hmm["states"],
        columns=range(number_of_observations))

    # avoid crash
    if observed_states_training_nodes is None:
        observed_states_training_nodes = pd.DataFrame(columns=["node","state"])
        
    for k in forward_tree_sequence:
        boolean_value = set([k]).issubset(list(observed_states_training_nodes["node"]))
        desired_state = list(observed_states_training_nodes["state"][observed_states_training_nodes["node"] == k])[0] if boolean_value else list()
        
        previous_state = np.nonzero(adjacent_matrix[:, k] != 0)[0]
        length_of_next_state = len(previous_state)

        if length_of_next_state == 0:
            for state in hmm["states"]:
                forward_probabilities.loc[state, k] = math.log(hmm["initial_probabilities"][state])
            if boolean_value:
                desired_state_index = np.where(desired_state != np.array(hmm["states"]))[0]
                # 'mapdf' is a dataframe which maps from old state to new state
                mapdf = np.array([[i, j] for i, j in zip(
                    range(number_of_states), hmm["states"])])
                mapdf = pd.DataFrame(
                    data=mapdf, columns=[
                        "old_states", "new_states"])
                mapdf["old_states"] = pd.to_numeric(mapdf["old_states"])
                tozero = list(
                    mapdf["new_states"][mapdf["old_states"].isin(desired_state_index)])[0]
                forward_probabilities.loc[tozero, k] = -math.inf
            continue

        prev_array = np.array(
            list(
                itertools.product(
                    hmm["states"],
                    repeat=length_of_next_state)))
        inter = list(set(previous_state) & set(observed_states_training_nodes.iloc[:, 0]))
        len_inter = len(inter)
        # 'true_boolean_array' is a boolean array with only True boolean elements
        true_boolean_array = np.repeat(True, prev_array.shape[0], axis=0)

        if len_inter != 0:
            for i in range(len_inter):
                ind = np.where(observed_states_training_nodes.iloc[:, 0] == inter[i])[0][0]
                ind1 = np.where(inter[i] == previous_state)[0][0]
                desired_state = observed_states_training_nodes.iloc[ind, 1]
                true_boolean_array = np.logical_and(
                    len(np.where(prev_array[:, ind1] == desired_state)[0]), true_boolean_array)

        # 'index_array' is a numpy array comprising the index of all the True values of true_boolean_array
        index_array = np.where(true_boolean_array)[0]

        for state in hmm["states"]:
            logsum = []

            for i in index_array:
                prev = 0
                for j in range(prev_array.shape[1]):
                    prev = prev + (forward_probabilities.loc[prev_array[i,j], previous_state[j]])

                output_ = noisy_or(hmm, prev_array[i, :], state)

                if output_ == 0:
                    temp = -math.inf
                else:
                    temp = prev + math.log(output_)

                if -math.inf < temp < 0:
                    logsum.append(temp)

            emit = 0
            for m in range(number_of_levels):
                if emission_observation[m][k] is not None:
                    try:
                        emit = math.log(hmm["emission_probabilities"][m].loc[state, emission_observation[m][k]]) + emit
                        if verbose:
                            print("Fwd Emit is working for node :", k)
                    except ValueError:
                        emit = -math.inf
                        break

            if not logsum:
                forward_probabilities.loc[state, k] = -math.inf
            else:
                forward_probabilities.loc[state, k] = logsumexp(logsum) + emit

        if boolean_value:
            old_states = range(number_of_states)
            new_states = hmm["states"]
            desired_state_index = np.where(
                desired_state != np.array(hmm["states"]))[0]
            mapdf = np.array([[i, j] for i, j in zip(old_states, new_states)])
            mapdf = pd.DataFrame(
                data=mapdf, columns=[
                    "old_states", "new_states"])
            mapdf["old_states"] = pd.to_numeric(mapdf["old_states"])
            tozero = mapdf["new_states"][mapdf["old_states"].isin(
                desired_state_index)].tolist()[0]
            forward_probabilities.loc[tozero, k] = -math.inf

    return forward_probabilities


def run_an_example_1():
    """sample run for noisy_or function"""
    from treehmm.initHMM import initHMM

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    emissions = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM(states, emissions)
    transition_prob = noisy_or(hmm,states,"P") # for transition from P & N simultaneously to P
    print(transition_prob)

def run_an_example_2():
    """sample run for forward function"""
    from treehmm.initHMM import initHMM
    from treehmm.fwd_seq_gen import forward_sequence_generator
    from scipy.sparse import csr_matrix

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    sparse_sample_tree = csr_matrix(sample_tree)
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    emissions = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM(states, emissions)
    emission_observation = [["L", "L", "R", "R", "L"]]
    forward_tree_sequence = forward_sequence_generator(sparse_sample_tree)
    data = {'node': [1], 'state': ['P']}
    observed_states_training_nodes = pd.DataFrame(data=data, columns=["node", "state"])

    forward_probs = forward(hmm, sparse_sample_tree, emission_observation,forward_tree_sequence,observed_states_training_nodes, True)
    print(forward_probs)

if __name__ == "__main__":
    run_an_example_1()
    run_an_example_2()
