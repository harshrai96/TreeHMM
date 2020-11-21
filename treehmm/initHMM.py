#!/usr/bin/env python3

# This file initializes the treeHMM with given parameters

# Importing the libraries

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Defining the initHMM function

def initHMM(
        states,
        emissions,
        initial_probabilities=None,
        state_transition_probabilities=None,
        random_init_state_transition_probabilities = False,
        emission_probabilities=None,
        random_init_emission_probabilities = False):
    """
    Args:
        states: It is a numpy array with first element being discrete state
            value for the cases(or positive) and second element being discrete
            state value for the controls(or negative) for given treeHMM
        emissions: It is a list of numpy array consisting discrete values of
            emissions(where "M" is the possible number of emissions) for each
            feature variable
        adjacent_symmetry_matrix: It is the Adjacent Symmetry Matrix that
            describes the topology of the tree
        initial_probabilities: It is a numpy array of shape (N * 1) containing
            initial probabilities for the states, where "N" is the possible
            number of states(Optional). Default is equally probable states
        random_init_state_transition_probabilities: boolean value, if true, random
            init init_state_transition_probabilities
        state_transition_probabilities: It is a numpy array of shape (N * N)
            containing transition probabilities for the states, where "N" is the
            possible number of states(Optional).
        emission_probabilities: It is a list of numpy arrays of shape (N * M)
            containing emission probabilities for the states, for each feature
            variable(optional). Default is equally probable emissions
        random_init_emission_probabilities: boolean value, if true, random
            init init_random_init_emission_probabilities

    Returns:
        hmm: A dictionary describing the parameters of treeHMM
    """

    
    number_of_states = len(states)
    number_of_levels = len(emissions)
    emission_probabilities_values = list()
    default_initial_probabilities = np.repeat(
        (1 / number_of_states), number_of_states, axis=0)
    default_transition_probabilities = 0.5 * np.identity(number_of_states) + np.ones(shape=(number_of_states, number_of_states)) * (0.5 / number_of_states)
    initial_probabilities_values = dict(
        zip(states, default_initial_probabilities))
    transition_probabilities_values = pd.DataFrame(
        data=default_transition_probabilities, index=states, columns=states)

    if initial_probabilities is not None:
        initial_probabilities_values = dict(zip(states, initial_probabilities))

    # random init state transition
    if random_init_state_transition_probabilities:
        state_transition_probabilities = np.random.dirichlet(np.ones(number_of_states),size=number_of_states)
         
    if state_transition_probabilities is not None: 
        state_transition_probabilities = np.array(
            state_transition_probabilities)
        transition_probabilities_values = pd.DataFrame(
             data=state_transition_probabilities, index=states, columns=states)
    
    # init emssion matrices
    for i in range(number_of_levels):
        number_of_symbols = len(emissions[i])
        if not random_init_emission_probabilities:
            # uniform init
            emission_probabilities_values.append(np.ones(shape=(number_of_states, number_of_symbols)) * (1 / number_of_symbols))
        else:
            # random init
            emission_probabilities_values.append(np.random.dirichlet(np.ones(number_of_symbols),size=number_of_states))
        emission_probabilities_values[i] = pd.DataFrame(data=emission_probabilities_values[i], index=states, columns=emissions[i])
        if emission_probabilities is not None:
            emission_probabilities_values[i] = pd.DataFrame(data=emission_probabilities[i], index=states, columns=emissions[i])

    return {
        "states": states,
        "emissions": emissions,
        "initial_probabilities": initial_probabilities_values,
        "state_transition_probabilities": transition_probabilities_values,
        "emission_probabilities": emission_probabilities_values}

def run_an_example():
    """
    sample run for initHMM function
    """
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)
    states = ['P', 'N']
    emissions = [['L', 'R']]
    hmm = initHMM(states, emissions, sample_tree)

if __name__ == "__main__":
    run_an_example()
