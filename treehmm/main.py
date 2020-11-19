#!/usr/bin/env python3
import numpy as np
import pandas as pd
import treehmm.initHMM as initHMM
from treehmm.fwd_seq_gen import forward_sequence_generator
import treehmm.baumWelch as baumWelch
import copy

# Sample test variable declaration

def example1():
    sample_tree = np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(6,6)
    #sparse = csr_matrix(sample_tree)
    states = ['P','N']
    emissions = [['L','M','H'],['L','M','H']]
    state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states,emissions,sample_tree,state_transition_probabilities = state_transition_probabilities)

    # For finding the forward_tree_sequence
    forward_tree_sequence = forward_sequence_generator(hmm)

    # For finding the forward_tree_sequence
    backward_tree_sequence = forward_sequence_generator(hmm)

    # Declaring the emission_observation list
    emission_observation = [["L","M","H","M","L","L"],["M","L","H","H","L","L"]]

    # Declaring the observed_states_training_nodes
    data = {'node' : [0,3,4], 'state' : ['P','N','P']}
    observed_states_training_nodes = pd.DataFrame(data = data,columns=["node","state"])

    # Declaring the observed_states_validation_nodes
    data1 = {'node' : [1,2], 'state' : ['N','P']}
    observed_states_validation_nodes = pd.DataFrame(data = data1,columns=["node","state"])

    # For calculating the forward probabilities
    # ForwardProbs = forward.forward(hmm,emission_observation,forward_tree_sequence,observed_states_training_nodes)
    # print(ForwardProbs)

    # For calculating the backward probabilities
    # BackwardProbs = backward.backward(hmm,emission_observation,backward_tree_sequence,observed_states_training_nodes)
    # print(BackwardProbs)

    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)
    #learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)

    print("newparam :", newparam)
    print("\n")
    #print("learntHMM : ", learntHMM)
    #print("\n")

if __name__ == "__main__":
    example1()
