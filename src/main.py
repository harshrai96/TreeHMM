#!/usr/bin/env python3
import numpy as np
import pandas as pd
import initHMM as initHMM
import bwd_seq_gen as bwd_seq_gen
import fwd_seq_gen as fwd_seq_gen
import forward
import backward as backward
import baumWelch
from baumWelch import baumWelchRecursion
import copy
import scipy
from scipy.sparse import csr_matrix
import rpy2.robjects as robjects

# R Code Examples run
#
# if __name__ == "__main__":
# sample_tree = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
# states = ['P','N']
# symbols = [['L','R']]
# hmm = initHMM.initHMM(states,symbols,sample_tree)
# forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)
# print(forward_tree_sequence)
# emission_observation = [["L","L","R","R","L"]]
# backward_tree_sequence = bwd_seq_gen.bwd_seq_gen(hmm)
#
#
# data = {'node':[1], 'state':['P']}
# kn_states = pd.DataFrame(data = data, columns=["node","state"])
# #BackwardProbs = backward.backward(hmm,emission_observation,backward_tree_sequence,kn_states)
# #Transprob = forward.noisy_or(hmm,states,"P")
# #ForwardProbs = forward.forward(hmm,emission_observation,forward_tree_sequence,kn_states)
# data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
# #observed_states_training_nodes = pd.DataFrame(data = data,columns=["node","state"])
# observed_states_validation_nodes = pd.DataFrame(data = data1,columns=["node","state"])
# newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm),emission_observation,kn_states, observed_states_validation_nodes)
# learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm),emission_observation,kn_states, observed_states_validation_nodes)
# print("newparam :",newparam)
# print("\n")
# print("learntHMM : ",learntHMM)
# print("\n")
# print(type(learntHMM))

# Oldman river dataset testing

# robjects.r['load']("train.RData")
# robjects.r['load']("valid.RData")
# robjects.r['load']("adjm.RData")
# data_frame_for_sample_tree_sparse_matrix = pd.read_csv("~/sparse_matrix.csv")
#
# data_frame_for_sample_tree_sparse_matrix["i"] = data_frame_for_sample_tree_sparse_matrix['i'] - 1
# data_frame_for_sample_tree_sparse_matrix["j"] = data_frame_for_sample_tree_sparse_matrix['j'] - 1
#
# rows_of_sparse_matrix = list(data_frame_for_sample_tree_sparse_matrix["i"])
# cols_of_sparse_matrix = list(data_frame_for_sample_tree_sparse_matrix["j"])
# data_inside_the_sparse_matrix = list(data_frame_for_sample_tree_sparse_matrix["x"])
#
# sparse_matrix = csr_matrix((data_inside_the_sparse_matrix, (rows_of_sparse_matrix, cols_of_sparse_matrix)),
#                           shape=(164960, 164960))  # .todense()
# dense_matrix = sparse_matrix.todense()
#
# # smaller sparse matrix
# sample_tree = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
# sparse_matrix_test = csr_matrix(sample_tree)
# #
# if __name__ == "__main__":
#     from csv import reader
#
#     # read csv file as a list of lists
#     with open('flabels_with_impute.csv', 'r') as read_obj:
#         # pass the file object to reader() to get the reader object
#         csv_reader = reader(read_obj)
#         # Pass reader object to list() to get a list of lists
#         list_of_rows = list(csv_reader)
#         new_list = list_of_rows[:]
#         col = np.transpose(new_list)
#         col_list = col.tolist()
#         col_test_list = col_list[0:9]
#
#
# sample_tree = sparse_matrix # adjm.Rdata
# states = ['P', 'N']
# emission_observation = col_test_list
# symbols = [['L', 'M', 'H'] for i in range(len(emission_observation))]
# state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
# hmm = initHMM.initHMM(states, symbols, sample_tree, state_transition_probabilities = state_transition_probabilities)
# data = {'node': list(robjects.r['kn_train'][0]), 'state': list(robjects.r['kn_train'][1])}  # kn_train.Rdata
# x = data.get("node")
# for i in range(len(x)):
#     x[i] = x[i] - 1
# kn_states = pd.DataFrame(data=data, columns=["node", "state"])
# data1 = {'node': list(robjects.r['kn_valid'][0]), 'state': list(robjects.r['kn_valid'][1])}  # Kn_vt.Rdata
# y = data1.get("node")
# for i in range(len(y)):
#     y[i] = y[i] - 1
# observed_states_validation_nodes = pd.DataFrame(data=data1, columns=["node", "state"])
# print("New kn_states and observed_states_validation_nodes with flabels_with_impute")
#
# newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm), emission_observation, kn_states, observed_states_validation_nodes)
# learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm), emission_observation, kn_states, observed_states_validation_nodes)
#
# print("newparam :", newparam)
# print("\n")
# print("learntHMM : ", learntHMM)
# print("\n")


# pdb.set_trace()
# print(hmm)
# hmm = initHMM.initHMM(states, symbols, sample_tree)
# print("Bwd_seq inititating")
# backward_tree_sequence = bwd_seq_gen.bwd_seq_gen(hmm)
# forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)
# print(forward_tree_sequence)

# def saveList(myList,filename):
#     # the filename should mention the extension 'npy'
#     np.save(filename,myList)
#     print("Saved successfully!")
#
# saveList(forward_tree_sequence,'forward_tree_seq_of_maindata.npy')

# def saveList(myList,filename):
#     # the filename should mention the extension 'npy'
#     np.save(filename,myList)
#     print("backward_tree_sequence Saved successfully!")
#
# saveList(backward_tree_sequence,'backward_tree_seq_of_maindata.npy')

# # pdb.set_trace()
#
# def loadList(filename):
#     # the filename should mention the extension 'npy'
#     tempNumpyArray = np.load(filename)
#     return tempNumpyArray.tolist()
#
#
# forward_tree_sequence = loadList('forward_tree_seq_of_maindata.npy')
# backward_tree_sequence = loadList('backward_tree_seq_of_maindata.npy')

# pdb.set_trace()
# print("CASE 3 : Check backward probs separately")

# forward_probs = forward.forward(hmm, emission_observation, forward_tree_sequence, kn_states)
# pdb.set_trace()
# # print(forward_probs)
# # forward_probs.to_csv(fwd_probs_test-data.csv, sep='\t')
#
# BackwardProbs = backward.backward(hmm,emission_observation,backward_tree_sequence,kn_states)
# # # BackwardProbs.to_csv('BackwardProbs_main_data.csv')
# # # print("code complete now debug it")
# pdb.set_trace()
# # print(BackwardProbs)
# # pdb.set_trace()
# print("new logsumexp with -inf w/o try except")




# Sample test variable declaration

def example1():
    sample_tree = np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(6,6)
    sparse = csr_matrix(sample_tree)
    states = ['P','N']
    symbols = [['L','M','H'],['L','M','H']]
    state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states,symbols,sparse,state_transition_probabilities = state_transition_probabilities)

    # For finding the forward_tree_sequence
    forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)

    # For finding the forward_tree_sequence
    backward_tree_sequence = bwd_seq_gen.bwd_seq_gen(hmm)

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
    learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)

    print("newparam :", newparam)
    print("\n")
    print("learntHMM : ", learntHMM)
    print("\n")

if __name__ == "__main__":
    example1()
