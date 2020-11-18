
# Importing the libraries

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import time
from treehmm.fwd_seq_gen import forward_sequence_generator
from treehmm.bwd_seq_gen import backward_sequence_generator
from treehmm.forward import forward
from treehmm.backward import backward
import copy
from scipy.special import logsumexp

# Implementation of the Baum Welch Algorithm as a special case of Expectation Maximization algorithm
# The function hmm_train_and_test recursively calls this function to give a final estimate of parameters for tree HMM

# Defining the baumWelchRecursion function
eps = 1e-7

def multi_samples_baumWelch(hmm, adj_matrices, emission_observations):
    """
    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        adj_matrices: the adj_matrices for each observations
        emission_observations: emission_observations is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number

    Returns:
        newparam: A dictionary containing estimated Transition and Emission
        probability matrices
    """

    Transition_Matrix = hmm["state_transition_probabilities"].copy()
    Transition_Matrix.iloc[:, :] = 0
    Emission_Matrix = copy.deepcopy(hmm["emission_probabilities"])
    
    instances_tree_sequences = []
    instances_t_probs = []
    instances_gammas = []
    
    for adj_matrix, emission_observation in zip(adj_matrices, emission_observations):
         # 'tree_sequence' is the combined sequence consisting of both forward and backward tree sequences

        tree_sequence = [forward_sequence_generator(adj_matrix), backward_sequence_generator(adj_matrix)]
        instances_tree_sequences.append(tree_sequence)

        number_of_levels = len(emission_observation)
        for i in range(number_of_levels):
            Emission_Matrix[i].iloc[:, :] = eps


        # 'fwd' and 'bwd' are the forward and backward probabilities calculated with given custom arguments
        fwd = forward(hmm, emission_observation, tree_sequence[0], adj_matrix) #alpha
        bwd = backward(hmm, emission_observation, tree_sequence[1], adj_matrix) #beta

        # compute gamma
        number_of_states = len(hmm["states"])
        gamma = fwd + bwd
        for i in tree_sequence[0]:
            summ = logsumexp(gamma.iloc[:, i])
            if summ == -math.inf:
                pass
            else:
                gamma.iloc[:, i] = gamma.iloc[:, i] - summ     # Step where gamma is being normalised

        # 'nonzero_pos_adj_sym_mat_val' gives position of nonzero values in adjacent_symmetry_matrix
        nonzero_pos_adj_sym_mat_val = np.transpose(np.nonzero(adj_matrix))
        instances_gammas.append(gamma)

        t_prob = {}

        for key in hmm["states"]:
            temp = np.zeros((len(nonzero_pos_adj_sym_mat_val), number_of_states))
            t_prob[key] = pd.DataFrame(data=temp, index=range(len(nonzero_pos_adj_sym_mat_val)), columns=hmm["states"])

        for i in range(nonzero_pos_adj_sym_mat_val.shape[0]):
            for x in hmm["states"]:
                for y in hmm["states"]:
                    emit = 0
                    for m in range(number_of_levels):
                        if emission_observation[m][nonzero_pos_adj_sym_mat_val[i, 1]] is not None:
                            try:
                                emit = math.log(hmm["emission_probabilities"][m].loc[y, emission_observation[m][nonzero_pos_adj_sym_mat_val[i, 1]]]) + emit
                            except ValueError:
                                emit = -math.inf
                                break
                    try:
                        t_prob[y].loc[i, x] = f.loc[x, nonzero_pos_adj_sym_mat_val[i, 0]] + math.log(
                            hmm["state_transition_probabilities"].loc[x, y]) + b.loc[
                                                y, nonzero_pos_adj_sym_mat_val[i, 1]] + emit
                    except ValueError:
                        t_prob[y].loc[i, x] = -math.inf
                        break

            shape_ = (len(t_prob.keys()), len(list(t_prob.values())[0].columns))
            arr_ = np.zeros(shape_)

            for idx, (k, df) in enumerate(t_prob.items()):
                arr_[idx, :] = df.iloc[i, :].tolist()

            summ = logsumexp(arr_)
            if summ == -math.inf:
                pass
            else:
                for k in t_prob.keys():
                    t_prob[k].iloc[i, :] -= summ                     # Step where t_prob is being normalised
        
        instances_t_probs.append(t_prob)

    # TODO FIX THIS
    for x in hmm["states"]:
        sumd = 0
        for gamma, t_prob, tree_sequence in zip(instances_gammas, instances_t_probs, instances_tree_sequences):
            isnatce_sumd = logsumexp(gamma.loc[x, tree_sequence[0]])
            sumd = logsumexp([isnatce_sumd, sumd])
            for y in hmm["states"]:
                summ = logsumexp(t_prob[y].loc[:, x])
                Transition_Matrix.loc[x, y] = np.exp(summ - sumd)
    # TODO FIX THIS
    for m in range(number_of_levels):
        for x in hmm["states"]:
            sumd = logsumexp(gamma.loc[x, tree_sequence[0]])
            for s in hmm["emissions"][m]:
                indi = list(set(np.where(np.array(emission_observation[m]) == s)[0]) & set(tree_sequence[0]))
                if len(indi) > 0:
                    # if this emissions is observated
                    summ = logsumexp(gamma.loc[x, (indi)])
                    Emission_Matrix[m].loc[x, s] = np.exp(summ - sumd)

    return {"Transition_Matrix": Transition_Matrix, "Emission_Matrix": Emission_Matrix, "results": gamma}
    
# Inferring the parameters of a tree Hidden Markov Model via the Baum-Welch algorithm

# For an initial Hidden Markov Model (HMM) with some assumed initial parameters and a given set of observations at
# all the nodes of the tree, the Baum-Welch algorithm infers optimal parameters to the HMM. Since the Baum-Welch
# algorithm is a variant of the Expectation-Maximisation algorithm, the algorithm converges to a local solution which
# might not be the global optimum. Note that if you give the training and validation data, the function will message
# out AUC and AUPR values after every iteration. Also, validation data must contain more than one instance of either
# of the possible states

# Defining the hmm_train_and_test function

def train_hmm_multi_samples(
        hmm,
        adj_matrices,
        emission_observations,
        maxIterations=50,
        delta=1e-5,
        pseudoCount=0):
    #TODO：FIX THIS
    """inferred HMM whose representation is equivalent to the representation in
    initHMM.py, second being a list of statistics of algorithm and third being
    the final state probability distribution at all nodes.

    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        adj_matrices: It is a the  list of adj_matrices, one per each emission_observations
        emission_observations: emission_observation is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number
        maxIterations: It is the maximum number of iterations in the Baum-Welch
            algorithm. Default is 50
        delta: Additional termination condition, if the transition and emission
            matrices converge, before reaching the maximum number of iterations
            (code{maxIterations}). The difference of transition and emission
            parameters in consecutive iterations must be smaller than
            code{delta} to terminate the algorithm. Default is 1e-5
        pseudoCount: Adding this amount of pseudo counts in the estimation-step
            of the Baum-Welch algorithm. Default is zero

    Returns:
        learntHMM: A dictionary of three elements, first being the infered HMM whose representation is equivalent to the representation in initHMM, second being a list of statistics of algorithm and third being the final state probability distribution at all nodes.
    """
    # 'temporary_hmm' is a copy of the dictionary hmm
    temporary_hmm = copy.deepcopy(hmm)
    temporary_hmm["state_transition_probabilities"].fillna(0, inplace=True)

    temporary_hmm["emission_probabilities"][0].fillna(0, inplace=True)

    diff = []
    iter_t = []
    auc_iter = []
    aupr_iter = []

    for i in range(maxIterations):
        print("Iteration_running: ", i)
        print("\n")
        start_time_it = time.time()

        bw = baumWelch(temporary_hmm, emission_observation, observed_states_training_nodes, observed_states_validation_nodes)

        TM = bw["Transition_Matrix"].copy()
        EM = copy.deepcopy(bw["Emission_Matrix"])

        TM[hmm["state_transition_probabilities"].notna()] += pseudoCount

        for m in range(number_of_levels):
            EM[m][hmm["emission_probabilities"][m].notna()] += pseudoCount

        # Maximization Step (Maximise Log-Likelihood for Transitions and Emissions-Probabilities)
        TM = (TM / np.tile(np.array(np.sum(TM, axis=1)).reshape(len(TM), 1), (1, len(TM.columns))))

        for m in range(number_of_levels):
            EM[m] = (EM[m] / np.tile(np.array(np.sum(EM[m], axis=1)).reshape(len(EM[m]), 1), (1, len(EM[m].columns))))

        summ = 0
        for m in range(number_of_levels):
            di = np.sqrt(np.sum(np.square(np.array(temporary_hmm["emission_probabilities"][m] - EM[m]))))

            summ = summ + di

        d = np.sqrt(np.sum(np.square(np.array(temporary_hmm["state_transition_probabilities"] - TM)))) + summ
        print("Delta:", d)
        print("\n")

        diff.append(d)

        temporary_hmm["state_transition_probabilities"] = TM

        for m in range(number_of_levels):
            temporary_hmm["emission_probabilities"][m] = EM[m]

        end_time_it = time.time()
        iter_time = end_time_it - start_time_it
        print("iter_time = ", iter_time)
        print("\n")

        iter_t.append(iter_time)

        #if observed_states_training_nodes is None:
        #    gammaa_iter[i] = bw["results"]
        if observed_states_training_nodes is not None:
            auc_iter.append(bw["results"][0])
            aupr_iter.append(bw["results"][1])

        if np.all(d < delta):
            print("Convergence reached :")
            print("\n")
            break

    temporary_hmm["state_transition_probabilities"].fillna(0, inplace=True)

    for m in range(number_of_levels):
        temporary_hmm["emission_probabilities"][m].fillna(0, inplace=True)
    if observed_states_training_nodes is None:
        return {"hmm": temporary_hmm, "stats": diff, "finprob": np.exp(bw["results"][2])}
    else:
        return {"hmm": temporary_hmm, "stats": [diff, auc_iter, aupr_iter], "finprob": np.exp(bw["results"][2])}

def run_an_example_1():
    """sample run for baumWelchRecursion function"""
    import treehmm.initHMM
    #TODO FIX THIS    
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    emissions = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, emissions, sample_tree)
    data = {'node': [1], 'state': ['P']}
    observed_states_training_nodes = pd.DataFrame(data=data, columns=["node", "state"])
    data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
    observed_states_validation_nodes = pd.DataFrame(data = data1,columns=["node","state"])
    emission_observation = [["L", "L", "R", "R", "L"]]
    #newparam = baumWelchRecursion(copy.deepcopy(hmm),emission_observation,observed_states_training_nodes, observed_states_validation_nodes)
    print(newparam)

def run_an_example_2():
    """sample run for hmm_train_and_test function"""
    import treehmm.initHMM
    #TODO FIX THIS
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    emissions = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, emissions, sample_tree)
    data = {'node': [1], 'state': ['P']}
    observed_states_training_nodes = pd.DataFrame(data=data, columns=["node", "state"])
    data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
    observed_states_validation_nodes = pd.DataFrame(data = data1,columns=["node","state"])
    emission_observation = [["L", "L", "R", "R", "L"]]
    #learntHMM = hmm_train_and_test(copy.deepcopy(hmm),emission_observation,observed_states_training_nodes, observed_states_validation_nodes)
    print(learntHMM)


if __name__ == "__main__":
    pass
    #run_an_example_1()
    #run_an_example_2()
