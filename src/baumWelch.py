# Importing the libraries

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import time
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from fwd_seq_gen import fwd_seq_gen
from bwd_seq_gen import bwd_seq_gen
from forward import forward
from backward import backward
import itertools
import copy
import scipy
from scipy.special import logsumexp

# Implementation of the Baum Welch Algorithm as a special case of Expectation Maximization algorithm
# The function baumWelch recursively calls this function to give a final estimate of parameters for tree HMM

# Defining the baumWelchRecursion function
def baumWelchRecursion(hmm, observation, kn_states=None, kn_verify=None):
    """
    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        observation: observation is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number
        kn_states: It is a (L * 2) dataframe where L is the number of training
            nodes where state values are known. First column should be the node
            number and the second column being the corresponding known state
            values of the nodes
        kn_verify: It is a (L * 2) dataframe where L is the number of validation
            nodes where state values are known. First column should be the node
            number and the second column being the corresponding known state
            values of the nodes

    Returns:
        newparam: A dictionary containing estimated Transition and Emission
        probability matrices
    """
    # 'tree_sequence' is the combined sequence consisting of both forward and backward tree sequences
    tree_sequence = [fwd_seq_gen(hmm), bwd_seq_gen(hmm)]
    adjacent_symmetry_matrix_values = hmm["adjacent_symmetry_matrix"]
    Transition_Matrix = hmm["state_transition_probabilities"].copy()
    Transition_Matrix.iloc[:, :] = 0
    Emission_Matrix = copy.deepcopy(hmm["emission_probabilities"])
    number_of_levels = len(observation)
    for i in range(number_of_levels):
        Emission_Matrix[i].iloc[:, :] = 0

    kn_verify.iloc[:, 1][np.where(kn_verify.iloc[:, 1] == np.array(hmm["states"])[0])[0]] = 1
    kn_verify.iloc[:, 1][np.where(kn_verify.iloc[:, 1] == np.array(hmm["states"])[1])[0]] = 0
    kn_verify.iloc[:, 1] = kn_verify.iloc[:, 1].astype('int32')

    # 'fwd' and 'bwd' are the forward and backward probabilities calculated with given custom arguments
    fwd = forward(hmm, observation, tree_sequence[0], kn_states)
    bwd = backward(hmm, observation, tree_sequence[1], kn_states)

    f = fwd
    b = bwd

    number_of_states = len(hmm["states"])
    gamma = f + b
    for i in tree_sequence[0]:
        summ = logsumexp(gamma.iloc[:, i])
        if summ == -math.inf:
            pass
        else:
            gamma.iloc[:, i] = gamma.iloc[:, i] - summ     # Step where gamma is being normalised

    if kn_states is not None:
        pred_prob = np.array(np.exp(gamma.loc[hmm["states"][0], kn_verify.iloc[:, 0]]))
        act_prob = kn_verify.iloc[:, 1]
        fg = pred_prob[np.where(act_prob == 1)[0]]
        bg = pred_prob[np.where(act_prob == 0)[0]]
        pos_class = np.ones(fg.shape[0])
        neg_class = np.zeros(bg.shape[0])
        y = np.concatenate([pos_class, neg_class])
        score = np.concatenate([fg, bg], axis=0)
        roc_obj = metrics.roc_auc_score(y, score)
        precision, recall, threshold = precision_recall_curve(y, score)
        pr_obj = auc(recall, precision)
        print("AUC : ", roc_obj)
        print("\n")

    # 'nonzero_pos_adj_sym_mat_val' gives position of nonzero values in adjacent_symmetry_matrix_values
    nonzero_pos_adj_sym_mat_val = np.transpose(np.nonzero(adjacent_symmetry_matrix_values))

    t_prob = {}

    for key in hmm["states"]:
        temp = np.zeros((len(nonzero_pos_adj_sym_mat_val), number_of_states))
        t_prob[key] = pd.DataFrame(data=temp, index=range(len(nonzero_pos_adj_sym_mat_val)), columns=hmm["states"])

    for i in range(nonzero_pos_adj_sym_mat_val.shape[0]):
        for x in hmm["states"]:
            for y in hmm["states"]:
                emit = 0
                for m in range(number_of_levels):
                    if observation[m][nonzero_pos_adj_sym_mat_val[i, 1]] is not None:
                        try:
                            emit = math.log(hmm["emission_probabilities"][m].loc[y, observation[m][nonzero_pos_adj_sym_mat_val[i, 1]]]) + emit
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

    for x in hmm["states"]:
        sumd = logsumexp(gamma.loc[x, tree_sequence[0]])
        for y in hmm["states"]:
            summ = logsumexp(t_prob[y].loc[:, x])
            Transition_Matrix.loc[x, y] = np.exp(summ - sumd)

    for m in range(number_of_levels):
        for x in hmm["states"]:
            sumd = logsumexp(gamma.loc[x, tree_sequence[0]])
            for s in hmm["symbols"][m]:
                indi = list(set(np.where(np.array(observation[m]) == s)[0]) & set(tree_sequence[0]))
                summ = logsumexp(gamma.loc[x, (indi)])
                Emission_Matrix[m].loc[x, s] = np.exp(summ - sumd)

    if kn_states is None:
        return {"Transition_Matrix": Transition_Matrix, "Emission_Matrix": Emission_Matrix, "results": gamma}
    else:
        return {"Transition_Matrix": Transition_Matrix, "Emission_Matrix": Emission_Matrix,
                "results": [roc_obj, pr_obj, gamma]}


# Inferring the parameters of a tree Hidden Markov Model via the Baum-Welch algorithm

# For an initial Hidden Markov Model (HMM) with some assumed initial parameters and a given set of observations at
# all the nodes of the tree, the Baum-Welch algorithm infers optimal parameters to the HMM. Since the Baum-Welch
# algorithm is a variant of the Expectation-Maximisation algorithm, the algorithm converges to a local solution which
# might not be the global optimum. Note that if you give the training and validation data, the function will message
# out AUC and AUPR values after every iteration. Also, validation data must contain more than one instance of either
# of the possible states

# Defining the baumWelch function

def baumWelch(
        hmm,
        observation,
        kn_states=None,
        kn_verify=None,
        maxIterations=50,
        delta=1e-5,
        pseudoCount=0):
    """inferred HMM whose representation is equivalent to the representation in
    initHMM.py, second being a list of statistics of algorithm and third being
    the final state probability distribution at all nodes.

    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        observation: observation is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number
        kn_states: It is a (L * 2) dataframe where L is the number of training
            nodes where state values are known. First column should be the node
            number and the second column being the corresponding known state
            values of the nodes
        kn_verify: It is a (L * 2) dataframe where L is the number of validation
            nodes where state values are known. First column should be the node
            number and the second column being the corresponding known state
            values of the nodes
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

    number_of_levels = len(observation)

    kn_verify.iloc[:, 1][np.where(kn_verify.iloc[:, 1] == np.array(hmm["states"])[0])[0]] = 1
    kn_verify.iloc[:, 1][np.where(kn_verify.iloc[:, 1] == np.array(hmm["states"])[1])[0]] = 0
    kn_verify.iloc[:, 1] = kn_verify.iloc[:, 1].astype('int32')

    for m in range(number_of_levels):
        temporary_hmm["emission_probabilities"][m].fillna(0, inplace=True)

    diff = []
    iter_t = []
    auc_iter = []
    aupr_iter = []

    for i in range(maxIterations):
        print("Iteration_running: ", i)
        print("\n")
        start_time_it = time.time()

        bw = baumWelchRecursion(temporary_hmm, observation, kn_states, kn_verify)

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

        if kn_states is None:
            gammaa_iter[i] = bw["results"]
        else:
            auc_iter.append(bw["results"][0])
            aupr_iter.append(bw["results"][1])

        if np.all(d < delta):
            print("Convergence reached :")
            print("\n")
            break

    temporary_hmm["state_transition_probabilities"].fillna(0, inplace=True)

    for m in range(number_of_levels):
        temporary_hmm["emission_probabilities"][m].fillna(0, inplace=True)
    if kn_states is None:
        return {"hmm": temporary_hmm, "stats": diff, "finprob": np.exp(bw["results"][2])}
    else:
        return {"hmm": temporary_hmm, "stats": [diff, auc_iter, aupr_iter], "finprob": np.exp(bw["results"][2])}

def run_an_example_1():
    """sample run for baumWelchRecursion function"""
    import initHMM
    import forward
    import backward
    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    symbols = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    data = {'node': [1], 'state': ['P']}
    kn_states = pd.DataFrame(data=data, columns=["node", "state"])
    data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
    kn_verify = pd.DataFrame(data = data1,columns=["node","state"])
    newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm),observation,kn_states, kn_verify)
    print(newparam)

def run_an_example_2():
    """sample run for baumWelch function"""
    import initHMM
    import fwd_seq_gen
    import bwd_seq_gen
    import baumWelchRecursion

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    symbols = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    data = {'node': [1], 'state': ['P']}
    kn_states = pd.DataFrame(data=data, columns=["node", "state"])
    data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
    kn_verify = pd.DataFrame(data = data1,columns=["node","state"])
    observation = [["L", "L", "R", "R", "L"]]
    learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm),observation,kn_states, kn_verify)
    print(learntHMM)


if __name__ == "__main__":
    run_an_example_1()
    run_an_example_2()
