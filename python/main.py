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



if __name__ == "__main__":
	sample_tree = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
	states = ['P','N'] 
	symbols = [['L','R']]
	hmm = initHMM.initHMM(states,symbols,sample_tree)
	observation = [["L","L","R","R","L"]]
	backward_tree_sequence = bwd_seq_gen.bwd_seq_gen(hmm)
	forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)
	data = {'node':[1], 'state':['P']}
	kn_states = pd.DataFrame(data = data, columns=["node","state"])
	#BackwardProbs = backward.backward(hmm,observation,backward_tree_sequence,kn_states)
	#Transprob = forward.noisy_or(hmm,states,"P")
	#ForwardProbs = forward.forward(hmm,observation,forward_tree_sequence,kn_states)
	data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
	#kn_st = pd.DataFrame(data = data,columns=["node","state"])
	kn_verify = pd.DataFrame(data = data1,columns=["node","state"])
	newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm),observation,kn_states, kn_verify)
	learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm),observation,kn_states, kn_verify)
	print("newparam :",newparam)
	print("\n")
	print("learntHMM : ",learntHMM)
	print("\n")
	print(type(learntHMM))
	#print(BackwardProbs)





