import numpy as np
import pandas as pd
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

def baumWelchRecursion (hmm, observation, kn_states=None, kn_verify=None):
	t_seq = [fwd_seq_gen(hmm),bwd_seq_gen(hmm)]
	treemat = hmm["adjacent_symmetry_matrix"]

	Transition_Matrix = hmm["state_transition_probabilities"].copy()
	Transition_Matrix.iloc[:,:] = 0
	Emission_Matrix = copy.deepcopy(hmm["emission_probabilities"])

	number_of_levels = len(observation)

	kn_verify.iloc[:,1][np.where(kn_verify.iloc[:,1]==np.array(hmm["States"])[0])[0]]=1
	kn_verify.iloc[:,1][np.where(kn_verify.iloc[:,1]==np.array(hmm["States"])[1])[0]]=0
	kn_verify.iloc[:,1] = kn_verify.iloc[:,1].astype('int32')


	fwd = forward(hmm,observation,t_seq[0],kn_states)
	bwd = backward(hmm,observation,t_seq[1],kn_states)
	
	#trv = False
	f_count  = 1
	b_count = 1

	f = fwd
	boolf = np.any(f.iloc[:,t_seq[0]]==None)
	b = bwd
	boolb = np.any(b.iloc[:,t_seq[1]]==None)

	number_of_states = len(hmm["States"])
	number_of_observations = len(observation[0])
	gam = f+b

	for i in t_seq[0]:
		summ = np.log(np.sum(np.exp(gam.iloc[:,i])))
		gam.iloc[:,i] = gam.iloc[:,i]-summ

	if kn_states is not None:
		pred_prob = np.array(np.exp(gam.loc[hmm["States"][0],kn_verify.iloc[:,0]]))
		act_prob = kn_verify.iloc[:,1]
		fg = pred_prob[np.where(act_prob==1)[0]]
		bg = pred_prob[np.where(act_prob==0)[0]]
		pos_class = np.ones(fg.shape[0])
		neg_class = np.zeros(bg.shape[0])
		y = np.concatenate([pos_class,neg_class])
		score = np.concatenate([fg, bg], axis=0)
		roc_obj = metrics.roc_auc_score(y,score)
		presicion, recall, threshold = precision_recall_curve(y,score)
		pr_obj = auc(recall,presicion)
	

	ps_st = np.array(list(itertools.product(hmm["States"],repeat=number_of_states)))
	links = np.transpose(np.nonzero(treemat))

	t_prob = {}

	for key in hmm["States"]:
		temp = np.zeros((len(links),number_of_states))
		t_prob[key] = pd.DataFrame(data=temp,index=range(len(links)),columns=hmm["States"])

	for i in range(links.shape[0]):
		for x in hmm["States"]:
			for y in hmm["States"]:
				emit = 0
				for m in range(number_of_levels):
					if observation[m][links[i,1]] != None:
						emit = math.log(hmm["emission_probabilities"][m].loc[y,observation[m][links[i,1]]]) + emit
					
				t_prob[y].loc[i,x] = f.loc[x,links[i,0]] + math.log(hmm["state_transition_probabilities"].loc[x,y]) + b.loc[y,links[i,1]] + emit

		shape_ = (len(t_prob.keys()), len(list(t_prob.values())[0].columns))
		arr_ = np.zeros(shape_)

		for idx, (k,df) in enumerate(t_prob.items()):
			arr_[idx,:] = df.iloc[i,:].tolist()

		summ = np.log(np.sum(np.exp(arr_)))

		for k in t_prob.keys():
			t_prob[k].iloc[i,:] -= summ
		
	for x in hmm["States"]:
		sumd = np.log(np.sum(np.exp(gam.loc[x,t_seq[0]])))
		for y in hmm["States"]:
			summ = np.log(np.sum(np.exp(t_prob[y].loc[:,x])))
			Transition_Matrix.loc[x,y] = np.exp(summ-sumd)

	for i in range(number_of_levels):
		Emission_Matrix[i].iloc[:,:] = 0

	for m in range(number_of_levels):
		for x in hmm["States"]:
			sumd = np.log(np.sum(np.exp(gam.loc[x,t_seq[0]])))
			for s in hmm["Symbols"][m]:
				indi = list(set(np.where(np.array(observation[m])==s)[0]) & set(t_seq[0]))
				summ = np.log(np.sum(np.exp(gam.loc[x,(indi)])))
				Emission_Matrix[m].loc[x,s] = np.exp(summ-sumd)

	if kn_states is None:
		return {"Transition_Matrix" : Transition_Matrix, "Emission_Matrix" : Emission_Matrix, "results" : gam}
	else:
		return {"Transition_Matrix" : Transition_Matrix, "Emission_Matrix" : Emission_Matrix, "results" : [roc_obj,pr_obj,gam]}


def baumWelch(hmm, observation,kn_states=None,kn_verify=None,maxIterations=50, delta=1e-5, pseudoCount=0):
	t_seq = [fwd_seq_gen(hmm),bwd_seq_gen(hmm)]
	tempHmm = copy.deepcopy(hmm)
	tempHmm["state_transition_probabilities"].fillna(0, inplace=True)

	number_of_levels = len(observation)

	kn_verify.iloc[:, 1][np.where(kn_verify.iloc[:, 1] == np.array(hmm["States"])[0])[0]]=1
	kn_verify.iloc[:, 1][np.where(kn_verify.iloc[:, 1] == np.array(hmm["States"])[1])[0]]=0
	kn_verify.iloc[:, 1] = kn_verify.iloc[:, 1].astype('int32')

	for m in range(number_of_levels):
		tempHmm["emission_probabilities"][m].fillna(0, inplace=True)

	diff = []
	iter_t = []
	auc_iter = []
	aupr_iter = []
	

	for i in range(maxIterations):
		print("Iteration_running: ", i)
		print("\n")
		start_time_it = time.time()

		bw = baumWelchRecursion(tempHmm, observation, kn_states, kn_verify)

		TM = bw["Transition_Matrix"].copy()
		EM = copy.deepcopy(bw["Emission_Matrix"])
		
		TM[hmm["state_transition_probabilities"].notna()] += pseudoCount

		for m in range(number_of_levels):
			EM[m][hmm["emission_probabilities"][m].notna()] += pseudoCount  
			

		# Maximization Step (Maximise Log-Likelihood for Transitions and Emissions-Probabilities)
		TM = (TM/np.tile(np.array(np.sum(TM,axis=1)).reshape(len(TM),1), (1,len(TM.columns))))

		for m in range(number_of_levels):
			EM[m] = (EM[m]/np.tile(np.array(np.sum(EM[m],axis=1)).reshape(len(EM[m]),1), (1,len(EM[m].columns))))

		summ = 0
		for m in range(number_of_levels):
			di = np.sqrt(np.sum(np.square(np.array(tempHmm["emission_probabilities"][m]-EM[m]))))
		
			summ = summ + di

		d = np.sqrt(np.sum(np.square(np.array(tempHmm["state_transition_probabilities"]-TM)))) + summ  
		print("Delta:", d)
		print("\n")

		diff.append(d) 

		tempHmm["state_transition_probabilities"] = TM

		for m in range(number_of_levels):
			tempHmm["emission_probabilities"][m] = EM[m]

		end_time_it = time.time()
		iter_time = end_time_it - start_time_it
		print(iter_time)
		print("\n")

		iter_t.append(iter_time)
		if kn_states is None:
			gama_iter[i] = bw["results"]
		else:
			auc_iter.append(bw["results"][0])
			aupr_iter.append(bw["results"][1])

		if np.all(d<delta):
			print("Convergence reached :")
			print("\n")

			break

	tempHmm["state_transition_probabilities"].fillna(0, inplace=True)
	
	for m in range(number_of_levels):
		tempHmm["emission_probabilities"][m].fillna(0, inplace=True)

	if kn_states is None:
		return {"hmm" : tempHmm, "stats" : diff, "finprob" : np.exp(bw["results"][2])}
	else:
		return {"hmm" : tempHmm, "stats" : [diff,auc_iter,aupr_iter], "finprob" : np.exp(bw["results"][2])}













