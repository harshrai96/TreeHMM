import numpy as np
import pandas as pd
import initHMM
import pdb

def bwd_seq_gen(hmm,nlevel=100):
	adjm=hmm["adjsym"]
	pair = np.transpose(np.nonzero(adjm)) 

   #colnames(pair)=NULL #To change

	row_sums = np.sum(adjm,axis=1)
	col_sums = np.sum(adjm,axis=0)
	roots = np.where(np.logical_and(row_sums==0, col_sums!=0))[0] # np.array()
	order=list()
	order.append(roots) # [array]

	#pdb.set_trace()
	for o in order:
		prev_level = o
		nxt_level=np.array([pair[list(np.where(pair[:,1]==i)[0]),0] for i in prev_level]) 
		nxt_level = np.unique(nxt_level)
		if (len(nxt_level)==0):
			break
		order.append(nxt_level)

	#order.append(np.array([]))
	pdb.set_trace()
	l = len(order)

	for i in range(1,l-1):
		shift = [] 

		for j in order[i]:
			to = np.where(adjm[j,:]!=0)[0]
			fbool = set(to).issubset(np.hstack(order[:i]))
			if fbool==False:
				shift.append(j) 
		element_to_update = [i for i in order[i] if i not in shift]
		order[i]= np.unique(element_to_update)
		order[i+1] = np.unique(list(order[i+1]) + shift)

	border = []
	for i in order:
		border = border + list(i)

	return border  


 