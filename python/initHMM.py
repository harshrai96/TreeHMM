#!/usr/bin/env python3
import numpy as np
import pandas as pd

def initHMM (States, Symbols, treemat, startProbs = None, transProbs = None, emissionProbs = None):
	'''
	States = np.array(shape=(N,)) 
	Symbols = [np.array(shape=(N,)), ...]
	treemat = np.array(shape=(N,N))
	startProbs = np.array(shape=(N,))
	transProbs = np.array(shape=(N,N))
	emissionProbs = [np.array(shape=(N,M)), ....]
	T = pandas.DataFrame
	E = [pandas.DataFrame, ...]
	S = dict()
	'''
	nStates = len(States)
	nLevel = len(Symbols)
	E=list()
	default_startProb = np.repeat((1/nStates),nStates, axis=0) 
	default_transProb = 0.5 * np.identity(nStates) + np.ones(shape=(nStates,nStates)) * (0.5/nStates) 
	S = dict(zip(States, default_startProb))
	T = pd.DataFrame(data=default_transProb, index=States, columns=States)

	if (startProbs is not None):
		S = dict(zip(States, startProbs))

	if (transProbs is not None):
		transProbs = np.array(transProbs)
		T = pd.DataFrame(data=transProbs, index=States, columns=States)

	for i in range(nLevel):
		nSymbols = len(Symbols[i])
		E.append(np.ones(shape=(nStates, nSymbols)) * (1/nSymbols))
		E[i] = pd.DataFrame(data=E[i], index=States, columns=Symbols[i])
		if (emissionProbs is not None):
			E[i] = pd.DataFrame(data=emissionProbs, index=States, columns=Symbols[i])


	return {"States" : States, "Symbols" : Symbols, "startProbs" : S,
	          "transProbs" : T, "emissionProbs" : E, "adjsym" : treemat}


if __name__ == "__main__":
	# sample call to the function
	tmat = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5) 
	states = ['P','N'] 
	symbols = [['L','R']] 
	hmm = initHMM(states,symbols,tmat)

	for k,v in hmm.items():
		print(k,v)
