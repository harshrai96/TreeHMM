#!/usr/bin/env python3
import numpy as np
import pandas as pd

def initHMM (States, Symbols, treemat, initial_probabilities = None, state_transition_probabilities = None, emission_probabilities = None):
	'''
	States = np.array(shape=(N,)) 
	Symbols = [np.array(shape=(N,)), ...]
	treemat = np.array(shape=(N,N))
	initial_probabilities = np.array(shape=(N,))
	state_transition_probabilities = np.array(shape=(N,N))
	emission_probabilities = [np.array(shape=(N,M)), ....]
	T = pandas.DataFrame
	E = [pandas.DataFrame, ...]
	S = dict()
	'''
	number_of_states = len(States)
	number_of_levels = len(Symbols)
	E=list()
	default_initial_probabilities = np.repeat((1/number_of_states),number_of_states, axis=0) 
	default_transition_probabilities = 0.5 * np.identity(number_of_states) + np.ones(shape=(number_of_states,number_of_states)) * (0.5/number_of_states) 
	S = dict(zip(States, default_initial_probabilities))
	T = pd.DataFrame(data=default_transition_probabilities, index=States, columns=States)

	if (initial_probabilities is not None):
		S = dict(zip(States, initial_probabilities))

	if (state_transition_probabilities is not None):
		state_transition_probabilities = np.array(state_transition_probabilities)
		T = pd.DataFrame(data=state_transition_probabilities, index=States, columns=States)

	for i in range(number_of_levels):
		number_of_symbols = len(Symbols[i])
		E.append(np.ones(shape=(number_of_states, number_of_symbols)) * (1/number_of_symbols))
		E[i] = pd.DataFrame(data=E[i], index=States, columns=Symbols[i])
		if (emission_probabilities is not None):
			E[i] = pd.DataFrame(data=emission_probabilities, index=States, columns=Symbols[i])


	return {"States" : States, "Symbols" : Symbols, "initial_probabilities" : S,
	          "state_transition_probabilities" : T, "emission_probabilities" : E, "adjacent_symmetry_matrix" : treemat}


if __name__ == "__main__":
	# sample call to the function
	tmat = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5) 
	states = ['P','N'] 
	symbols = [['L','R']] 
	hmm = initHMM(states,symbols,tmat)

	for k,v in hmm.items():
		print(k,v)
