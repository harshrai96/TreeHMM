#!/usr/bin/env python3
import numpy as np
import pandas as pd
import initHMM


def fwd_seq_gen(hmm,number_of_levels=100):

  adjm=hmm["adjacent_symmetry_matrix"]
  pair = np.transpose(np.nonzero(adjm)) 


  row_sums = np.sum(adjm,axis=1)
  col_sums = np.sum(adjm,axis=0)
  roots = np.where(np.logical_and(row_sums!=0, col_sums==0))[0] # np.array()
  order=list()
  order.append(roots) # [array]

  for o in order:
    previous_level = o
    next_level=np.array([pair[list(np.where(pair[:,0]==i)[0]),1] for i in previous_level]) 
    next_level = np.unique(next_level)
    if (len(next_level)==0):
      break
    order.append(next_level)

  order.append(np.array([]))

  l = len(order)
  for i in range(1,l-1):
    shift = [] 

    for j in order[i]:
      _from = np.where(adjm[:,j]!=0)[0]
      fbool = set(_from).issubset(np.hstack(order[:i]))
      if fbool==False:
        shift.append(j) 
    element_to_update = [i for i in order[i] if i not in shift]
    order[i]= np.unique(element_to_update)
    order[i+1] = np.unique(list(order[i+1]) + shift)
  
  forder = []
  for i in order:
    forder = forder + list(i)
  
  return forder  


if __name__ == "__main__":
  import initHMM
  # sample call to the function
  tmat = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5) 
  states = ['P','N'] 
  symbols = [['L','R']] 
  hmm = initHMM.initHMM(states,symbols,tmat)

  f = fwd_seq_gen(hmm)

  print(f)








        






