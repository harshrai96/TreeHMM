import numpy as np
import pandas as pd
import itertools
import math
import pdb
import ipdb

def noisy_or(hmm, prev_state, cur_state):

  l = len(np.where(np.array(prev_state) == np.array(hmm["States"][0]))[0])
  fin = math.pow(hmm["transProbs"].loc[hmm["States"][0],hmm["States"][1]],l)

  if cur_state ==hmm["States"][1]:
      return fin
  else:
      return 1-fin


def forward (hmm, observation, ft_seq, kn_states=None):
  '''
  observation = [[]]

  '''

  if kn_states is None:
    kn_states =  pd.DataFrame(columns=["node","state"])

  treemat = hmm["adjsym"]
  hmm["transProbs"].fillna(0, inplace=True)
  nLevel = len(observation)
  #ipdb.set_trace()
  for m in range(nLevel):
    hmm["emissionProbs"][m].fillna(0, inplace=True)

  nObservations = len(observation[0])

  nStates = len(hmm["States"])
  f = np.zeros(shape=(nStates,nObservations))
  f = pd.DataFrame(data=f, index=hmm["States"], columns=range(nObservations))
  
  for k in ft_seq:

    _bool = set([k]).issubset(list(kn_states["node"]))
    if _bool==True:
      st = list(kn_states["state"][kn_states["node"]==k])[0]
    from_state = np.where(treemat[:,k]!=0)[0]
    len_link = len(from_state)

    if len_link==0:
      for state in hmm["States"]:
          f.loc[state,k] = math.log(hmm["startProbs"][state])
      if _bool ==True:
        st_ind = np.where(st!=np.array(hmm["States"]))[0]
        mapdf = np.array([[i,j] for i,j in zip(range(nStates),hmm["States"])])
        mapdf = pd.DataFrame(data=mapdf, columns=["old","new"])
        mapdf["old"] = pd.to_numeric(mapdf["old"])
        #pdb.set_trace()
        tozero = list(mapdf["new"][mapdf["old"].isin(st_ind)])[0]
        f.loc[tozero,k] = -math.inf
      continue


    prev_array = np.array(list(itertools.product(hmm["States"],repeat=len_link)))
    inter = list(set(from_state) & set(kn_states.iloc[:,0]))
    len_inter = len(inter)
    t_value = np.repeat(True, prev_array.shape[0], axis=0)

    if len_inter != 0:
      for i in range(len_inter):
        ind = np.where(kn_states.iloc[:, 0] == inter[i])[0][0]
        ind1 = np.where(inter[i] == from_state)[0][0]
        st = kn_states.iloc[ind, 1]
        t_value = np.logical_and(len(np.where(prev_array[:, ind1] == st)[0]), t_value)

    ind_arr = np.where(t_value)[0]

    for state in hmm["States"]:
      logsum = []

      for i in ind_arr:
        prev = 0
        for j in range(prev_array.shape[1]):
          prev += f.loc[prev_array[i,j],from_state[j]]

        output_ = noisy_or(hmm,prev_array[i,:],state)

        if output_ == 0:
          temp = -math.inf
        else:
          temp = prev + math.log(output_)

        if (temp > -math.inf and temp <0):
          logsum.append(temp)

      emit = 0
      #print(hmm["emissionProbs"][m])
      for m in range(nLevel):
        if observation[m][k]!= None:
         # print(hmm["emissionProbs"][m].loc[state, observation[m][k]])
          #try:
          #pdb.set_trace()

          emit = math.log(hmm["emissionProbs"][m].loc[state, observation[m][k]]) + emit
          #print(emit)
          # except:
          #   emit = -math.inf
      #try:
      f.loc[state,k] = np.log(np.sum(np.exp(logsum))) + emit
      # except:
      #   f.loc[state, k] = -math.inf


    if _bool==True:
      old = range(nStates)
      new = hmm["States"]
      st_ind = np.where(st!=np.array(hmm["States"]))[0]
      mapdf = np.array([[i,j] for i,j in zip(old,new)])
      mapdf = pd.DataFrame(data=mapdf, columns=["old","new"] )
      mapdf["old"] = pd.to_numeric(mapdf["old"])
      tozero = mapdf["new"][mapdf["old"].isin(st_ind)].tolist()[0]
      f.loc[tozero,k] = -math.inf

  return f      