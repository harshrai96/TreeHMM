import numpy as np
import pandas as pd
import math
import itertools
import pdb

def backward (hmm, observation, bt_seq, kn_states=None):
  '''
  observation = [[]]

  '''

  if kn_states is None:

    kn_states =  pd.DataFrame(columns=["node","state"])

  treemat = hmm["adjsym"]
  hmm["transProbs"].fillna(0, inplace=True)
  nLevel = len(observation)
  
  for m in range(nLevel):
    hmm["emissionProbs"][m].fillna(0, inplace=True)

  nObservations = len(observation[0])

  nStates = len(hmm["States"])
  b = np.zeros(shape=(nStates,nObservations))
  b = pd.DataFrame(data=b, index=hmm["States"], columns=range(nObservations))
  
  for k in bt_seq:

    _bool = set([k]).issubset(list(kn_states["node"]))
    if _bool==True:
      st = list(kn_states["state"][kn_states["node"]==k])[0]
    nxt_state = np.where(treemat[k,:]!=0)[0]
    len_link = len(nxt_state)

    if len_link==0:
      if _bool ==True:
        st_ind = np.where(st!=np.array(hmm["States"]))[0]
        mapdf = np.array([[i,j] for i,j in zip(range(nStates),hmm["States"])])
        mapdf = pd.DataFrame(data=mapdf, columns=["old","new"] )
        mapdf["old"] = pd.to_numeric(mapdf["old"])
        tozero = list(mapdf["new"][mapdf["old"].isin(st_ind)])[0]
        b.loc[tozero,k] = -math.inf
      continue


    next_array = np.array(list(itertools.product(hmm["States"],repeat=len_link)))
    inter = list(set(nxt_state) & set(kn_states.iloc[:,0]))
    len_inter = len(inter)
    t_value = np.repeat(True, next_array.shape[0], axis=0)
    if len_inter != 0:
      for i in range(len_inter):
        ind = np.where(kn_states.iloc[:, 0] == inter[i])[0][0]
        ind1 = np.where(inter[i] == nxt_state)[0][0]
        st = kn_states.iloc[ind, 1]
        t_value = np.logical_and(len(np.where(next_array[:, ind1] == st)[0]), t_value)



    ind_arr = np.where(t_value)[0]
    for state in hmm["States"]:
      logsum = []
      for i in ind_arr:
        temp = 0
        for j in range(next_array.shape[1]):
          #try:
          emit = np.sum([math.log(hmm["emissionProbs"][m].loc[state, observation[m][k]]) for m in range(nLevel) if observation[m][k]!= None])
          #except:
            # emit = -math.inf
          # emit = 0
          # for m in range(nLevel):
          #   if observation[m][k]!= None:#Doubtful
          #     emit = math.log(hmm["emissionProbs"][m].loc[state, observation[m][k]]) + emit
          #try:
          temp += b.loc[next_array[i, j], nxt_state[j]] + math.log(hmm["transProbs"].loc[state, next_array[i, j]]) + emit
          # except :
          #   temp = -math.inf

        if (temp > -math.inf and temp <0):
          logsum.append(temp)

      b.loc[state,k] = np.log(np.sum(np.exp(logsum)))

    if _bool==True:
      old = range(nStates)
      new = hmm["States"]
      st_ind = np.where(st!=np.array(hmm["States"]))[0]
      mapdf = np.array([[i,j] for i,j in zip(old,new)])
      mapdf = pd.DataFrame(data=mapdf, columns=["old","new"] )
      mapdf["old"] = pd.to_numeric(mapdf["old"])
      tozero = mapdf["new"][mapdf["old"].isin(st_ind)].tolist()[0]
      b.loc[tozero,k] = -math.inf

  return b      



