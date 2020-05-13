import numpy as np
import pandas as pd


def backward (hmm, observation,bt_seq, kn_states=NULL):
	if (np.any(kn_states==NONE)):
		node = list()
	    state = list()
	    kn_states =  pd.DataFrame(index = node, columns = state)#Doubtful............

	treemat=hmm$adjsym#To change...............
	treemat = initHmm(treemat)
	 
    hmm$transProbs[is.na(hmm$transProbs)] = 0#To change...........			
    initHMM(T)

    nLevel=len(observation)

    for m in range(nLevel):
    	{
    hmm$emissionProbs[[m]][is.na(hmm$emissionProbs[[m]])] = 0#To change...........
  }
    nObservations = len (observation[1])
    nStates = len (hmm$States)#To change...........
    b = np.ones(nStates,nObservations) * NA
    b = pd.DataFrame()#To change...........
     





backward= function (hmm, observation,bt_seq, kn_states=NULL)
{
  if(is.null(kn_states))
    kn_states=data.frame(node=integer(),state=character(),stringsAsFactors = F)
  treemat=hmm$adjsym
  hmm$transProbs[is.na(hmm$transProbs)] = 0
  nLevel = length(observation)
  for(m in 1:nLevel)
  {
    hmm$emissionProbs[[m]][is.na(hmm$emissionProbs[[m]])] = 0
  }
  nObservations =length(observation[[1]])
  nStates = length(hmm$States)
  b = array(NA,c(nStates, nObservations))
  dimnames(b) = list(states = hmm$States, index = 1:nObservations)
  for (x in 1:length(bt_seq))
  {
    k=bt_seq[x]
    bool= k %in% kn_states[,1]
    if(bool==TRUE)
    {
      ind=match(k,kn_states[,1])
      st=kn_states[ind,2]
    }
    nxtstate=which(treemat[k,]!=0)
    len_link=length(nxtstate)
    if(len_link==0)
    {
      for (state in hmm$States)
      {
        b[state,k]=0
      }
      if(bool==TRUE)
      {
        st_ind=which(st!=hmm$States)
        mapdf = data.frame(old=c(1:nStates),new=hmm$States)
        tozero=as.character(mapdf$new[match(st_ind,mapdf$old)])
        b[tozero,k]=-Inf
      }
      next
    }
    next_array=gtools::permutations(n=nStates, r=len_link, v=hmm$States, repeats.allowed = TRUE )
    inter= intersect(nxtstate, kn_states[,1])
    len_inter=length(inter)
    t_value=rep(TRUE,dim(next_array)[1])
    if(len_inter!=0)
    {
      for(i in 1:len_inter)
      {
        ind=match(inter[i],kn_states[,1])
        ind1=match(inter[i], nxtstate)
        st=kn_states[ind,2]
        t_value=which(next_array[,ind1]==st) & t_value
      }
    }
    ind_arr=which(t_value)
    for (state in hmm$States)
    {
      logsum=c()
      for (d in 1:length(ind_arr))
      {
        i=ind_arr[d]
        temp=0
        for (j in 1:dim(next_array)[2])
        {
          emit=0
          for(m in 1:nLevel)
          {
            if(!is.na(observation[[m]][k]))
              emit=log(hmm$emissionProbs[[m]][state, observation[[m]][k]]) + emit
          }
          temp = temp + (b[next_array[i,j], nxtstate[j]] + log(hmm$transProbs[state, next_array[i,j]]) + emit)
        }
        if(temp > - Inf & temp< 0)
        {
          logsum = c(logsum,temp)
        }
      }
      b[state, k] = matrixStats::logSumExp(logsum)
    }
    if(bool==TRUE)
    {
      st_ind=which(st!=hmm$States)
      mapdf = data.frame(old=c(1:nStates),new=hmm$States)
      tozero=as.character(mapdf$new[match(st_ind,mapdf$old)])
      b[tozero,k]=-Inf
    }
  }
  return(b)
}





