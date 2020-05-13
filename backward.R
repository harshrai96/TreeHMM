library("Matrix")
library("gtools")
library("matrixStats")

#' Infer the backward probabilities for all the nodes of the treeHMM
#'
#' \code{backward} calculates the backward probabilities for all the nodes
#'
#' The backward probability for state X and observation at node k is defined as the probability of observing the sequence of observations e_k+1, ... ,e_n under the condition that the state at node k is X.
#' That is:\cr\code{b[X,k] := Prob(E_k+1 = e_k+1, ... , E_n = e_n | X_k = X)}
#' \cr where \code{E_1...E_n = e_1...e_n} is the sequence of observed emissions and \code{X_k} is a random variable that represents the state at node \code{k}
#'
#' @param hmm hmm Object of class List given as output by \code{\link{initHMM}}
#' @param observation A list consisting "k" vectors for "k" features, each vector being a character series of discrete emmision values at different nodes serially sorted by node number
#' @param bt_seq A vector denoting the order of nodes in which the tree should be traversed in backward direction(from leaves to roots). Output of \code{\link{bwd_seq_gen}} function.
#' @param kn_states (Optional) A (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes
#' @return (N * D) matrix denoting the backward probabilites at each node of the tree, where "N" is possible no. of states and "D" is the total number of nodes in the tree
#' @examples
#' tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' hmmA = initHMM(c("P","N"),list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' obsv = list(c("L","L","R","R","L")) #emissions for the one feature for the 5 nodes in order 1:5
#' bt_sq = bwd_seq_gen(hmmA)
#' kn_st = data.frame(node=c(3),state=c("P"),stringsAsFactors = FALSE) 
#'                    #state at node 3 is known to be "P"
#' BackwardProbs = backward(hmmA,obsv,bt_sq,kn_st)
#' @seealso \code{\link{forward}}

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
