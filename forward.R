library("Matrix")
library("gtools")
library("matrixStats")

#' Calculating the probability of transition from multiple nodes to given node in the tree
#'
#' @param hmm Object of class List given as output by \code{\link{initHMM}},
#' @param prev_state vector containing state variable values for the previous nodes
#' @param cur_state character denoting the state variable value for current node
#' @return The Noisy_OR probability for the transition
#' @examples
#' tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' hmmA = initHMM(c("P","N"),list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' Transprob = noisy_or(hmmA,c("P","N"),"P") #for transition from P & N simultaneously to P

noisy_or =function(hmm, prev_state, cur_state)
{
  l=length(which(prev_state==hmm$States[1]))
  fin=(hmm$transProbs[hmm$States[1],hmm$States[2]])^l
  if(cur_state==hmm$States[2])
    return(fin)
  else
    return(1-fin)
}

#' Infer the forward probabilities for all the nodes of the treeHMM
#'
#' \code{forward} calculates the forward probabilities for all the nodes
#'
#' The forward probability for state X up to observation at node k is defined as the probability of observing the sequence of observations e_1,..,e_k given that the state at node k is X.
#' That is:\cr\code{f[X,k] := Prob( X_k = X | E_1 = e_1,.., E_k = e_k)}
#' \cr where \code{E_1...E_n = e_1...e_n} is the sequence of observed emissions and \code{X_k} is a random variable that represents the state at node \code{k}
#'
#' @param hmm hmm Object of class List given as output by \code{\link{initHMM}}
#' @param observation A list consisting "k" vectors for "k" features, each vector being a character series of discrete emmision values at different nodes serially sorted by node number
#' @param ft_seq A vector denoting the order of nodes in which the tree should be traversed in forward direction(from roots to leaves). Output of \code{\link{fwd_seq_gen}} function.
#' @param kn_states (Optional) A (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes 
#' @return (N * D) matrix denoting the forward probabilites at each node of the tree, where "N" is possible no. of states and "D" is the total number of nodes in the tree
#' @examples
#' tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                 5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' hmmA = initHMM(c("P","N"),list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' obsv = list(c("L","L","R","R","L")) #emissions for the one feature for the 5 nodes in order 1:5
#' ft_sq = fwd_seq_gen(hmmA)
#' kn_st = data.frame(node=c(3),state=c("P"),stringsAsFactors = FALSE) 
#'                    #state at node 3 is known to be "P"
#' ForwardProbs = forward(hmmA,obsv,ft_sq,kn_st)
#' @seealso \code{\link{backward}}

forward = function (hmm, observation,ft_seq, kn_states=NULL)
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
  f = array(NA, c(nStates, nObservations))
  dimnames(f) = list(states = hmm$States, index = 1:nObservations)
  for (x in 1:length(ft_seq))
  {
    k=ft_seq[x]
    bool= k %in% kn_states[,1]
    if(bool==TRUE)
    {
      ind=match(k,kn_states[,1])
      st=kn_states[ind,2]
    }
    fromstate=which(treemat[,k]!=0)
    len_link= length(fromstate)
    if(len_link==0)
    {
      for (state in hmm$States)
      {
        f[state,k]=log(hmm$startProbs[state])
      }
      if(bool==TRUE)
      {
        st_ind=which(st!=hmm$States)
        mapdf = data.frame(old=c(1:nStates),new=hmm$States)
        tozero=as.character(mapdf$new[match(st_ind,mapdf$old)])
        f[tozero,k]=-Inf
      }
      next
    }
    prev_array=gtools::permutations(n=nStates, r=len_link, v=hmm$States, repeats.allowed = TRUE)
    inter= intersect(fromstate, kn_states[,1])
    len_inter=length(inter)
    t_value=rep(TRUE,dim(prev_array)[1])
    if(len_inter!=0)
    {
      for(i in 1:len_inter)
      {
        ind=match(inter[i],kn_states[,1])
        ind1=match(inter[i], fromstate)
        st=kn_states[ind,2]
        t_value=which(prev_array[,ind1]==st) & t_value
      }
    }
    ind_arr=which(t_value)

    for (state in hmm$States)
    {
      logsum=c()
      for (d in 1:length(ind_arr))
      {
        i=ind_arr[d]
        prev=0
        for (j in 1:dim(prev_array)[2])
        {
          prev=prev + (f[prev_array[i,j], fromstate[j]])
        }

        temp = prev + log(noisy_or(hmm,prev_array[i,],state))
        if(temp > - Inf & temp < 0)
        {
          logsum = c(logsum,temp)
        }
      }
      emit=0
      for(m in 1:nLevel)
      {
        if(!is.na(observation[[m]][k]))
          emit=log(hmm$emissionProbs[[m]][state, observation[[m]][k]]) + emit
      }
      f[state, k] = matrixStats::logSumExp(logsum) + emit
    }
    if(bool==TRUE)
    {
      st_ind=which(st!=hmm$States)
      mapdf = data.frame(old=c(1:nStates),new=hmm$States)
      tozero=as.character(mapdf$new[match(st_ind,mapdf$old)])
      f[tozero,k]=-Inf
    }
  }
  return(f)
}
