library("Matrix")
library("future")
library("matrixStats")
library("PRROC")
library("gtools")

plan(multiprocess)
#' Implementation of the Baum Welch Algorithm as a special case of EM algorithm
#'
#' \code{\link{baumWelch}} recursively calls this function to give a final estimate of parameters for tree HMM
#' Uses Parallel Processing to speed up calculations for large data. Should not be used directly.
#'
#' @param hmm hmm Object of class List given as output by \code{\link{initHMM}}
#' @param observation A list consisting "k" vectors for "k" features, each vector being a character series of discrete emmision values at different nodes serially sorted by node number
#' @param kn_states (Optional) A (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes  
#' @param kn_verify (Optional) A (L * 2) dataframe where L is the number of validation nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes  
#' @return List containing estimated Transition and Emission probability matrices
#' @examples
#' tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' hmmA = initHMM(c("P","N"),list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' obsv = list(c("L","L","R","R","L")) #emissions for the one feature for the 5 nodes in order 1:5
#' kn_st = data.frame(node=c(2),state=c("P"),stringsAsFactors = FALSE)
#'                    #state at node 2 is known to be "P"
#' kn_vr = data.frame(node=c(3,4,5),state=c("P","N","P"),stringsAsFactors = FALSE) 
#'                    #state at node 3,4,5 are "P","N","P" respectively
#' newparam= baumWelchRecursion(hmmA,obsv,kn_st, kn_vr)
#' @seealso \code{\link{baumWelch}}
#'
baumWelchRecursion = function(hmm, observation, kn_states=NULL, kn_verify=NULL)
{
  t_seq=list(fwd_seq_gen(hmm),bwd_seq_gen(hmm))
  treemat=hmm$adjsym
  TransitionMatrix    = hmm$transProbs
  TransitionMatrix[,] = 0
  EmissionMatrix      = hmm$emissionProbs
  nLevel = length(observation)
  kn_verify[,2][which(kn_verify[,2]==hmm$States[1])]=1
  kn_verify[,2][which(kn_verify[,2]==hmm$States[2])]=0
  kn_verify[,2]=as.integer(kn_verify[,2])

  for(i in 1:nLevel)
  {
    EmissionMatrix[[i]][,]   = 0
  }
  fwd = future::future(forward(hmm,observation,t_seq[[1]],kn_states))
  bwd = future::future(backward(hmm, observation,t_seq[[2]],kn_states))
  fb_start=Sys.time()
  message("Forward_backward loop started in parallel_processes:")
  message("\n")
  trv=FALSE
  f_count=1
  b_count=1
  while(trv==FALSE)
  {
    trv= future::resolved(fwd) & future::resolved(bwd)
    if(future::resolved(fwd)==TRUE & f_count==1)
    {
      f_time=Sys.time()
      message("Forward_loop finished. ")
      message(difftime(f_time,fb_start, units=c("auto")))
      #message("\n")
      f_count=0
    }
    if(future::resolved(bwd)==TRUE & b_count==1)
    {
      b_time=Sys.time()
      message("Backward_loop finished. ")
      message(difftime(b_time,fb_start, units=c("auto")))
      #message("\n")
      b_count=0
    }
  }

  f=future::value(fwd)
  boolf=any(is.na(f[,t_seq[[1]]]))
  b=future::value(bwd)
  boolb=any(is.na(b[,t_seq[[2]]]))
  
  nStates=length(hmm$States)
  nObservations =length(observation[[1]])
  gam=f+b
  for(x in 1:length(t_seq[[1]]))
  {
    i=t_seq[[1]][x]
    summ=matrixStats::logSumExp(gam[,i])
    gam[,i]=gam[,i]-summ
  }
  message("\n")
  if(is.null(kn_states)==FALSE)
  {
    pred_prob=exp(gam[hmm$States[1],kn_verify[,1]])
    act_prob=kn_verify[,2]
    fg= pred_prob[which(act_prob==1)]
    bg=pred_prob[which(act_prob==0)]
    roc_obj = PRROC::roc.curve(scores.class0 = fg,scores.class1 = bg,curve = T)
    pr_obj = PRROC::pr.curve(scores.class0 = fg,scores.class1 = bg,curve = T)
    message("AUC:",roc_obj$auc)
    message("\n")
    message("AUPR_Integral:",pr_obj$auc.integral)
    message("\n")
    message("AUPR_DG:",pr_obj$auc.davis.goadrich)
    message("\n")
  }
  ps_st=gtools::permutations(n=nStates,r=nStates,v=hmm$States)
  links=which(treemat!=0,arr.ind = TRUE)
  colnames(links)=NULL
  t_prob=array(0, c(dim(links)[1],nStates,nStates))
  dimnames(t_prob)=list(c(1:dim(links)[1]),hmm$States,hmm$States)

  for(i in 1:dim(links)[1])
  {
    for(x in hmm$States)
    {
      for(y in hmm$States)
      {
        emit=0
        for(m in 1:nLevel)
        {
          if(!is.na(observation[[m]][links[i,2]]))
            emit=log(hmm$emissionProbs[[m]][y,observation[[m]][links[i,2]]]) + emit
        }
        t_prob[i,x,y] = f[x,links[i,1]] + log(hmm$transProbs[x,y]) + b[y,links[i,2]] + emit
      }
    }
    summ=matrixStats::logSumExp(t_prob[i,,])
    t_prob[i,,]=t_prob[i,,]-summ
  }

  for(x in hmm$States)
  {
    sumd=matrixStats::logSumExp(gam[x,t_seq[[1]]])
    for(y in hmm$States)
    {
      summ=matrixStats::logSumExp(t_prob[,x,y])
      TransitionMatrix[x,y]=exp(summ-sumd)
    }
  }
  for(m in 1:nLevel)
  {
    for(x in hmm$States)
    {
      sumd=matrixStats::logSumExp(gam[x,t_seq[[1]]])
      for(s in hmm$Symbols[[m]])
      {
        indi=intersect(which(observation[[m]]==s),t_seq[[1]])
        summ=matrixStats::logSumExp(gam[x,indi])
        EmissionMatrix[[m]][x,s] = exp(summ-sumd)
      }
    }
  }
  if(is.null(kn_states))
    return(list(TransitionMatrix=TransitionMatrix,EmissionMatrix=EmissionMatrix,results= gam))
  else
    return(list(TransitionMatrix=TransitionMatrix,EmissionMatrix=EmissionMatrix, results= list(roc_obj,pr_obj,gam)))
}

#' Inferring the parameters of a tree Hidden Markov Model via the Baum-Welch algorithm
#'
#' For an initial Hidden Markov Model (HMM) with some assumed initial parameters and a given set of observations at all the nodes of the tree, the
#' Baum-Welch algorithm infers optimal parameters to the HMM. Since the Baum-Welch algorithm is a variant of the Expectation-Maximisation algorithm, the algorithm converges to a local solution which might not be the global optimum. 
#' Note that if you give the training and validation data, the function will message out AUC and AUPR values after every iteration. Also, validation data must contain more than one instance of either of the possible states
#' @param hmm hmm Object of class List given as output by \code{\link{initHMM}}
#' @param observation A list consisting "k" vectors for "k" features, each vector being a character series of discrete emmision values at different nodes serially sorted by node number
#' @param kn_states (Optional) A (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes  
#' @param kn_verify (Optional) A (L * 2) dataframe where L is the number of validation nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes  
#' @param maxIterations (Optional) The maximum number of iterations in the Baum-Welch algorithm. Default is 100
#' @param delta (Optional) Additional termination condition, if the transition and emission matrices converge, before reaching the maximum number of iterations (\code{maxIterations}). The difference
#' of transition and emission parameters in consecutive iterations must be smaller than \code{delta} to terminate the algorithm. Default is 1e-9
#' @param pseudoCount (Optional) Adding this amount of pseudo counts in the estimation-step of the Baum-Welch algorithm. Default is zero
#' @return List of three elements, first being the infered HMM whose representation is equivalent to the representation in \code{\link{initHMM}}, second being a list of statistics of algorithm and third being the final state probability distribution at all nodes. 
#' @examples
#' tmat= matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' hmmA= initHMM(c("P","N"),list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' obsv= list(c("L","L","R","R","L")) #emissions for the one feature for the 5 nodes in order 1:5
#' kn_st = data.frame(node=c(2),state=c("P"),stringsAsFactors = FALSE)
#'                    #state at node 2 is known to be "P"
#' kn_vr = data.frame(node=c(3,4,5),state=c("P","N","P"),stringsAsFactors = FALSE) 
#'                    #state at node 3,4,5 are "P","N","P" respectively
#' learntHMM= baumWelch(hmmA,obsv,kn_st, kn_vr)
#' @seealso \code{\link{baumWelchRecursion}}

baumWelch = function(hmm, observation,kn_states=NULL,kn_verify=NULL,maxIterations=50, delta=1E-5, pseudoCount=0)
{
  t_seq=list(fwd_seq_gen(hmm),bwd_seq_gen(hmm))
  tempHmm = hmm
  tempHmm$transProbs[is.na(hmm$transProbs)] = 0
  nLevel = length(observation)
  kn_verify[,2][which(kn_verify[,2]==hmm$States[1])]=1
  kn_verify[,2][which(kn_verify[,2]==hmm$States[2])]=0
  kn_verify[,2]=as.integer(kn_verify[,2])
  for(m in 1:nLevel)
  {
    tempHmm$emissionProbs[[m]][is.na(hmm$emissionProbs[[m]])] = 0
  }
  diff = c()
  iter_t=c()
  auc_iter=list()
  aupr_iter=list()
  for(i in 1:maxIterations)
  {
    message("Iteration_running: ",i)
    message('\n')
    start_time_it=Sys.time()
    # Expectation Step (Calculate expected Transitions and Emissions)
    bw = baumWelchRecursion(tempHmm, observation, kn_states, kn_verify)
    #message(bw)
    TM  = bw$TransitionMatrix
    EM  = bw$EmissionMatrix
    # Pseudocounts
    TM[!is.na(hmm$transProbs)]    = TM[!is.na(hmm$transProbs)]    + pseudoCount
    for(m in 1:nLevel)
    {
      EM[[m]][!is.na(hmm$emissionProbs[[m]])] = EM[[m]][!is.na(hmm$emissionProbs[[m]])] + pseudoCount
    }
    # Maximization Step (Maximise Log-Likelihood for Transitions and Emissions-Probabilities)
    TM = (TM/apply(TM,1,sum))
    for(m in 1:nLevel)
    {
      EM[[m]] = (EM[[m]]/apply(EM[[m]],1,sum))
    }
    summ=0
    for(m in 1:nLevel)
    {
      di= sqrt(sum((tempHmm$emissionProbs[[m]]-EM[[m]])^2))
      summ=summ+di
    }
    d = sqrt(sum((tempHmm$transProbs-TM)^2)) + summ
    message("Delta: ",d)
    message("\n")
    diff = c(diff, d)
    tempHmm$transProbs    = TM
    for(m in 1:nLevel)
    {
      tempHmm$emissionProbs[[m]] = EM[[m]]
    }
    end_time_it=Sys.time()
    iter_time=difftime(end_time_it,start_time_it, units=c("auto"))
    message(iter_time)
    message("\n")
    message("\n")
    iter_t=c(iter_t,iter_time)
    if(is.null(kn_states))
      gama_iter[[i]]=bw$results
    else
    {
      auc_iter[[i]]=bw$results[[1]]
      aupr_iter[[i]]=bw$results[[2]]
    }
    if(d<delta)
    {
      message("Convergence Reached")
      message("\n")
      break
    }
  }
  tempHmm$transProbs[is.na(hmm$transProbs)] = NA
  for(m in 1:nLevel)
  {
    tempHmm$emissionProbs[[m]][is.na(hmm$emissionProbs[[m]])] = NA
  }
  if(is.null(kn_states))
    return(list(hmm=tempHmm,stats=list(diff), finprob=exp(bw$results[[3]])))
  else
    return(list(hmm=tempHmm,stats=list(diff, auc_iter,aupr_iter), finprob=exp(bw$results[[3]])))
}
