
#' Calculate the order in which nodes in the tree should be traversed during the forward pass(roots to leaves)
#'
#' Tree is a complex graphical model where we can have multiple parents and multiple children for a node. Hence the order in which the tree should be tranversed becomes significant. Forward algorithm is a dynamic programming problem where to calculate the values at a node,
#' we need the values of the parent nodes beforehand, which need to be traversed before this node. This algorithm outputs a possible(not unique) order of the traversal of nodes ensuring that the parents are traversed first before the children.
#'
#' @param hmm hmm Object of class List given as output by \code{\link{initHMM}}
#' @param nlevel No. of levels in the tree, if known. Default is 100
#' @return Vector of length "D", where "D" is the number of nodes in the tree
#' @examples
#' tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' hmmA = initHMM(c("A","B"),list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' ft_sq = fwd_seq_gen(hmmA)
#' @seealso \code{\link{forward}}

fwd_seq_gen= function(hmm,nlevel=100)
{
  adjm=hmm$adjsym
  pair=which(adjm!=0,arr.ind = TRUE)
  colnames(pair)=NULL
  roots=which(rowSums(adjm)!=0 & colSums(adjm)==0)
  order=list()
  order[[1]]=roots
  for(i in 1:nlevel)
  {
    prev_level=order[[i]]
    nxt_level=c()
    for(j in 1:length(prev_level))
    {
      nxt_nodes=pair[which(pair[,1]==prev_level[j]),2]
      nxt_level=c(nxt_level,nxt_nodes)
    }
    if(length(nxt_level)==0)
    {
      break
    }
    order[[i+1]]=unique(nxt_level)
  }
  order[[(length(order)+1)]]=vector()
  l=length(order)
  for(i in 2:(l-1))
  {
    shift=c()
    for(j in 1:length(order[[i]]))
    {
      from=which(adjm[,order[[i]][j]]!=0)
      fbool=TRUE
      for(m in 1:length(from))
      {
        el=from[m]
        for(k in (i-1):1)
        {
          bool=(el %in% order[[k]])
          if(bool==TRUE)
          {
            break
          }
        }
        fbool= fbool & bool
        if(fbool==FALSE)
        {
          break
        }
      }
      if(fbool==FALSE)
      {
        shift=c(shift,order[[i]][j])
      }
    }
    order[[i]]=unique(order[[i]][! (order[[i]] %in% shift)])
    order[[i+1]]=unique(append(order[[i+1]],shift))
  }
  forder=c()
  for(i in 1:length(order))
  {
    forder=append(forder,order[[i]])
  }
  return(forder)
}
