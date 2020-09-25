bwd_seq_gen= function(hmm,nlevel=100)
{
  adjm=hmm$adjsym
  pair=which(adjm!=0,arr.ind = TRUE)
  colnames(pair)=NULL
  roots=which(rowSums(adjm)==0 & colSums(adjm)!=0)
  order=list()
  order[[1]]=roots
  for(i in 1:nlevel)
  {
    prev_level=order[[i]]
    nxt_level=c()
    for(j in 1:length(prev_level))
    {
      nxt_nodes=pair[which(pair[,2]==prev_level[j]),1]
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
      to=which(adjm[order[[i]][j],]!=0)
      fbool=TRUE
      for(m in 1:length(to))
      {
        el=to[m]
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
  border=c()
  for(i in 1:length(order))
  {
    border=append(border,order[[i]])
  }
  return(border)
}