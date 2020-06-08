import numpy as np
import pandas as pd
import initHMM as initHMM
import bwd_seq_gen as bwd_seq_gen
import fwd_seq_gen as fwd_seq_gen

import forward
import backward as backward

import baumWelch
from baumWelch import baumWelchRecursion
import pdb
import copy



tmat = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
states = ['P','N'] 
symbols = [['L','R']]
hmm = initHMM.initHMM(states,symbols,tmat)
#pdb.set_trace()
obsv = [["L","L","R","R","L"]]
bt_sq = bwd_seq_gen.bwd_seq_gen(hmm)
ft_seq = fwd_seq_gen.fwd_seq_gen(hmm)
data = {'node':[1], 'state':['P']}
kn_states = pd.DataFrame(data = data, columns=["node","state"])
#BackwardProbs = backward.backward(hmm,obsv,bt_sq,kn_states)
#Transprob = forward.noisy_or(hmm,states,"P")
#ForwardProbs = forward.forward(hmm,obsv,ft_seq,kn_states)
data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
#kn_st = pd.DataFrame(data = data,columns=["node","state"])
kn_vr = pd.DataFrame(data = data1,columns=["node","state"])
newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm),obsv,kn_states, kn_vr)
learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm),obsv,kn_states, kn_vr)
print("newparam :",newparam)
print("\n")
print("learntHMM : ",learntHMM)


#kn_st = data.frame(node=c(2),state=c("P"),stringsAsFactors = FALSE)
#kn_vr = data.frame(node=c(3,4,5),state=c("P","N","P"),stringsAsFactors = FALSE)
#newparam= baumWelchRecursion(hmmA,obsv,kn_st, kn_vr)
#learntHMM= baumWelch(hmmA,obsv,kn_st, kn_vr)



#ForwardProbs = forward(hmmA,obsv,ft_sq,kn_st)

# tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),5,5, byrow= TRUE )
# hmmA = initHMM(c("P","N"),list(c("L","R")), tmat)
# obsv = list(c("L","L","R","R","L"))
# bt_sq = bwd_seq_gen(hmmA)
#ft_sq = fwd_seq_gen(hmmA)

# kn_st = data.frame(node=c(3),state=c("P"),stringsAsFactors = FALSE)
# BackwardProbs = backward(hmmA,obsv,bt_sq,kn_st)
#Transprob = noisy_or(hmmA,c("P","N"),"P")




