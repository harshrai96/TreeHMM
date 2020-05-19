import numpy as np
import pandas as pd
import initHMM as initHMM
import bwd_seq_gen as bwd_seq_gen

import backward as backward



tmat = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
states = ['P','N'] 
symbols = [['L','R']]
hmm = initHMM.initHMM(states,symbols,tmat)
obsv = [["L","L","R","R","L"]]
bt_sq = bwd_seq_gen.bwd_seq_gen(hmm)
print(bt_sq)
data = {'node':[2], 'state':['P']}
kn_states = pd.DataFrame(data = data, columns=["node","state"])
BackwardProbs = backward.backward(hmm,obsv,bt_sq,kn_states)

print(BackwardProbs,"++++")
# tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),5,5, byrow= TRUE )
# hmmA = initHMM(c("P","N"),list(c("L","R")), tmat)
# obsv = list(c("L","L","R","R","L"))
# bt_sq = bwd_seq_gen(hmmA)
# kn_st = data.frame(node=c(3),state=c("P"),stringsAsFactors = FALSE)
# BackwardProbs = backward(hmmA,obsv,bt_sq,kn_st)




