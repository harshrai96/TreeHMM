import numpy as np
import initHMM
import bwd_seq_gen as bwd_seq_gen
from scipy.sparse import csr_matrix

# sample call to the function
sample_tree = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
sparse_matrix = csr_matrix(sample_tree)
tmat = sparse_matrix
states = ['P','N'] 
symbols = [['L','M','H']] 
hmm = initHMM.initHMM(states,symbols,tmat)
print(hmm)

#b = bwd_seq_gen.bwd_seq_gen(hmm)

#print(b)