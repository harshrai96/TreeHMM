import numpy as np
import initHMM


tmat = np.array([0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]).reshape(5,5) 
states = ['P','N'] 
symbols = [['L','R']] 
initHMM.initHMM(states,symbols,tmat)
