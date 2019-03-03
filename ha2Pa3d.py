# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:50:58 2016
from Matlab
@author: Max, Gerald Schuller
"""



def ha2Pa3d(ha,N):
    '''Produces the polyphase matrix Pa in 3D matrix representation from baseband
    filter ha with a cosine modulation
    N: Blocklength'''  
    import numpy as np
    L = max(np.shape(ha)) 
    blocks = int(np.ceil(L/N))
    Pa = np.zeros((N,N,blocks)) 
    for k in range(N): #subband
        for m in range(blocks): #m: block number
            for nph in range(N): #np: phase
                n = m*N+nph;
                #indexing like impulse response, phase index is reversed (N-np):
                Pa[N-nph-1,k,m] = ha[n]*np.cos(np.pi/N*(k+0.5)*(blocks*N-1-n-N/2.0+0.5))*np.sqrt(2.0/N)        
    return Pa
