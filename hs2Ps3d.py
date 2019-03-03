# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 19:37:53 2016, from Matlab
@author: Max, Gerald Schuller
"""



def hs2Ps3d(hs,N):
    '''produces the polyphase matrix Ps, for the synthesis filter bank from
    a baseband filter hs with a cosine modulation''' 
    import numpy as np
    L=max(np.shape(hs))
    blocks = int(np.ceil(L/N))
    Ps = np.zeros((N,N,blocks))
    for k in range(N): #subband
        for nph in range(N): #Phase
            for m in range(blocks): #blocks
                n = m*N+nph;
                #Ps(k,nph,m) = hs[n]*np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(blocks*N-1-n+N/2.0+0.5));
                if np.remainder(blocks,2.0) == 0:
                    Ps[k,nph,m] = hs[n]*np.cos(np.pi/N*(k+0.5)*(n-N/2.0+0.5))*np.sqrt(2.0/N)
                else:
                    Ps[k,nph,m] = hs[n]*np.cos(np.pi/N*(k+0.5)*(n+N/2.0+0.5))*np.sqrt(2.0/N)                    
    return Ps;
