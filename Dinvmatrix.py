# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 18:34:05 2016
from Matlab
@author: Max, Gerald Schuller
"""


def Dinvmatrix(N):
    """produces a causal inverse delay matrix D^{-1}(z), which has 
    delays z^-1  on the lower half in 3D polynomial representation (exponents of z^-1 are in third dimension)
    N is the number of subbands and size of the polynomial matrix (NxN)
    N is even"""
    import numpy as np
    D = np.zeros((N,N,2))
    D[:,:,0] = np.diag((np.append(np.ones((1,int(N/2))),np.zeros((1,int(N/2))))))
    D[:,:,1] = np.diag((np.append(np.zeros((1,int(N/2))),np.ones((1,int(N/2))))))
    return D
