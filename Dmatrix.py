# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 17:34:26 2016
from Matlab
@author: Max, Gerald Schuller
"""

import numpy as np
def Dmatrix(N):
    """produces a delay matrix D(z), which has delay z^-1 on the upper half of its diagonal
    in a 3D polynomial representation (exponents of z^-1 are in the third dimension) 
    N is number of subbands and size of the polynomial matrix (NxN) 
    N is even""" 
    D=np.zeros((N,N,2));    
    D[:,:,0] = np.diag(np.append(np.zeros((1,int(N/2))), np.ones((1,int(N/2)))));
    D[:,:,1] = np.diag(np.append(np.ones((1,int(N/2))), np.zeros((1,int(N/2)))));  
    return D;
