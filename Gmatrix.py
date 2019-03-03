# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:44:26 2016
from Matlab
@author: Max, Gerald Schuller
"""


def Gmatrix(g):
    '''produces a zero delay matrix G(z), which has delays z^-1 multiplied
    with the g coefficients on the upper half of its diagonal. g is a row vector
    with N/2 coefficients.
    In a 3D polynomial representation (exponents of z^-1 are in the third dimension)
    N is number of subbands and size of the polynomial matrix (NxN)
    N is even'''
    import numpy as np

    N = max(np.shape(g))*2;    
    G = np.zeros((N,N,2));
    G[:,:,0] = np.fliplr(np.eye(N))
    G[:,:,1] = np.diag(np.concatenate((g,np.zeros(int(N/2)))))
    return G
