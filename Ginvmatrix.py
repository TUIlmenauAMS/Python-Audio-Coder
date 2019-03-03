# -*- coding: utf-8 -*-
"""
Inverse of zero delay matrix G(z)
Gerald Schuller, September 2017
"""


def Ginvmatrix(g):
    '''produces the inverse of the zero delay matrix G(z). It has delays z^-1 multiplied
    with the reverse ordered neg. g coefficients on the lower half of its diagonal. g is a 1-d vector
    with N/2 coefficients.
    In a 3D polynomial representation (exponents of z^-1 are in the third dimension).
    N is number of subbands and size of the polynomial matrix (NxN)
    N is even'''
    import numpy as np
    N = max(np.shape(g))*2;    
    G = np.zeros((N,N,2));
    G[:,:,0] = np.fliplr(np.eye(N))
    G[:,:,1] = np.diag(np.concatenate((np.zeros(int(N/2)), -np.flipud(g))))
    return G
