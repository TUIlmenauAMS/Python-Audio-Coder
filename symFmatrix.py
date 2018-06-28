# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 18:29:15 2016 from Matlab, author: Max, 
Gerald Schuller, Sep. 2017
"""
from __future__ import print_function  
import numpy as np
def symFmatrix(f):
    """produces a diamond shaped folding matrix F from the coefficients f
    (f is a 1-d array)
    which leads to identical analysis and synthesis baseband impulse responses
    Hence has det 1 or -1
    If N is number of subbands, then f is a vector of size 1.5*N coefficients.
    N is even
    returns: F of shape (N,N,1)
    """   
    sym=1.0; #The kind of symmetry: +-1
    N = int(len(f)/1.5);
    F=np.zeros((N,N,1))
    F[0:int(N/2),0:int(N/2),0]=np.fliplr(np.diag(f[0:int(N/2)]))
    F[int(N/2):N,0:int(N/2),0]=np.diag(f[int(N/2):N])
    F[0:int(N/2),int(N/2):N,0]=np.diag(f[N:(N+int(N/2))])
    ff = np.flipud((sym*np.ones((int(N/2))) - (f[N:(int(1.5*N))])*f[N-1:int(N/2)-1:-1])/f[0:int(N/2)])     
    F[int(N/2):N,int(N/2):N,0]=-np.fliplr(np.diag(ff))
    return F
    
if __name__ == '__main__':  
   N=4;
   n= np.arange(0,1.5*N)
   f=np.sin(np.pi/(2*N)*(n+0.5))
   Fa=symFmatrix(f)
   print( "Fa=\n", Fa[:,:,0])
   
