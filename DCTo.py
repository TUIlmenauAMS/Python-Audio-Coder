# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 17:51:22 2016
from Matlab
@author: Max, Gerald Schuller
"""
import numpy as np
def DCTo(N):
    """odd DCT with size NxN"""
    import numpy as np
    y = np.zeros((N,N,1))
    for n in range(N):
        for k in range(N):
            y[n,k,0] = np.cos(np.pi/N*(k+0.5)*(n+0.5));
    return np.sqrt(2.0/N)*y
    
#Testing:
if __name__ == '__main__':
  T=DCTo(4)
  print T[:,:,0]    
