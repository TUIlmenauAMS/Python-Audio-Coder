# -*- coding: utf-8 -*-
"""
modified, Gerald Schuller, August 2017
"""
from __future__ import print_function
import numpy as np

def x2polyphase(x,N):
    """Converts input signal x (a 1D array) into a polyphase row vector 
    xp for blocks of length N, with shape: (1,N,#of blocks)"""     
    import numpy as np
    #Convert stream x into a 2d array where each row is a block:
    #xp.shape : (y,x, #num blocks):
    x=x[:int(len(x)/N)*N] #limit signal to integer multiples of N
    xp=np.reshape(x,(N,-1),order='F') #order=F: first index changes fastest
    #add 0'th dimension for function polmatmult:
    xp=np.expand_dims(xp,axis=0)
    return xp
    
if __name__ == '__main__':
  #testing:
  x=np.arange(0,9)
  xp=x2polyphase(x,4)
  print(xp[:,:,0], xp[:,:,1])    
