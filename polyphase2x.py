# -*- coding: utf-8 -*-
"""
 Gerald Schuller, August 2017
"""
from __future__ import print_function
import numpy as np

def polyphase2x(xp):
   """Converts polyphase input signal xp (a row vector) into a contiguos row vector
   For block length N, for 3D polyphase representation (exponents of z in the third 
   matrix/tensor dimension)"""

   x=np.reshape(xp,(1,1,-1),order='F') #order=F: first index changes fastest
   x=x[0,0,:]
   return x

if __name__ == '__main__':
  #testing:
  from x2polyphase import *
  x=np.arange(0,8)
  xp=x2polyphase(x,4)
  xrek=polyphase2x(xp)
  print(xrek)
