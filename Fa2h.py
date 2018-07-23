# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 17:59:00 2016
from Matlab
@author: Tobias Heinl, Gerald Schuller

Function extracts analysis baseband impulse response (reverse window function) 
from the folding matrix Fa of a cosine 
modulated filter bank, with modulation function for the analysis IR : 
h_k(n)=h(n)*cos(pi/N*(k+0.5)*(L-1-n+0.5-N/2));

"""

 
def Fa2h(Fa):
   """ Function extracts analysis baseband impulse response (reverse window function) 
   from the folding matrix Fa of a cosine modulated filter bank.
   """
   import numpy as np
   from polmatmult import polmatmult
   [N,y,blocks] = np.shape(Fa) 
   h0 = np.zeros(blocks * N)
   #First column of DCT-4:
   T = np.zeros((N,1,1)) 
   T[:,0,0]=np.cos(np.pi/N*(0.5)*(np.arange(N)+0.5))
   #Compute first column of Polyphase matrix Pa(z):
   Pa =  polmatmult(Fa,T)
   #Extract impulse response h0(n):
   for m in range(blocks): 
      h0[m*N+np.arange(N)] = np.flipud(Pa[:,0,m]) 
   #Baseband prototype h(n), divide by modulation func.:
   #h = h0 / np.cos(np.pi/N*0.5*(np.arange(blocks*N)+ 0.5+(N/2)))
   h = -h0 / np.cos(np.pi/N*0.5*(np.arange(blocks*N-1,-1,-1)+ 0.5-(N/2)))
   return h;  
    
    
#Testing:
if __name__ == '__main__':
  import numpy as np
  from symFmatrix import *
  from Dmatrix import *
  from polmatmult import *
  f=np.arange(1,7)
  print("f=", f)
  Fa=symFmatrix(f)
  N=4
  D=Dmatrix(N)
  Faz=polmatmult(Fa,D)
  print("Faz=", Faz[:,:,0],"\n", Faz[:,:,1])
  h=Fa2h(Faz)
  print("h=", h)
      
