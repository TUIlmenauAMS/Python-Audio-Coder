#Functions to implement the complete Low Delay filter bank. File based, it first reads in the complete audio file and then computes the Low Delay filter bank output.
#Gerald Schuller, September 2017.
from __future__ import print_function
from symFmatrix import symFmatrix
from polmatmult import polmatmult 
from Dmatrix import Dmatrix
from Gmatrix import Gmatrix
from DCT4 import *
from x2polyphase import *
   
def LDFBana(x,N,fb):
   #Low Delay analysis filter bank.
   #Arguments: x: input signal, e.g. audio signal, a 1-dim. array
   #N: number of subbands
   #fb: coefficients for the MDCT filter bank, for the F matrix, np.array with 1.5*N coefficients.
   #returns y, consisting of blocks of subband in in a 2-d array of shape (N,# of blocks)
   
   Fa=symFmatrix(fb[0:int(1.5*N)])
   print("Fa.shape=",Fa.shape)
   D=Dmatrix(N)
   G=Gmatrix(fb[int(1.5*N):(2*N)])
   y=x2polyphase(x,N)
   print("y[:,:,0]=", y[:,:,0])
   y=polmatmult(y,Fa)   
   y=polmatmult(y,D)
   y=polmatmult(y,G)
   y=DCT4(y)
   #strip first dimension:
   y=y[0,:,:]
   return y

from Dinvmatrix import Dinvmatrix
from Ginvmatrix import *
from polyphase2x import *   
   
def LDFBsyn(y,fb):
   #Low Delay synthesis filter bank.
   #Arguments: y: 2-d array of blocks of subbands, of shape (N, # of blokcs)
   #returns xr, the reconstructed signal, a 1-d array.   
   
   Fa=symFmatrix(fb[0:int(1.5*N)])
   #invert Fa matrix for synthesis after removing last dim:
   Fs=np.linalg.inv(Fa[:,:,0])
   #add again last dimension for function polmatmult:
   Fs=np.expand_dims(Fs, axis=-1)
   Ginv=Ginvmatrix(fb[int(1.5*N):(2*N)])
   Dinv=Dinvmatrix(N)
   #Display the synthesis folding matrix Fs(z):
   Fsz=polmatmult(polmatmult(Ginv,Dinv),Fs)
   #add first dimension to y for polmatmult:
   y=np.expand_dims(y,axis=0)
   xp=DCT4(y)
   xp=polmatmult(xp,Ginv)
   xp=polmatmult(xp,Dinv)
   xp=polmatmult(xp,Fs)
   xr=polyphase2x(xp)
   return xr





#Testing:
if __name__ == '__main__':
   import numpy as np
   import matplotlib.pyplot as plt

   #Number of subbands:
   N=4

   #D=Dmatrix(N)
   #Dinv=Dinvmatrix(N)
   #Filter bank coefficients, 1.5*N of sine window:
   #fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   fb=np.loadtxt("LDFBcoeff.txt")
   print("fb=", fb)
   #input test signal, ramp:
   x=np.arange(64)
   plt.plot(x)
   plt.title('Input Signal')
   plt.xlabel('Sample')
   plt.show()
   y=LDFBana(x,N,fb)
   plt.imshow(np.abs(y))
   plt.title('LDFB Subbands')
   plt.xlabel('Block No.')
   plt.ylabel('Subband No.')
   plt.show()
   xr=LDFBsyn(y,fb)
   plt.plot(xr)
   plt.title('Reconstructed Signal')
   plt.xlabel('Sample')
   plt.show()
   #Input to the synthesis filter bank: unit pulse in lowest subband
   #to see its impulse response:
   y=np.zeros((4,16))
   y[0,0]=1
   xr=LDFBsyn(y,fb)
   plt.plot(xr[0:4*N])
   plt.title('Impulse Response of Modulated Synthesis Subband 0')
   plt.xlabel('Sample')
   plt.show()

