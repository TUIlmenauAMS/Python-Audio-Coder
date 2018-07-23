# -*- coding: utf-8 -*-
"""
Program to optimize cosine modulated low delay filter banks.

Gerald Schuller, July 2018
"""

import numpy as np
import scipy as sp
import scipy.signal as sig
from symFmatrix import symFmatrix
from polmatmult import polmatmult
from Dmatrix import Dmatrix
from Fa2h import Fa2h
from Gmatrix import Gmatrix

def optimfuncLDFB(x,N):
    '''function for optimizing an MDCT type filter bank.
    x: unknown matrix coefficients, N: Number of subbands.
    '''
    #Analysis folding matrix:
    Fa = symFmatrix(x[0:int(1.5*N)])
    Faz = polmatmult(Fa,Dmatrix(N))
    Faz = polmatmult(Faz,Gmatrix(x[int(1.5*N):(2*N)]))
    #baseband prototype function h:
    h = Fa2h(Faz)
    #'Fa2h is returning 2D array. Squeeze to make it 1D'
    #h= np.squeeze(h)
    
    #Frequency response of the the baseband prototype function at 1024 frequency sampling points
    #between 0 and Nyquist:
    w,H = sig.freqz(h,1,1024)
    #desired frequency response
    #Limit of desired pass band (passband is between -pb and +pb, hence ../2)
    pb = int(1024/N/2.0)
    #Ideal desired frequency response:
    Hdes = np.concatenate((np.ones(pb),np.zeros(1024-pb)))
    #transition band width to allow filter transition from pass band to stop band:
    tb = int(np.round(1.0*pb))
    #Weights for differently weighting errors in pass band, transition band and stop band:
    weights = np.concatenate((1.0*np.ones(pb), np.zeros(tb),1000*np.ones(1024-pb-tb)))
    #Resulting total error number as the sum of all weighted errors:
    err = np.sum(np.abs(H-Hdes)*weights)
    return err
    
if __name__ == '__main__':  #run the optimization
  import scipy as sp
  import scipy.signal
  import matplotlib.pyplot as plt

  N=4  #number of subbands
  s=2*N
  bounds=[(-14,14)]*s
  xmin = sp.optimize.differential_evolution(optimfuncLDFB, bounds, args=(N,), disp=True)
  print("error after optim.=",xmin.fun)
  print("optimized coefficients=",xmin.x)
  np.savetxt("LDFBcoeff.txt", xmin.x)
  x=xmin.x;
  #Baseband Impulse Response:
  Fa = symFmatrix(x[0:int(1.5*N)])
  #print("Fa=", Fa[:,:,0])
  Faz = polmatmult(Fa,Dmatrix(N))
  Faz = polmatmult(Faz,Gmatrix(x[int(1.5*N):(2*N)]))
  h = Fa2h(Faz)
  plt.plot(h)
  plt.xlabel('Sample')
  plt.ylabel('Value')
  plt.title('Baseband Impulse Response of the Low Delay Filter Bank')
  #Magnitude Response:
  w,H=sp.signal.freqz(h) 
  plt.figure()
  plt.plot(w,20*np.log10(abs(H)))
  plt.axis([0, 3.14, -60,20])
  plt.xlabel('Normalized Frequency')
  plt.ylabel('Magnitude (dB)')
  plt.title('Mag. Frequency Response of the Low Delay Filter Bank')
  plt.show()
