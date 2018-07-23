# -*- coding: utf-8 -*-
"""
Program for optimizing an MDCT type filter bank with N subbands. 
Gerald Schuller, July 2018
"""


def optimfuncMDCT(x, N):
    """Computes the error function for the filter bank optimization
    for coefficients x, a 1-d array, N: Number of subbands""" 
    import numpy as np
    import scipy.signal as sig
    from polmatmult import polmatmult 
    from Dmatrix import Dmatrix
    from symFmatrix import symFmatrix
    from Fa2h import Fa2h
   
    #x = np.transpose(x) 
    Fa = symFmatrix(x)
    D = Dmatrix(N)
    Faz = polmatmult(Fa,D)
    h = Fa2h(Faz)
    h = np.hstack(h)
    w, H = sig.freqz(h,1,1024)
    pb = int(1024/N/2)
    Hdes = np.concatenate((np.ones((pb,1)) , np.zeros(((1024-pb, 1)))), axis = 0)
    tb = np.round(pb)
    weights = np.concatenate((np.ones((pb,1)) , np.zeros((tb, 1)), 1000*np.ones((1024-pb-tb,1))), axis = 0) 
    err = np.sum(np.abs(H-Hdes)*weights)
    return err  
    
if __name__ == '__main__':  #run the optimization
  import numpy as np
  import scipy as sp
  import scipy.optimize
  import scipy.signal
  import matplotlib.pyplot as plt
  from symFmatrix import symFmatrix
  from polmatmult import polmatmult 
  from Dmatrix import Dmatrix
  from Fa2h import Fa2h

  N=4
  #Start optimization with some starting point:
  x0 = -np.random.rand(int(1.5*N))
  print("starting error=", optimfuncMDCT(x0, N)) #test optim. function
  xmin = sp.optimize.minimize(optimfuncMDCT, x0, args=(N,), options={'disp':True})
  print("optimized coefficients=", xmin.x)
  np.savetxt("MDCTcoeff.txt", xmin.x)
  print("error after optim.=", optimfuncMDCT(xmin.x, N))
  #Baseband Impulse Response:
  Fa = symFmatrix(xmin.x) 
  Faz = polmatmult(Fa, Dmatrix(N))
  h = Fa2h(Faz)
  print("h=", h)
  plt.plot(h)
  plt.xlabel('Sample')
  plt.ylabel('Value')
  plt.title('Baseband Impulse Response of our Optimized MDCT Filter Bank')
  plt.figure()
  #Magnitude Response:
  w,H=sp.signal.freqz(h) 
  plt.plot(w,20*np.log10(abs(H)))
  plt.axis([0, 3.14, -60,20])
  plt.xlabel('Normalized Frequency')
  plt.ylabel('Magnitude (dB)')
  plt.title('Mag. Frequency Response of the MDCT Filter Bank') 
  plt.show()   
  
