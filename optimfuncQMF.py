# -*- coding: utf-8 -*-
"""
Python programs to optimize an PQMF filter bank with N subbands
Gerald Schuller, July 2018
"""


from __future__ import print_function
def optimfuncQMF(x,N):
    """Optimization function for a PQMF Filterbank
    x: coefficients to optimize (first half of prototype h), N: Number of subbands
    err: resulting total error
    """
    import numpy as np
    import scipy as sp
    import scipy.signal as sig 

    h = np.append(x,np.flipud(x));
    #H = sp.freqz(h,1,512,whole=True)
    f,H_im = sig.freqz(h)
    H=np.abs(H_im) #only keeping the real part
    posfreq = np.square(H[0:int(512/N)]);
    #Negative frequencies are symmetric around 0:
    negfreq = np.flipud(np.square(H[0:int(512/N)]))
    #Sum of magnitude squared frequency responses should be close to unity (or N)
    unitycond = np.sum(np.abs(posfreq+negfreq - 2*(N*N)*np.ones(int(512/N))))/512;
    #plt.plot(posfreq+negfreq);
    #High attenuation after the next subband:
    att = np.sum(np.abs(H[int(1.5*512/N):]))/512;
    #Total (weighted) error:
    err = unitycond + 100*att;
    return err
    
if __name__ == '__main__':  #run the optimization
   from scipy.optimize import minimize
   import scipy as sp
   import matplotlib.pyplot as plt

   N=4 #Number of subbands
   #Start optimization with "good" starting point:
   x0 = 16*sp.ones(4*N)
   print("starting error=", optimfuncQMF(x0,N)) #test optim. function
   xmin = minimize(optimfuncQMF,x0, args=(N,), method='SLSQP')
   print("error after optim.=",xmin.fun)
   print("optimized coefficients=",xmin.x)
   #Store the found coefficients in a text file: 
   sp.savetxt("QMFcoeff.txt", xmin.x)
   #we compute the resulting baseband prototype function:
   h = sp.concatenate((xmin.x,sp.flipud(xmin.x)))
   plt.plot(h)
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.title('Baseband Impulse Response of the Optimized PQMF Filter Bank')
   #plt.xlim((0,31))
   plt.show()
   #The corresponding frequency response:
   w,H=sp.signal.freqz(h) 
   plt.plot(w,20*sp.log10(abs(H)))
   plt.axis([0, 3.14, -100,20])
   plt.xlabel('Normalized Frequency')
   plt.ylabel('Magnitude (dB)')
   plt.title('Mag. Frequency Response of the PQMF Filter Bank')
   plt.show()
   #Checking the "unity condition":
   posfreq = sp.square(abs(H[0:int(512/N)]));
   negfreq = sp.flipud(sp.square(abs(H[0:int(512/N)])))
   plt.plot(posfreq+negfreq)
   plt.xlabel('Frequency (512 is Nyquist)')
   plt.ylabel('Magnitude')
   plt.title('Unity Condition, Sum of Squared Magnitude of 2 Neigh. Subbands')
   plt.show() 
   
