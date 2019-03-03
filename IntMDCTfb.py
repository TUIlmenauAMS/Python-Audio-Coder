#Functions to implement the Integer-to-Integer MDCT filter bank. File based, it first reads in the complete audio file and then computes the MDCT filter bank output.
#Only works with stereo input!
#Gerald Schuller, July 2018.

from MDCTfb import *
#from Dmatrix import *
#from Dinvmatrix import *
from LiftingFmat import *

def IntMDCTanafb(x,N,fb):
   #IntMDCT analysis filter bank.
   #usage: y0,y1=IntMDCTanafb(x,N,fb)
   #Arguments: x: integer valued input signal, e.g. audio signal, 
   #must be a 2-dim. (stereo) array, 
   #2nd index is channel index
   #N: number of subbands
   #fb: coefficients for the IntMDCT filter bank, for the F matrix, np.array with 1.5*N coefficients.
   #returns y0, y1, consisting of blocks of subbands in in a 2-d array of shape (N,# of blocks)
   
   F0,L0,L1=LiftingFmat(fb)
   D=Dmatrix(N)
   #2 polyphase arrays, 1 for each channel:
   xwin=np.zeros((1,N,int(len(x[:,0])/N)+1,2))
   for chan in range(2):  #iterate over the 2 stereo channels
      y=x2polyphase(x[:,chan],N)
      #Lifting step for F, add last (z) dimension for polmatmult:  
      y=polmatmult(y,np.expand_dims(F0,axis=-1)) 
      y=np.round(y)   #rounding after Listing step
      y=polmatmult(y,np.expand_dims(L0,axis=-1)) 
      y=np.round(y)
      y=polmatmult(y,np.expand_dims(L1,axis=-1)) 
      y=np.round(y)
      y=polmatmult(y,D)
      xwin[:,:,:,chan]=y
   #The IntDCT using multidimensional Lifting:
   y0=xwin[:,:,:,1]
   y1=xwin[:,:,:,0]
   y0=y0+np.round(DCT4(y1))
   y1=y1-np.round(DCT4(y0))
   y0=y0+np.round(DCT4(y1))
   y0=-y0  #compensate for the sign change of the Lifting implementaion of the stereo DCT4 (see above)
   #test:
   #y0=np.round(DCT4(xwin[:,:,:,0]))
   #y1=np.round(DCT4(xwin[:,:,:,1]))
   return y0[0,:,:], y1[0,:,:]
   
def IntMDCTsynfb(y0,y1,fb):
   #IntMDCT synthesis filter bank.
   #usage: xrek=IntMDCTsynfb(y0,y1,fb)
   #Arguments: y0,y1: integer valued subband signals from a stereo file, 
   #consisting of blocks of subbands in in a 2-d array of shape (N,# of blocks)
   #fb: coefficients for the IntMDCT filter bank, for the F matrix, np.array with 1.5*N coefficients.
   #returns xrec, the reconstructed stereo audio signal
   #The synthesis is obtained by inverting each step of the IntMDCTanafb.
   
   F0,L0,L1=LiftingFmat(fb)
   N=len(y0[:,0])
   Dinv=Dinvmatrix(N)
   #2 polyphase arrays, 1 for each channel:
   xwin=np.zeros((1,N,len(y0[0,:]),2))
   xrek=np.zeros((1,N*(len(y0[0,:])+1),2))
   y0=np.expand_dims(y0,axis=0)
   y1=np.expand_dims(y1,axis=0)
   y0=-y0  #compensate for the sign change of the Lifting implelentaion of the stereo DCT4
   #The inverse IntDCT:
   y0=y0-np.round(DCT4(y1))
   y1=y1+np.round(DCT4(y0))
   y0=y0-np.round(DCT4(y1))
   xwin[:,:,:,1]=y0
   xwin[:,:,:,0]=y1
   for chan in range(2):
      x=xwin[:,:,:,chan] #iterate of the 2 stereo channels
      x=polmatmult(x,Dinv)
      #inverse Lifting steps for the inverse F matrix, 
      #add last (z) dimension for polmatmult: 
      x=polmatmult(x,np.expand_dims(np.linalg.inv(L1),axis=-1)) 
      x=np.round(x)  #rounding after Listing step
      x=polmatmult(x,np.expand_dims(np.linalg.inv(L0),axis=-1)) 
      x=np.round(x)
      x=polmatmult(x,np.expand_dims(np.linalg.inv(F0),axis=-1)) 
      x=np.round(x)   
      xrek[:,:,chan]=polyphase2x(x)
   return xrek[0,:,:]
   
#Testing:
if __name__ == '__main__':
   import numpy as np
   import matplotlib.pyplot as plt
   import sympy
   
   #The transform can be an orthonormal DCT4, where T=T**(-1), 1 and 0 can be identity and zero matrices resp.
   T = sympy.symbols('T')
   Lifting0=sympy.Matrix([[1, T],[0,1]])
   sympy.pprint(Lifting0)
   print(sympy.latex(Lifting0)) #for LaTeX
   Lifting1=sympy.Matrix([[1,0],[-T**(-1),1]])
   sympy.pprint(Lifting1)
   print(sympy.latex(Lifting1)) 
   Lifting2=sympy.Matrix([[1, T],[0,1]])
   sympy.pprint(Lifting2)
   print(sympy.latex(Lifting2)) 
   Prod=Lifting0*Lifting1*Lifting2
   sympy.pprint(Prod)
   print(sympy.latex(Prod))
   #Matrix([
   #[   0, T],
   #[-1/T, 0]])
   
   #Number of subbands:
   N=4
   #Filter bank coefficients for sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   #fb=np.loadtxt("MDCTcoeff.txt") #Coeff. from optimization
   print("fb=", fb)
   #input test signal, 2 ramp signals:
   x=np.zeros((64,2))
   x[:,0]=np.arange(64)
   x[:,1]=np.arange(64)
   print("x=\n",np.transpose(x))
   plt.plot(x,'*')
   plt.title('Stereo Input Signal')
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.show()
   y0,y1=IntMDCTanafb(x,N,fb)
   print("y0=\n", y0)
   print("y1=\n", y1)
   plt.imshow(np.abs(y0))
   plt.title('IntMDCT Subbands Channel 0')
   plt.xlabel('Block No.')
   plt.ylabel('Subband No.')
   plt.show()
   xrek=IntMDCTsynfb(y0,y1,fb)
   print("xrek=\n",np.transpose(xrek))
   plt.plot(xrek,'*')
   plt.title('The Reconstructed Signal')
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.show()

