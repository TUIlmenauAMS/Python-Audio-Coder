#The fast DCT4 implemented using a DCT3
#Gerald Schuller, Sep. 2017.

import scipy.fftpack as spfft
#The DCT4 transform:
def DCT4(samples):
   #Argument: 3-D array of samples, shape (y,x,# of blocks), each row correspond to 1 row 
   #to apply the DCT to.
   #Output: 3-D array where each row ist DCT4 transformed, orthonormal.
   import numpy as np
   #use a DCT3 to implement a DCT4:
   r,N,blocks=samples.shape
   samplesup=np.zeros((1,2*N,blocks))
   #upsample signal:
   samplesup[0,1::2,:]=samples
   y=spfft.dct(samplesup,type=3,axis=1,norm='ortho')*np.sqrt(2)
   return y[:,0:N,:]
   
   
