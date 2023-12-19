#Psycho-acoustic threshold function
#Gerald Schuller, September 2023
import sys
import scipy.signal
sys.path.append('./PythonPsychoacoustics')
from psyacmodel import *


def psyacthresh(ys,fs):
  #input: ys: 2d array of sound STFT (from a mono signal, shape N+1,M)
  #fs: sampling frequency in samples per second
  #returns: mT, the masking threshold in N+1 subbands for the M blocks (shape N+1,M)

  maxfreq=fs/2
  alpha=0.8  #Exponent for non-linear superposition of spreading functions
  nfilts=64  #number of subbands in the bark domain
  #M=len(snd)//nfft
  M=ys.shape[1]
  #N=nfft//2
  N=ys.shape[0]-1
  nfft=2*N

  W=mapping2barkmat(fs,nfilts,nfft)
  W_inv=mappingfrombarkmat(W,nfft)
  spreadingfunctionBarkdB=f_SP_dB(maxfreq,nfilts)
  #maxbark=hz2bark(maxfreq)
  #bark=np.linspace(0,maxbark,nfilts)
  spreadingfuncmatrix=spreadingfunctionmat(spreadingfunctionBarkdB,alpha, nfilts)
  #Computing the masking threshold in each block of nfft samples:
  mT=np.zeros((N+1,M))
  for m in range(M): #M: number of blocks
    #mX=np.abs(np.fft.fft(snd[m*nfft+np.arange(2048)],norm='ortho'))[0:1025]
    mX=np.abs(ys[:,m])
    mXbark=mapping2bark(mX,W,nfft)
    #Compute the masking threshold in the Bark domain:
    mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts)
    #Massking threshold in the original frequency domain
    mT[:,m]=mappingfrombark(mTbark,W_inv,nfft)

  return mT #the masking threshold in N+1 subbands for the M blocks
  
def percloss(orig, modified, fs):
  #computes the perceptually weighted distance between the original (orig) and modified audio signals,
  #with sampling rate fs. The psycho-acoustic threshold is computed from orig, hence it is not commutative.
  #returns: ploss, the perceptual loss value, the mean squarred difference of the two spectra, normalized to the masking threshold of the orig.
  #Gerald Schuller, September 2023

  nfft=2048  #number of fft subbands
  N=nfft//2

  #print("orig.shape=", orig.shape)
  f,t,origys=scipy.signal.stft(orig,fs=2*np.pi,nperseg=2*N, axis=0)
  #origsys.shape= freq.bin, channel, block
  if len(orig.shape)==2: #multichannel
    chan=orig.shape[1]
    for c in range(chan):
      if c==0: #initialize masking threshold tensor mT
        mT0=psyacthresh(origys[:,c,:],fs)
        rows, cols=mT0.shape
        mT=np.zeros((rows,chan,cols))
        mT[:,0,:]=mT0
      else:
        mT[:,c,:]=psyacthresh(origys[:,c,:],fs)
  else:
    chan=1
    mT=psyacthresh(origys,fs)
  """
  plt.plot(20*np.log10(np.abs(origys[:,0,400])+1e-6))
  plt.plot(20*np.log10(mT[:,0,400]+1e-6))
  plt.legend(('Original spectrum','Masking threshold'))
  plt.title("Spectrum over bins")
  """
  #print("origys.shape=",origys.shape, "mT.shape=",mT.shape)

  f,t,modifiedys=scipy.signal.stft(modified,fs=2*np.pi,nperseg=2*N, axis=0)

  #normalized diff. spectrum:
  normdiffspec=abs((origys-modifiedys)/mT)
  #Plot difference spectrum, normalized to masking threshold:
  """
  plt.plot(20*np.log10(normdiffspec[:,0,400])+1e-6)
  plt.title("normalized diff. spectrum")
  plt.show()
  """
  ploss=np.mean(normdiffspec**2)
  return ploss
  
if __name__ == '__main__': #testing
   import scipy.io.wavfile as wav
   import scipy.signal
   import numpy as np
   import matplotlib.pyplot as plt

   fs, snd =wav.read('fantasy-orchestra.wav')
   plt.plot(snd[:,0])
   plt.title("The original sound")
   plt.show()
   
   
   nfft=2048  #number of fft subbands
   N=nfft//2

   print("snd.shape=", snd.shape)
   f,t,ys=scipy.signal.stft(snd[:,0],fs=2*np.pi,nperseg=2*N)
   #scaling for the application of the
   #resulting masking threshold to MDCT subbands:
   ys *= np.sqrt(2*N/2)/2/0.375

   print("fs=", fs)
   mT=psyacthresh(ys,fs)

   print("mT.shape=",mT.shape)
   plt.plot(20*np.log10(np.abs(ys[:,400])+1e-6))
   plt.plot(20*np.log10(mT[:,400]+1e-6))
   plt.legend(('Original spectrum','Masking threshold'))
   plt.title("Spectrum over bins")

   plt.figure()
   plt.imshow(20*np.log10(np.abs(ys)+1e-6))
   plt.title("Spectrogram of Original")
   plt.show()
   
   #Audio signal with uniform quantization and de-quantization
   snd_quant=(np.round(snd/10000))*10000
   
   ploss=percloss(snd, snd_quant, fs)
   print("psyco-acoustic loss=", ploss)
   
