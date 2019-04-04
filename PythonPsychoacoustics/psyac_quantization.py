#Program to implement quantization with a psycho-acoustic model
#Gerald Schuller, May 2018

import numpy as np
import scipy.signal





from psyacmodel import *
import sys
sys.path.append('../')
from MDCTfb import *

def MDCT_psayac_quant_enc(x,fs,fb,N, nfilts=64,quality=100):
   #Function to compute the quantized psycho-acoustic masking threshold
   #barguments: signal x, 
   #sampling frequency of x: fs, 
   #MDCT window fb
   #quality: Quality in percent, default: 100, higher is better quality
   #returns: yq, the signed quantization indices of the MDCT subands
   #y: the MDCT subbands,
   #mTbarkquant: The quantied masking threshold in the Bark domain
   #usage: yq, y, mTbarkquant =MDCT_psayac_quant_enc(x,fs,fb,N)
   print("quality=", quality)
   maxfreq=fs/2
   alpha=0.8  #Exponent for non-linear superposition of spreading functions
   nfft=2*N  #number of fft subbands
   
   W=mapping2barkmat(fs,nfilts,nfft)
   W_inv=mappingfrombarkmat(W,nfft)
   spreadingfunctionBarkdB=f_SP_dB(maxfreq,nfilts)
   spreadingfuncmatrix=spreadingfunctionmat(spreadingfunctionBarkdB,alpha,nfilts)

   #Quantization of MDCT subbands:

   print("Computing MDCT for sound file,")
   #Sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   #MDCT analysis filter bank:
   y=MDCTanafb(x,N,fb)
   M=y.shape[1] #number of blocks in the signal
   #y.shape : 1024,32
   f,t,ys=scipy.signal.stft(x,fs=2*np.pi,nperseg=2*N)
   ys *= np.sqrt(2*N/2)/2/0.375
   #ys.shape: 1025,33
   print("masking threshold calculation,")
   mT=np.zeros((N+1,M))
   mTbarkquant=np.zeros((nfilts,M))
   for m in range(M): #M: number of blocks
     #Observe: MDCT instead of STFT as input can be used by replacing ys by y:
     mXbark=mapping2bark(np.abs(ys[:,m]),W,nfft)
     mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts)/(quality/100)
     #Logarithmic quantization of the "scalefactors":
     mTbarkquant[:,m]=np.round(np.log2(mTbark)*4) #quantized scalefactors
     mTbarkquant[:,m]=np.clip(mTbarkquant[:,m],0,None) #dequantized is at least 1 
     #Logarithmic de-quantization of the "scalefactors" (decoder in encoder):
     mTbarkdequant=np.power(2,mTbarkquant[:,m]/4)
     #Masking threshold in the original frequency domain
     mT[:,m]=mappingfrombark(mTbarkdequant,W_inv,nfft)
   print("quantization according to the masking threshold,")
   #print("mTbark=", mTbark)
   #Quantization and scale factors:
   #
   #
   #The maximum of the magnitude of the quantization error is delta/2, we can set
   #delta=mT*2, delta is vector of length 1025 frequency bins,
   delta=mT*2 
   #delta.shape: 1025,32
   delta=delta[:-1,:] #drop the last stft band to obtain an even number of bands
   #quantization with step size delta,
   yq=np.round(y/delta) #uniform mid-tread quantization
   return yq, y, mTbarkquant
   
def MDCTsyn_dequant_dec(yq, mTbarkquant, fs, fb, N, nfilts=64):
   #Function to dequantizee scalefactors and subband values,
   #and apply the MDCT synthesis filter bank, for an audio decoder
   #Arguments: yq: quantized subband values shape: (number of MDCT subbands, number of blocks)
   #mTbarkquant: quantized scalefactors, shape: (number of Bark subbands nfilts, number of blocks)
   #fs: sampling rate, fb: MDCT prototype, N: number of MDCT subbands, 
   #nfilts: number of Bark subbands
   #returns: xrek: the reconstructed audio signal, mT: the reconstucted masking threshold
   #ydeq: the de-quantized MDCT subband values.
   
   #Logarithmic de-quantization of the masking threshold 
   #in the Bark subbands, the "scalefactors":
   mTbarkdequant=np.power(2,mTbarkquant/4)
   #Massking threshold in the original frequency domain
   nfft=2*N  #number of fft subbands
   W=mapping2barkmat(fs,nfilts,nfft)
   W_inv=mappingfrombarkmat(W,nfft)
   mT=np.transpose(mappingfrombark(np.transpose(mTbarkdequant),W_inv,nfft))
   #The quantization step-sizes
   delta=mT*2
   #delta.shape: 1025,32
   delta=delta[:-1,:] #drop the last stft band to obtain an even number of bands
   #De-quantization of the subband values:
   ydeq=yq*delta
   #print("ydeq.shape=", ydeq.shape)
   print("Inverse MDCT")
   #MDCT synthesis filter bank in a decoder:
   xrek=MDCTsynfb(ydeq,fb)
   return xrek, mT, ydeq
   
if __name__ == '__main__':
   #Example, Demo:
   import sound
   import scipy.io.wavfile as wav 
   import os
   N=1024 #number of MDCT subbands
   nfilts=64  #number of subbands in the bark domain
   #fs, x= wav.read('fantasy-orchestra.wav')
   #fs, x= wav.read('symphony-sounds.wav')
   #take left channel (left column) of stereo file to make it mono:
   #x=x[:,0]
   fs, x= wav.read('sc03_16m.wav')
   #fs, x= wav.read('test48khz.wav')
   """
   os.system('espeak -s 120 "Say something for 5 seconds"')
   fs=44100
   x=sound.record(5,fs)
   os.system('espeak -s 120 "stop"')
   """
   print("Sampling Frequency=", fs, "Hz")
   #Sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   print("Encoder part:")
   #MDCT and quantization:
   yq, y, mTbarkquant = MDCT_psayac_quant_enc(x,fs,fb,N, nfilts,quality=60)

   print("Decoder part:")
   xrek, mT, ydeq = MDCTsyn_dequant_dec(yq, mTbarkquant, fs, fb, N, nfilts)
   
   print("Original Signal")
   os.system('espeak -s 120 "Original Signal"')
   sound.sound(x,fs)
   print("Reconstructed Signal after Quantization according to the Masking threshold")
   os.system('espeak -s 120 "Reconstructed Signal after Quantization according to the Masking threshold"')
   sound.sound(xrek,fs)
   
   #print("y[3:6,10]=", y[3:6,10])
   #print("ydeq[3:6,10]=", ydeq[3:6,10])
   import matplotlib.pyplot as plt
   plt.plot(mTbarkquant) #value range: 0...75
   plt.title("The Quantization Indices of the Scalefactors")
   plt.xlabel("The Bark Subbands")
   plt.show()
   plt.plot(yq) #value range: -4...4, 
   plt.title("The Quantization Indices of the Subband values")
   plt.xlabel("The MDCT Subbands")
   plt.show()
   plt.plot(20*np.log10(np.abs(y[:,10])+1e-2))
   plt.plot(20*np.log10(mT[:-1,10]+1e-2))
   plt.title('Spectra for one Block')
   plt.plot(20*np.log10(np.abs(ydeq[:,10]-y[:,10])+1e-2))
   plt.plot(20*np.log10(np.abs(ydeq[:,10])+1e-2))
   plt.legend(('Magnitude Original Signal Spectrum','Masking Threshold', 
   'Magnitude Spectrum Reconstructed Signal Error', 'Magnitude Spectrum Reconstructed Signal'))
   plt.xlabel('MDCT subband')
   plt.ylabel("dB")
   plt.show()

   plt.specgram(x, NFFT=2048, Fs=2*np.pi)
   plt.title('Spectrogram of Original Signal')
   plt.xlabel("Block #")
   plt.ylabel("Normalized Frequency (pi is Nyquist freq.)")
   plt.figure()
   plt.specgram(xrek, NFFT=2048, Fs=2*np.pi)
   plt.title('Spectrogram of Reconstructed Signal')
   plt.xlabel("Block #")
   plt.ylabel("Normalized Frequency (pi is Nyquist freq.)")
   plt.show()
   
   plt.plot(x)
   plt.title("The reconstructed audio signal")
   plt.show()
   

