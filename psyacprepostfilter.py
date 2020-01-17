#Psycho-acoustic pre- and post-filter,
#Gerald Schuller, April 2019

import numpy as np
import os
import sys
sys.path.append('./PythonPsychoacoustics')
import psyac_quantization
import MDCTfb
import matplotlib.pyplot as plt

def psyacprefilter(x, fs, quality=100):
   #Psycho-acoustic Pre-filter,
   #Normlizes a signal to its masking threshold
   #Argument: audio signal x, quality (at masking threshold: 100),
   #sampling frequency fs
   #returns: pre-filtered audio signal xpref, 
   #masking theshold exponents for the bark subbands mTbarkquant
   
   N=128 #number of MDCT subbands
   nfilts=64  #number of subbands in the bark domain
   #Sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   
   #Analysis MDCT and normalization to the masking threshold and quantization:
   yq, y, mTbarkquant = psyac_quantization.MDCT_psayac_quant_enc(x,fs,fb,N, nfilts,quality=quality)
   #Synthesis MDCT, back to the time domain:
   xpref=MDCTfb.MDCTsynfb(yq,fb)
   return xpref, mTbarkquant
   
def psyacpostfilter(xpref, fs, mTbarkquant):
   #Psycho-acoustic post-filter,
   #De-normlizes a signal to its masking threshold
   #Argument: pre-filtered audio signal xpref,
   #Sampling frequency fs,
   #masking theshold exponents for the bark subbands mTbarkquant
   #returns: reconstructed audio signal xrek, 
   
   N=128 #number of MDCT subbands
   nfilts=64  #number of subbands in the bark domain
   #Sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   #Analysis MDCT to the time/frequency domain:
   yq=MDCTfb.MDCTanafb(xpref,N,fb); 
   yq=yq[:,1:-1]; print("yq.shape=", yq.shape) #remove first and last block, which the MDCT appended.
   #de-normalization to the masking threshold, de-quantization, and MDCT synthesis:
   xrek, mT, ydeq = psyac_quantization.MDCTsyn_dequant_dec(yq, mTbarkquant, fs, fb, N, nfilts)
   return xrek

if __name__ == '__main__':
   #Example, Demo:
   import sound
   import scipy.io.wavfile as wav 
   
   os.system('espeak -s 120 "Pre- and Post-Filter demonstration"')
   fs, x= wav.read('fantasy-orchestra.wav')
   #take left channel (left column) of stereo file to make it mono:
   x=x[:,0]
   #fs, x= wav.read('sc03_16m.wav')
   #fs, x= wav.read('test48khz.wav')
   print("Sampling Frequency=", fs, "Hz")
   plt.specgram(x, NFFT=256, Fs=6.28) #Fs needs to be a float number to avoid error message in Python3! 
   plt.title('Spectrogram of the Original Signal')
   plt.show()

   xpref, mTbarkquant = psyacprefilter(x, fs, quality=100)
   plt.plot(mTbarkquant)
   plt.title('The Masking Thresholds')
   plt.xlabel('The Bark Subbands')
   plt.show()
   
   xpref=np.round(xpref) # mid tread quantizer
   #xpref=np.floor(xpref)+0.5 #mid rise quantizer
   
   xrek = psyacpostfilter(xpref, fs, mTbarkquant)

   print("Original Signal")
   os.system('espeak -s 120 "Original Signal"')
   sound.sound(x,fs)
   print("Pre-filtered Signal")
   
   plt.plot(xpref)
   plt.xlabel('sample')
   plt.ylabel('Value')
   plt.title('The Psycho-Acoustically Prefiltered Signal')
   plt.show()
   os.system('espeak -s 120 "The amplified Pre-filtered Signal"')
   sound.sound(xpref*1000,fs)
   print("Reconstructed Signal after Quantization according to the Masking threshold")
   os.system('espeak -s 120 "Reconstructed Signal after the Postfilter"')
   sound.sound(xrek,fs)
   print("xrek.shape=", xrek.shape)
   plt.specgram(xrek, NFFT=256, Fs=6.28) #Fs needs to be a float number to avoid error message in Python3! 
   plt.title('Spectrogram of the Post-Filtered Signal')
   plt.show()
   
   

