#Psych-acoustic postfilter,
#Reads from wav files the prefitlered signal and the masking thresholds. 
#Also works for stereo and multichannel files
#Gerald Schuller, April 2019

import psyacprepostfilter

import numpy as np
import scipy.io.wavfile as wav 
import os
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 1:
  print("Usage: python3 psyacpostfilterFromFile audiopref.wav")
audiofile=sys.argv[1]
print("prefiltered audiofile=", audiofile)
fs, xpref= wav.read(audiofile)
#remove extension from file name:
name,ext=os.path.splitext(audiofile)
#remove "pref" from name:
name=name[:-4]
#new extension for compressed file:
postffile=name+'postf.wav'
maskingthresholdfile=name+'mT.wav'
print("Masking Threshold file name:", maskingthresholdfile)
fs, mTbarkquantflattenedout = wav.read(maskingthresholdfile)
try:
  channels=xpref.shape[1] #number of channels, 2 for stereo (2 columns in x)
except IndexError:
  channels=1  # 1 for mono
  xpref=np.expand_dims(xpref,axis=1) #add channels dimension 1
print("channels=", channels)
nfilts=64; N=128  #number of subbands in the bark domain and in MDCT domain
blocks=min(len(xpref[:,0])//N+1, len(mTbarkquantflattenedout[:,0])//nfilts); print("blocks=", blocks) #min number of signal blocks in the files  
for chan in range(channels): #loop over channels:
   print("channel ", chan)
   print("Compute Postfilter") 
   #subtract the 128 that was added in the prefilter to make it unsigned:
   xchan=xpref[:,chan]-128.0
   #reshape masking thresholds back into a matrix with column length nfilts:
   mTbarkquant=np.reshape(mTbarkquantflattenedout[0:nfilts*(blocks-1),chan], (nfilts,-1),order='F')
   xrek = psyacprepostfilter.psyacpostfilter(xchan[0:blocks*N], fs, mTbarkquant)
   print("xrek.shape=", xrek.shape)
   #avoid overflow in wav file by clipping:
   xrek=np.clip(xrek,-2**15,2**15-1)
   if chan==0:
      xpost=xrek
   else:
      xpost=np.vstack((xpost,xrek))
xpost=xpost.T   
print("Write to Postfiltered file:", postffile)
wav.write(postffile,fs,np.int16(xpost))  

"""
plt.plot(mTbarkquant)
plt.title('The Masking Thresholds')
plt.xlabel('The Bark Subbands')
plt.show()
plt.plot(xchan)
plt.xlabel('sample')
plt.ylabel('Value')
plt.title('The Psycho-Acoustically Prefiltered Signal')
plt.show()
"""
"""
plt.plot(xrek)
plt.xlabel('sample')
plt.ylabel('Value')
plt.title('The Psycho-Acoustically Postfiltered Signal')
plt.show()
"""
   
