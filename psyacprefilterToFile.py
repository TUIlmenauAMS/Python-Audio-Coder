#Psych-acoustic prefilter,
#Reads from file and writes masking thresholds and prefiltered signal to wav files.
#Also works for stereo and multichannel files
#Gerald Schuller, April 2019

import psyacprepostfilter
import numpy as np
import scipy.io.wavfile as wav 
import os
import sys


if len(sys.argv) < 2:
  print("Usage: python3 psyacprefilterToFile audiofile.wav [quality]")
  print("default for quaity is 100, higher number give higher quality but higher bit-rate")
  
audiofile=sys.argv[1]
print("audiofile=", audiofile)
if len(sys.argv) ==3:
   quality=float(sys.argv[2])
else:
   quality=100.0
fs, x= wav.read(audiofile)
#fs, x= wav.read('test48khz.wav')

try:
  channels=x.shape[1] #number of channels, 2 for stereo (2 columns in x)
except IndexError:
  channels=1  # 1 for mono
  x=np.expand_dims(x,axis=1) #add channels dimension 1
  
print("channels=", channels)

#remove extension from file name:
name,ext=os.path.splitext(audiofile)
#new extension for compressed file:
preffile=name+'pref.wav'
print("Prefiltered file:", preffile)
maskingthresholdfile=name+'mT.wav'
print("Masking Threshold file name:", maskingthresholdfile)

print("Compute Prefilter")
for chan in range(channels): #loop over channels:
   print("channel ", chan)
   xchan=x[:,chan]
   xpref, mTbarkquant = psyacprepostfilter.psyacprefilter(xchan, fs, quality=quality)
   print("xpref.shape=", xpref.shape)
   xpref=np.round(xpref) #quantize to nearest integer
   #Convert masking thresholds to 1-D array for storing as audio file:
   mTbarkquantflattened=np.reshape(mTbarkquant, (1,-1),order='F')
   print("mTbarkquantflattened.shape", mTbarkquantflattened.shape)
   mTbarkquantflattened=mTbarkquantflattened[0,:] #remove dimension 0
   if chan==0:
      xprefout=xpref
      mTbarkquantflattenedout=mTbarkquantflattened
   else:
      xprefout=np.vstack((xprefout,xpref))
      mTbarkquantflattenedout=np.vstack((mTbarkquantflattenedout,mTbarkquantflattened))
      
xprefout=xprefout.T
mTbarkquantflattenedout=mTbarkquantflattenedout.T
wav.write(preffile,fs,np.uint8(xprefout+128))  #write prefiltered audio to 8 bit unsigned integer audio file 
#(+128 to avoid negative numbers)
wav.write(maskingthresholdfile,fs,np.uint8(mTbarkquantflattenedout))
   
   
