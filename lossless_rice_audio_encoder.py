#Program for a lossless audio encoder using the Integer-to-Integer MDCT
#and a rice entropy coder
#Gerald Schuller, Aug. 2018

import sys
import numpy as np
import scipy.io.wavfile as wav 
import os
import struct
#import zlib
#for installation: sudo pip install audio.coders
from audio.coders import rice
from bitstream import BitStream
from IntMDCTfb import *

if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

if (len(sys.argv) <2):
  print('\033[93m'+"Need audio file as argument!"+'\033[0m') #warning in yellow font
  
audiofile=sys.argv[1]
print("audiofile=", audiofile)

fs, x= wav.read(audiofile)
print("Sampling Frequency in Hz=", fs)

try:
  channels=x.shape[1] #number of channels, needs to be 2 for stereo (2 columns in x)
except IndexError:
  channels=1  # 1 for mono

print("channels=", channels)
if channels!=2:
   print("Wrong number of channels, need a stereo file!")

N=1024 #number of MDCT subbands

#Sine window:
fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))

#Store in a pickle binary file:
#remove extension from file name:
name,ext=os.path.splitext(audiofile)
#new extension for compressed file:
encfile=name+'.lacodrice'
print("Compressed file:", encfile)
totalbytes=0

with open(encfile, 'wb') as codedfile: #open compressed file
   pickle.dump(fs, codedfile, protocol=-1)  #write sampling rate
   print("IntMDCT:")
   y0,y1=IntMDCTanafb(x,N,fb) #compute the IntMDCT of the stereo signal
   ychan=np.stack((y0,y1),axis=0) #combine spectra into a 3-d array, 1st dim is channel
   print("y0.shape=", y0.shape)
   numblocks=y0.shape[1]
   pickle.dump(numblocks, codedfile, protocol=-1)  #write length of each channel, number of blocks
   for chan in range(channels): #loop over channels:
      print("channel ", chan)
      print("Rice Coding:")
      #Suitable Rice coding coefficient estimation for the subbands:
      #https://ipnpr.jpl.nasa.gov/progress_report/42-159/159E.pdf
      meanabs=np.mean(np.abs(ychan[chan,:,:]),axis=-1)
      ricecoefff=np.clip(np.floor(np.log2(meanabs)),0,None)
      ricecoeffc=np.clip(np.ceil(np.log2((meanabs+1)*2/3)),0,None)
      ricecoeff=np.round((ricecoeffc+ricecoefff)/2.0).astype(np.int8) #integer, 8bit
      print("ricecoeff=", ricecoeff)
      s=struct.pack('b'*int(len(ricecoeff)),*ricecoeff)
      pickle.dump(s, codedfile, protocol=-1)
      totalbytes+=2*N
      #rc=struct.pack('B'*len(ricecoeff),*ricecoeff)
      #codedfile.write(rc)
      print("log2 of mean abs values of subbands: ", ricecoeff)
      """
      import matplotlib.pyplot as plt
      plt.plot(ricecoeff)
      plt.show()
      """
      
      for k in range(N): #loop across subbands:
         if (k%100==0): print("Subband:",k)
         #m=2**b
         signedrice=rice(b=ricecoeff[k],signed=True)
         yrice= BitStream(ychan[chan,k,:].astype(np.int32), signedrice)
         #see: http://boisgera.github.io/bitstream/
         #Turn bitstream format into sequence of bytes:
         ys=yrice.read(bytes, np.floor(len(yrice)/8.0))
         totalbytes+= len(ys)
         pickle.dump(ys, codedfile, protocol=-1)  #Rice coded subband samples
         
         #decoder: 
         #yricedec = BitStream(); 
         #yricedec.write(ys)
         #ychandec=yricedec.read(signedrice,100)
         #print("ychan[chan,k,:100].astype(np.int32)", ychan[chan,k,:100].astype(np.int32))
         #print("ychandec=", ychandec)
         #print("len(yricedec)", len(yricedec))

"""
numsamples=np.prod(x.shape)
print("Total number of bytes=", totalbytes)
print("Total number of samples:", numsamples)
print("bytes per sample=", totalbytes*1.0/numsamples)
print("Hence bis per sample=", 8*1.0*totalbytes/numsamples)
"""
