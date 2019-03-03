#Program for a lossless audio decoder using the Integer-to-Integer MDCT
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
if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle
from IntMDCTfb import *

if (len(sys.argv) <2):
  print('\033[93m'+"Need *.lacodrice encoded audio file  as argument!"+'\033[0m') #warning in yellow font
  
encaudiofile=sys.argv[1]
print("encoded audiofile=", encaudiofile)
#Store decoded audio in file:
#remove extension from file name:
name,ext=os.path.splitext(encaudiofile)
#new name end extension for decoded file:
decfile=name+'larek.wav'
print("Decoded file:", decfile)

N=1024 #number of MDCT subbands
#Sine window:
fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))

with open(encaudiofile, 'rb') as codedfile: #open compressed file
   fs=pickle.load(codedfile)
   channels=2
   print("fs=", fs, "channels=", channels, )
   numblocks=pickle.load(codedfile)
   numblocks-=4 #last encoded samples might be missing from rounding to bytes
   print("numblocks=", numblocks)
   for chan in range(channels): #loop over channels:
      print("channel ", chan)
      ricecoeffcomp=pickle.load(codedfile)
      ricecoeff =struct.unpack( 'B' * len(ricecoeffcomp), ricecoeffcomp);
      #print("ricecoeff=", ricecoeff)
      ychandec=np.zeros((N,numblocks))
      
      for k in range(N): #loop across subbands:
         if (k%100==0): print("Subband:",k)
         ys=pickle.load(codedfile)  #Rice coded subband samples
         #m=2**b
         signedrice=rice(b=ricecoeff[k],signed=True)
         yricedec = BitStream(); 
         yricedec.write(ys)
         ychandec[k,:]=yricedec.read(signedrice, numblocks)
      if chan==0: y0=ychandec
      if chan==1: y1=ychandec
         
print("Inverse IntMDCT:")
xrek=IntMDCTsynfb(y0,y1,fb)
xrek=np.clip(xrek,-2**15,2**15-1) 
print("Write decoded signal to wav file ", decfile)  
wav.write(decfile,fs,np.int16(xrek))         
      
