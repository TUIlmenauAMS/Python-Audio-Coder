#Program for a lossless audio decoder using lossless prediction
#and a Rice entropy coder. Rice library needs Python2.
#Gerald Schuller, Jan. 2019

from __future__ import print_function
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

def nlmslosslesspreddec(e,L,h):
   #Computes the NLMS lossless predictor
   #arguments: x: input signal (mono)
   #L: Predictor lenght
   #h: starting values for the L predictor coefficients
   #returns: e, the prediction error
   
   xrek=np.zeros(len(e))
   for n in range(L, len(e)):
      #prediction error and filter, using the vector of reconstructed samples,
      #predicted value from past reconstructed values, since it is lossless, xrek=x:
      xrekvec=xrek[n-L+np.arange(L)]
      P=np.dot(np.flipud(xrekvec), h)
      #quantize and de-quantize by rounding to the nearest integer:
      P=round(P)
      #reconstructed value from prediction error:
      xrek[n] = e[n] + P
      #NLMS update:
      h = h + 1.0* e[n]*np.flipud(xrekvec)/(0.1+np.dot(xrekvec,xrekvec))
   return xrek

if __name__ == '__main__':
   if (len(sys.argv) <2):
     print('\033[93m'+"Need *.lacodpred encoded audio file  as argument!"+'\033[0m') #warning, yellow font
     sys.exit();
   encaudiofile=sys.argv[1]
   print("encoded audiofile=", encaudiofile)
   #Store decoded audio in file:
   #remove extension from file name:
   name,ext=os.path.splitext(encaudiofile)
   print("Extension=", ext)
   if ext != '.lacodpred':
      print('\033[93m'+"Need *.lacodpred encoded audio file!"+'\033[0m') #warning, yellow font
      sys.exit();
   #new name end extension for decoded file:
   decfile=name+'larek.wav'
   print("Decoded file:", decfile)
   L=10 #Predictor order

   with open(encaudiofile, 'rb') as codedfile: #open compressed file
      fs=pickle.load(codedfile)
      N=int(fs*20e-3) #fs*20ms=640, number of samples per rice coder block
      channels=pickle.load(codedfile)
      print("fs=", fs, "channels=", channels, )
      numblocks=pickle.load(codedfile)
      #N-=4 #last encoded samples might be missing from rounding to bytes
      print("numblocks=", numblocks)
      xrek=np.zeros((numblocks*N, channels))
      for chan in range(channels): #loop over channels:
         print("channel ", chan)
         ricecoeffcomp=pickle.load(codedfile)
         ricecoeff =struct.unpack( 'B' * len(ricecoeffcomp), ricecoeffcomp);
         print("len(ricecoeff)=", len(ricecoeff))
         prederrordec=np.zeros(N*numblocks)
         for k in range(numblocks): #loop across blocks:
            if (k%100==0): print("Block number:",k)
            prederrors=pickle.load(codedfile)  #Rice coded block samples
            #m=2**b
            signedrice=rice(b=ricecoeff[k],signed=True)
            prederrorrice = BitStream(); 
            prederrorrice.write(prederrors)
            prederrordec[k*N:(k+1)*N]=prederrorrice.read(signedrice, N)
         print("NLMS prediction:")
         h=np.zeros(L)
         prederrordec=prederrordec*1.0 #convert to float to avoid overflow
         print("len(prederrordec)=", len(prederrordec))
         xrek[:len(prederrordec),chan]=nlmslosslesspreddec(prederrordec,L,h)

   print("Write decoded signal to wav file ", decfile)  
   wav.write(decfile,fs,np.int16(xrek))         
      
   
