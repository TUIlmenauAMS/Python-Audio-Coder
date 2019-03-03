#Program for a lossless audio encoder using lossless prediction
#and a Rice entropy coder. Rice library needs Python2.
#Gerald Schuller, January 2019

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

def nlmslosslesspredenc(x,L,h):
   #Computes the NLMS lossless predictor
   #arguments: x: input signal (mono)
   #L: Predictor lenght
   #h: starting values for the L predictor coefficients
   #returns: e, the prediction error
   
   e=np.zeros(len(x))
   for n in range(L, len(x)):
      #prediction error and filter, using the vector of reconstructed samples,
      #predicted value from past reconstructed values, since it is lossless, xrek=x:
      xrekvec=x[n-L+np.arange(L)]
      P=np.dot(np.flipud(xrekvec), h)
      #quantize and de-quantize by rounding to the nearest integer:
      P=round(P)
      #prediction error:
      e[n]=x[n]-P
      #NLMS update:
      h = h + 1.0* e[n]*np.flipud(xrekvec)/(0.1+np.dot(xrekvec,xrekvec))
      #if n%100==0: print("h=", h, "P=", P, "e[n]=", e[n], "xrekvec=", xrekvec)
   return e

if __name__ == '__main__':
   if (len(sys.argv) <2):
     print('\033[93m'+"Need audio file as argument!"+'\033[0m') #warning in yellow font
     
   audiofile=sys.argv[1]
   print("audiofile=", audiofile)

   fs, x= wav.read(audiofile)
   x=x*1.0 #make it float to avoid overflow!
   print("Sampling Frequency in Hz=", fs, "max(x)=", np.max(x))

   try:
     channels=x.shape[1] #number of channels, needs to be 2 for stereo (2 columns in x)
   except IndexError:
     channels=1  # 1 for mono, make x also 2-dimensional (chan is last dim):
     x=np.expand_dims(x,axis=-1)

   print("channels=", channels, "x.shape=", x.shape)

   N=int(fs*20e-3) #fs*20ms=640, number of samples per rice coder block
   L=10  #Predictor order

   #Store in a pickle binary file:
   #remove extension from file name:
   name,ext=os.path.splitext(audiofile)
   #new extension for compressed file:
   encfile=name+'.lacodpred'
   print("Compressed file:", encfile)
   totalbytes=0

   with open(encfile, 'wb') as codedfile: #open compressed file
      pickle.dump(fs, codedfile, protocol=-1)  #write sampling rate
      pickle.dump(channels, codedfile, protocol=-1)  #write number of channels
      for chan in range(channels): #loop over channels:
         print("channel ", chan)
         print("NLMS prediction:")
         h=np.zeros(L)
         e=nlmslosslesspredenc(x[:,chan],L,h) #compute the NLMS predicton error
         print("len(e)", len(e))
         numblocks=len(e)//N
         prederror=np.reshape(e[:numblocks*N], (N,numblocks), order='F')
         print("numblocks=", numblocks)
         pickle.dump(numblocks, codedfile, protocol=-1)  #write number of blocks
         print("Rice Coding:")
         #Suitable Rice coding coefficient estimation for the blocks:
         #https://ipnpr.jpl.nasa.gov/progress_report/42-159/159E.pdf
         meanabs=np.mean(np.abs(prederror),axis=0)
         ricecoefff=np.clip(np.floor(np.log2(meanabs)),0,None)
         ricecoeffc=np.clip(np.ceil(np.log2((meanabs+1)*2/3)),0,None)
         ricecoeff=np.round((ricecoeffc+ricecoefff)/2.0).astype(np.int8) #integer, 8bit
         #print("ricecoeff=", ricecoeff)
         s=struct.pack('b'*int(len(ricecoeff)),*ricecoeff)
         pickle.dump(s, codedfile, protocol=-1)
         totalbytes+=2*numblocks
         
         prederror=np.concatenate((prederror, np.zeros((4,numblocks))), axis=0) #add 4 zeros 
         #to each block of prediction errors to possibly complete rice bytes.
         for k in range(numblocks): #loop across blocks:
            if (k%100==0): print("block:",k)
            #Rice coding with m=2**b
            signedrice=rice(b=ricecoeff[k],signed=True)
            prederrorrice= BitStream(prederror[:,k].astype(np.int32), signedrice)
            #see: http://boisgera.github.io/bitstream/
            #Turn bitstream format into sequence of bytes:
            prederrors=prederrorrice.read(bytes, np.floor(len(prederrorrice)/8.0))
            totalbytes+= len(prederrors)
            pickle.dump(prederrors, codedfile, protocol=-1)  #Rice coded block samples
            
   #"""
   numsamples=np.prod(x.shape)
   print("Total number of bytes=", totalbytes)
   print("Total number of samples:", numsamples)
   print("bytes per sample=", totalbytes*1.0/numsamples)
   print("Hence bis per sample=", 8*1.0*totalbytes/numsamples)
   #"""
