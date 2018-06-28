#Program to decode the encoded audio signal in *.acod
#usage: python3 audio_decoder.py audiofile.acod
#Writes the decoded audio signal in .wav format in file audiofilerek.wav
#For multichannel signals it uses separate decoding.
#Gerald Schuller, June 2018

import sys
sys.path.append('./PythonPsychoacoustics')
from psyac_quantization import *
import numpy as np
import scipy.io.wavfile as wav 
import os
from dahuffman import HuffmanCodec
if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

if len(sys.argv) < 2:
  print("Usage: python3 audio_decoder audiofile.acod")
  
encfile=sys.argv[1]
print("encoded file=",encfile)

N=1024 #number of MDCT subbands
nfilts=64  #number of subbands in the bark domain
#Sine window:
fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))

#Open the pickle binary file:
#remove extension from file name:
name,ext=os.path.splitext(encfile)
#new name end extension for decoded file:
decfile=name+'rek.wav'
print("Decoded file:", decfile)

with open(encfile, 'rb') as codedfile: #open compressed file
   fs=pickle.load(codedfile)
   channels=pickle.load(codedfile)
   print("fs=", fs, "channels=", channels, )
   
   for chan in range(channels): #loop over channels:
      print("channel ", chan)
      tablemTbarkquant=pickle.load(codedfile) #scalefactor Huffman table
      tableyq=pickle.load(codedfile)  #subband sample Huffman table
      mTbarkquantc=pickle.load(codedfile) #Huffman coded scalefactors
      yqc=pickle.load(codedfile)  #Huffman coded subband samples
      
      #Huffman decoder for the scalefactors:
      codecmTbarkquant=HuffmanCodec(code_table=tablemTbarkquant, check=False)
      #Huffman decoded scalefactors: 
      mTbarkquantflattened=codecmTbarkquant.decode(mTbarkquantc)
      #reshape them back into a matrix with column length nfilts:
      mTbarkquant=np.reshape(mTbarkquantflattened, (nfilts,-1),order='F')
      
      #Huffman decoder for the subband samples:
      codecyq=HuffmanCodec(code_table=tableyq, check=False)
      #Huffman decode the subband samples:
      yqflattened=codecyq.decode(yqc)
      #reshape them back into a matrix with column length N:
      yq=np.reshape(yqflattened, (N,-1),order='F')
      #dequantize and compute MDCT synthesis
      xrek, mT, ydeq = MDCTsyn_dequant_dec(yq, mTbarkquant, fs, fb, N, nfilts)
      if chan==0:
         x=xrek
      else:
         x=np.vstack((x,xrek))
x=np.clip(x.T,-2**15,2**15-1)         
#Write decoded signal to wav file:  
wav.write(decfile,fs,np.int16(x))

