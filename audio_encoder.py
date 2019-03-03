#Program to implement an audio coder, including Huffman coding and writing a binary compressed file
#as encoded version.
#Usage: python3 audio_encoder.py audiofile.wav [quality], where quality is an optional quality argument.
#Default is quality=100 as 100%. For higher values, the masking threshold is lowered accordingly, 
#to reduce the quantization error and increase the quality, but also increase bit rate.
#For multi-channel audio signals, like stereo, it encodes the channels separately. 
#It writes the compressed signal to file audiofile.acod.
#Gerald Schuller, June 2018

import sys
sys.path.append('./PythonPsychoacoustics')
from psyac_quantization import *
import numpy as np
import scipy.io.wavfile as wav 
import os
#sudo pip3 install dahuffman
from dahuffman import HuffmanCodec
if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

if len(sys.argv) < 2:
  print("Usage: python3 audio_encoder audiofile.wav [quality]")
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

N=1024 #number of MDCT subbands
nfilts=64  #number of subbands in the bark domain
#Sine window:
fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))

#Store in a pickle binary file:
#remove extension from file name:
name,ext=os.path.splitext(audiofile)
#new extension for compressed file:
encfile=name+'.acod'
print("Compressed file:", encfile)
totalbytes=0

with open(encfile, 'wb') as codedfile: #open compressed file
   pickle.dump(fs,codedfile)  #write sampling rate
   pickle.dump(channels,codedfile) #write number of channels
   
   for chan in range(channels): #loop over channels:
      print("channel ", chan)
      #Compute quantized masking threshold in the Bark domain and quantized subbands   
      yq, y, mTbarkquant=MDCT_psayac_quant_enc(x[:,chan],fs,fb,N, nfilts,quality=quality)

      print("Huffman Coding")
      #Train Huffman coder for quantized masking threshold in the Bark domain (scalefactors),
      #with flattening the masking threshold array in column (subband) order:
      mTbarkquantflattened=np.reshape(mTbarkquant, (1,-1),order='F')
      mTbarkquantflattened=mTbarkquantflattened[0] #remove dimension 0
      codecmTbarkquant=HuffmanCodec.from_data(mTbarkquantflattened)
      #Huffman table for it:
      tablemTbarkquant=codecmTbarkquant.get_code_table()
      #Huffman encoded: 
      mTbarkquantc=codecmTbarkquant.encode(mTbarkquantflattened)

      #Compute Huffman coder for the quantized subband values:
      #Train with flattened quantized subband samples
      yqflattened=np.reshape(yq,(1,-1),order='F')
      yqflattened=yqflattened[0] #remove dimension 0
      codecyq=HuffmanCodec.from_data(yqflattened)
      #Huffman table for it:
      tableyq=codecyq.get_code_table()
      #Huffman encoded:
      yqc=codecyq.encode(yqflattened)
      
      pickle.dump(tablemTbarkquant ,codedfile) #scalefactor Huffman table
      pickle.dump(tableyq ,codedfile)  #subband sample Huffman table
      pickle.dump(mTbarkquantc ,codedfile) #Huffman coded scalefactors
      pickle.dump(yqc ,codedfile)  #Huffman coded subband samples
      totalbytes+= len(tablemTbarkquant)+len(tableyq)+len(mTbarkquantc)+len(yqc)
      
numsamples=np.prod(x.shape)
print("Total number of bytes=", totalbytes)
print("Total number of samples:", numsamples)
print("bytes per sample=", totalbytes/numsamples)
print("Hence bis per sample=", 8*totalbytes/numsamples)

