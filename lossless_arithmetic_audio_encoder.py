#Program for a lossless audio coder using the Integer-to-Integer MDCT
#and a rice entropy coder
#Gerald Schuller, Aug. 2018

import sys
import numpy as np
import scipy.io.wavfile as wav 
import os
import struct
#import zlib

#To install bigfloat:
#sudo apt install libgmp3-dev; sudo apt install libmpfr-dev; 
#pip install bigfloat
#or:
#sudo -H pip3 install bigfloat; or: sudo -H python3 -m pip install bigfloat
import bigfloat 

from arithmeticCoderGaussIntBigfloatTest import *


from IntMDCTfb import *

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
encfile=name+'.lacodarith'
print("Compressed file:", encfile)
totalbytes=0

with open(encfile, 'wb') as codedfile: #open compressed file
   #write sampling rate in file:
   fsb=struct.pack('B',np.ubyte(16)) #'I' for unsigned integer
   codedfile.write(fsb)
   
   print("IntMDCT:")
   y0,y1=IntMDCTanafb(x,N,fb) #compute the IntMDCT of the stereo signal
   ychan=np.stack((y0,y1),axis=0) #combine spectra into a 3-d array, 1st dim is channel
   print("y0.shape=", y0.shape)
   numblocks=y0.shape[1]
   #write number of samples in byte stream:
   numbl=struct.pack('I',np.uint32(numblocks)) #'I' for unsigned integer
   codedfile.write(numbl)
   bytearray=[]
   #bytearray=np.ubyte(np.append(bytearray,strlen%(2**8))) #LSB 
   #bytearray=np.ubyte(np.append(bytearray, strlen*(2**(-8)))) #MSB 
   print("numblocks=", numblocks)
   bigfloat.setcontext(bigfloat.precision(30*numblocks))
   print("BigFloat precision:", 30*numblocks)
   for chan in range(channels): #loop over channels:
      print("channel ", chan)
      print("Arithmetic Coding:")
      for k in range(N): #loop across subbands:
         #if (k%100==0): 
         print("Subband:",k)

         origintarray=ychan[chan,k,:].astype(np.int32)
         print("max(np.abs(origintarray))",max(np.abs(origintarray)))
         encoded, prec, sigma, strlen = encode(origintarray)
         #print("strlen=", strlen)
         print("prec=", prec, "sigma=", sigma)
         #convert bigfloat to bytes in encoder:
         encblock=encoded
         #Write sigma in byte stream:
         sigm=struct.pack('I',np.uint32(sigma)) #'I' for unsigned integer
         codedfile.write(sigm)
         """
         bytearray=np.ubyte(np.append(bytearray,sigma%(2**8))) #LSB of sigma
         bytearray=np.ubyte(np.append(bytearray, sigma*(2**(-8)))) #MSB of sigma
         """
         
         
         #Convert prec in BigFloat to number of bytes in int:
         numbytes=int(np.ceil(prec/8.0))
         print("number of bytes in subband ",k,"and channel", chan,"=", numbytes, "for", numblocks," samples")
         print("Hence bytes per sample: ", numbytes*1.0/numblocks)
         #Convert BigFloat number which encodes many samples into a byte array:
         for bytenum in range(numbytes):
            #print("bytenum=", bytenum)
            bytearray=np.append(bytearray, np.ubyte(encblock*(bigfloat.BigFloat(2)**8))) #left shift by 8 bits, keep int part
            encblock=encblock*(bigfloat.BigFloat(2)**8)%1 #keep fractional part
            
   ba=struct.pack('B'*len(bytearray),*bytearray) #'B' for unsigned byte
   codedfile.write(ba)
   totalbytes+= len(bytearray)

numsamples=np.prod(x.shape)
print("Total number of bytes=", totalbytes)
print("Total number of samples:", numsamples)
print("bytes per sample=", totalbytes*1.0/numsamples)
print("Hence bis per sample=", 8*1.0*totalbytes/numsamples)

