#%% LMS Example for a lossless audio coder, takes as input the non-normalized 
#16 bit integer sample values
#Gerald Schuller, January 2019
import numpy as np
from sound import *
import matplotlib.pyplot as plt

x, fs = wavread('fspeech.wav')
sound(x,fs)
#normalized float, -1<x<1
x = np.array(x,dtype=float)
P=0; #initialize predicted value
L=10 #predictor length
h = np.zeros(L)  #initialize predictor coefficients
#have same 0 starting values as in decoder:
#x[0:L]=0.0
x=np.concatenate((np.zeros(L),x))
print("np.size(x)=", np.size(x))
e = np.zeros(np.size(x))

#Encoder:
for n in range(L, len(x)):
    if n> 4000 and n< 4002:
      print("encoder h: ", h)
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

print("Mean squared prediction error:", np.dot(e, e) /np.max(np.size(e)))
# 269434.6781737087
print("Compare that with the mean squared signal power:", np.dot(x.transpose(),x)/np.max(np.size(x)))
# 7490094.202738763
print("The Signal to Error ratio is:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
#The Signal to Error ratio is: 27.79929537470223, a little less than with quant for NLMS.
#listen to it:
sound(e, fs)
hist, binedges=np.histogram(e,2**16); plt.plot((binedges[:-1]+binedges[1:])/2.0, hist*1.0/len(e)); plt.plot((binedges[:-1]+binedges[1:])/2.0, 0.01/2*np.exp(-0.01*np.abs((binedges[:-1]+binedges[1:])/2.0))); plt.title('Pred. Error Histogram')
plt.figure()
plt.plot(x)
#plt.hold(True)
plt.plot(e,'r')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Least Mean Squares (LMS) Prediction for Lossless Coding')
plt.legend(('Original','Prediction Error'))
plt.show()

# Decoder
h = np.zeros(L);
xrek = np.zeros(np.size(e));
for n in range(L, len(x)):
    if n> 4000 and n< 4002:
       print("decoder h: ", h)
    xrekvec=xrek[n-L+np.arange(L)]
    P=np.dot(np.flipud(xrekvec), h)
    P=round(P)
    xrek[n] = e[n] + P 
    #NLMS update:
    h = h + 1.0* e[n]*np.flipud(xrekvec)/(0.1+np.dot(xrekvec,xrekvec))

plt.plot(xrek)
plt.plot(x-xrek)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend(('Reconstructed Signal', 'Difference Original-Decoded'))
plt.title('The Reconstructed Signal')
plt.show()

#Listen to the reconstructed signal:
sound(xrek,fs)


