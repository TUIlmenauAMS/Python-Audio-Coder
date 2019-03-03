# WCLMS Example for a lossless audio coder, takes as input the non-normalized 
#16 bit integer sample values
#Gerald Schuller, February 2019
import numpy as np
from sound import *
import matplotlib.pyplot as plt

x, fs = wavread('fspeech.wav')
sound(x,fs)
#normalized float, -1<x<1
x = np.array(x,dtype=float)
P=0; #initialize predicted value
L1=30 #predictor length
L2=20
L3=10
Lw=70 #weighting computation window length
h1 = np.zeros(L1)  #initialize predictor coefficients
h2 = np.zeros(L2)
h3 = np.zeros(L3)

x=np.concatenate((np.zeros(L1),x))
print("np.size(x)=", np.size(x))
e = np.zeros(np.size(x))
e1 = np.zeros(np.size(x))
e2 = np.zeros(np.size(x))
e3 = np.zeros(np.size(x))
w1=1.0/3; w2=w1; w3=w1;
c=0.01

#Encoder:
for n in range(L1, len(x)):
    #prediction error and filter, using the vector of reconstructed samples,
    #predicted value from past reconstructed values, since it is lossless, xrek=x:
    xrekvec=x[n-L1+np.arange(L1)]
    P1=np.dot(np.flipud(xrekvec), h1)
    e1[n]=x[n]-P1
    e1vec=e1[n-L2+np.arange(L2)]
    ehat1=np.dot(np.flipud(e1vec), h2)
    e2[n]=e1[n]-ehat1
    e2vec=e2[n-L3+np.arange(L3)]
    ehat2=np.dot(np.flipud(e2vec), h3)
    e3[n]=e2[n]-ehat2
    
    P2=P1+ehat1
    P3=P2+ehat2
    #quantize and de-quantize by rounding to the nearest integer:
    P=round(w1*P1+w2*P2+w3*P3)
    #WCLMS prediction error:
    e[n]=x[n]-P
    #NLMS update:
    h1 = h1 + 1.0* e1[n]*np.flipud(xrekvec)/(0.1+np.dot(xrekvec,xrekvec))
    h2 = h2 + 1.0* e2[n]*np.flipud(e1vec)/(0.1+np.dot(e1vec,e1vec))
    h3 = h3 + 1.0* e3[n]*np.flipud(e2vec)/(0.1+np.dot(e2vec,e2vec))
    w1=np.exp(-c*np.sum(np.abs(e1[n-Lw+np.arange(Lw)]))/Lw)
    w2=np.exp(-c*np.sum(np.abs(e2[n-Lw+np.arange(Lw)]))/Lw)
    w3=np.exp(-c*np.sum(np.abs(e3[n-Lw+np.arange(Lw)]))/Lw)
    s=(w1+w2+w3)
    w1=w1/s
    w2=w2/s
    w3=w3/s
    #print("w1=", w1, "w2=", w2, "w3=", w3)
    
print("Mean squared prediction error:", np.dot(e, e) /np.max(np.size(e)))
# 218270.84050329364
print("Compare that with the mean squared signal power:", np.dot(x.transpose(),x)/np.max(np.size(x)))
# 7487930.186154128
print("The Signal to Error ratio is:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
#The Signal to Error ratio is: 37.3097818013121, better than for lossless NLMS.
#listen to it:
sound(e, fs)

plt.figure()
plt.plot(x)
plt.plot(e,'r')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Least Mean Squares (LMS) Prediction for Lossless Coding')
plt.legend(('Original','Prediction Error'))
plt.show()




