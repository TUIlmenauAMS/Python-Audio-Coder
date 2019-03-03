#%% LMS
import numpy as np
from sound import *
import matplotlib.pyplot as plt

x, fs = wavread('fspeech.wav')
#normalized float, -1<x<1
x = np.array(x,dtype=float)/2**15
sound(2**15*x, fs)
print("np.size(x)=", np.size(x))
e = np.zeros(np.size(x))
xrek=np.zeros(np.size(x));
P=0;
L=10
h = np.zeros(L)
#have same 0 starting values as in decoder:
x[0:L]=0.0
quantstep=0.01;
#Encoder:
for n in range(L, len(x)):
    if n> 4000 and n< 4002:
      print("encoder h: ", h)
    #prediction error and filter, using the vector of reconstructed samples:
    #predicted value from past reconstructed values:
    xrekvec=xrek[n-L+np.arange(L)]
    P=np.dot(np.flipud(xrekvec), h)
    #quantize and de-quantize e to step-size 0.05 (mid tread):
    e[n]=np.round((x[n]-P)/quantstep)*quantstep;
    #Decoder in encoder, new reconstructed value:
    xrek[n]=e[n]+P;
    #LMS update rule:
    #h = h + 1.0* e[n]*np.flipud(xrekvec)
    #NLMS update rule:
    h = h + 1.0* e[n]*np.flipud(xrekvec)/(0.1+np.dot(xrekvec,xrekvec))

print("Mean squared prediction error:", np.dot(e, e) /np.max(np.size(e)))
#with quantstep= 0.01 : 0.000244936708861
print("Compare that with the mean squared signal power:", np.dot(x.transpose(),x)/np.max(np.size(x)))
print("The Signal to Error ratio is:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
#The Signal to Error ratio is: 28.479576824 for LMS.
#The Signal to Error ratio is: 39.35867161114023 for NLMS.
#listen to it:
sound(2**15*e, fs)

plt.plot(x)
#plt.hold(True)
plt.plot(e,'r')
plt.xlabel('Sample')
plt.ylabel('Normalized Value')
plt.title('Least Mean Squares (LMS) Online Adaptation')
plt.legend(('Original','Prediction Error'))
plt.show()

# Decoder
h = np.zeros(L);
xrek = np.zeros(np.size(x));
for n in range(L, len(x)):
    if n> 4000 and n< 4002:
       print("decoder h: ", h)
    P=np.dot(np.flipud(xrek[n-L+np.arange(L)]), h)
    xrek[n] = e[n] + P 
    xrekvec=xrek[n-L+np.arange(L)]
    #LMS update:
    #h = h + 1.0 * e[n]*np.flipud(xrekvec);
    #NLMS update:
    h = h + 1.0* e[n]*np.flipud(xrekvec)/(0.1+np.dot(xrekvec,xrekvec))

plt.plot(xrek)
plt.plot(x-xrek)
plt.xlabel('Sample')
plt.ylabel('Normalized Sample')
plt.legend(('Reconstructed Signal', 'Difference Original-Decoded'))
plt.title('The Reconstructed Signal')
plt.show()

#Listen to the reconstructed signal:
sound(2**15*xrek,fs)


