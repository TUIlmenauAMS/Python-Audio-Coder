import numpy as np
from sound import *
import matplotlib.pyplot as plt
import scipy.signal as sp

x, fs = wavread('fspeech.wav');
#convert to float array type, normalize to -1<x<1:
x = np.array(x,dtype=float)/2**15
print("np.size(x)=",np.size(x))
sound(2**15*x,fs)

L=10 #predictor lenth
len0 = np.max(np.size(x))
e = np.zeros(np.size(x)) #prediction error variable initialization
blocks = np.int(np.floor(len0/640)) #total number of blocks
state = np.zeros(L) #Memory state of prediction filter
#Building our Matrix A from blocks of length 640 samples and process:
h=np.zeros((blocks,L)) #initialize pred. coeff memory

for m in range(0,blocks):
    A = np.zeros((640-L,L)) #trick: up to 630 to avoid zeros in the matrix
    for n in range(0,640-L):
        A[n,:] = np.flipud(x[m*640+n+np.arange(L)])

    #Construct our desired target signal d, one sample into the future:
    d=x[m*640+np.arange(L,640)];
    #Compute the prediction filter:
    h[m,:] = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)), A.transpose()), d)
    hperr = np.hstack([1, -h[m,:]])
    e[m*640+np.arange(0,640)], state = sp.lfilter(hperr,[1],x[m*640+np.arange(0,640)], zi=state)
    
    
#The mean-squared error now is:
print("The average squared error is:", np.dot(e.transpose(),e)/np.max(np.size(e)))
#The average squared error is: 0.000113347859337
#We can see that this is only about 1 / 4 of the previous pred. Error!
print("Compare that with the mean squared signal power:", np.dot(x.transpose(),x)/np.max(np.size(x)))
#0.00697569381701
print("The Signal to Error ratio is:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
#61.5423516403
#So our LPC pred err energy is more than a factor of 61 smaller than the 
#signal energy!
#Listen to the prediction error:
sound(2**15*e,fs)
#Take a look at the signal and it's prediction error:
plt.figure()
plt.plot(x)
#plt.hold(True)
plt.plot(e,'r')
plt.xlabel('Sample')
plt.ylabel('Normalized Value')
plt.legend(('Original','Prediction Error'))
plt.title('LPC Coding')
plt.show()

#Decoder:
xrek=np.zeros(x.shape) #initialize reconstructed signal memory
state = np.zeros(L) #Initialize Memory state of prediction filter
for m in range(0,blocks):
    hperr = np.hstack([1, -h[m,:]])
    #predictive reconstruction filter: hperr from numerator to denominator:
    xrek[m*640+np.arange(0,640)] , state = sp.lfilter([1], hperr,e[m*640+np.arange(0,640)], zi=state)

plt.plot(xrek)
plt.plot(x-xrek)
plt.xlabel('Sample')
plt.ylabel('Normalized Sample')
plt.legend(('Reconstructed Signal', 'Difference Original-Decoded'))
plt.title('The Reconstructed Signal for LPC')
plt.show()
#Listen to the reconstructed signal:
sound(2**15*xrek,fs)

