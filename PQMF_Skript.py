# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 19:36:15 2016

@author: Max
"""
import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opt
import scipy.signal as sig

from ha2Pa3d import ha2Pa3d
from hs2Ps3d import hs2Ps3d
from polmatmult import polmatmult
from optimfuncQMF import optimfuncQMF
##############################################################################
#p.80

#Anmerkung: Der Plot ist nicht wie der im Buch. optimfuncQMF liefert jedoch die richtigen Ergebnisse...
xmin = opt.minimize(optimfuncQMF,16*np.ones(16),method='SLSQP')
xmin = xmin["x"]

h = np.concatenate((xmin,np.flipud(xmin)))#?????
f,H = sig.freqz(h)

plt.plot(h)
plt.show()
plt.plot(f,20*np.log10(np.abs(H)))
plt.show() 



##############################################################################
#p.84
ha=range(31)
N=4

H=ha2Pa3d(ha,N)

G=hs2Ps3d(ha,N)

R=-polmatmult(H,G)

R[:,:,7] = np.eye(4)+R[:,:,7]

re = 0;
for m in range(15):
    re = re + np.sum(np.square(np.diag(R[:,:,m])))

# re
#20*np.log10(1.0/re)

###############################################################################

#pages 89-92
#p.89
h1 = np.sin(np.pi/2048*(np.arange(2048)+0.5))
h2 = np.sin(np.pi/256*(np.arange(256)+0.5))
hta = np.concatenate((h2[0:128], np.ones((1024-128)/2.0), h1[1024:2047]))
plt.plot(hta)

#p.91
h1 = np.sin(np.pi/2048*(np.arange(2048)+0.5))
h2 = np.sin(np.pi/256*(np.arange(256)+0.5))
hta = np.concatenate((h1[0:1024], np.ones((1024-128)/2.0), h2[128:256]))
plt.plot(hta)


