# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 18:09:15 2016

@author: Tobias

Code from p. 58 and following 
Modified: Gerald Schuller, July 2016
"""

## Start w/ actual optimization 


import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from optimfuncMDCT import optimfuncMDCT

x0 = np.random.random(6)
xmin = minimize(optimfuncMDCT, x0)
x=xmin.x
print (x)

from symFmatrix import symFmatrix
from polmatmult import polmatmult 
from Dmatrix import Dmatrix
from Fa2h import Fa2h

Fa = symFmatrix(x) 
#print("Fa=", Fa[:,:,0])
Faz = polmatmult(Fa, Dmatrix(4))
h = Fa2h(Faz)
print(h)

plt.plot(h)
plt.show()



