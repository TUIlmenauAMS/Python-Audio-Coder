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

from optimfuncLDFB import optimfuncLDFB

x0 = np.random.random(8)
xmin = minimize(optimfuncLDFB, x0)
x=xmin.x
print (x)



from symFmatrix import symFmatrix
from polmatmult import polmatmult 
from Dmatrix import Dmatrix
from Fa2h import Fa2h
from Gmatrix import Gmatrix

N=4
Fa = symFmatrix(x[0:(3*N/2)])
#print("Fa=", Fa[:,:,0])
Faz = polmatmult(Fa,Dmatrix(4))
Faz = polmatmult(Faz,Gmatrix(x[3*N/2:(2*N)]))
#baseband prototype function h:
h = Fa2h(Faz)

print(h)

plt.plot(h)
plt.show()




