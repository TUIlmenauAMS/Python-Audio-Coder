# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 16:59:32 2016
from Matlab
@author: Max, Gerald Schuller
"""


import numpy as np
def polmatmult( A,B ):
   """polmatmult(A,B)
   multiplies two polynomial matrices (arrays) A and B, where each matrix entry is a polynomial.
   Those polynomial entries are in the 3rd dimension
   The thirs dimension can also be interpreted as containing the (2D) coefficient
   exponent of z^-1.
   Result is C=A*B;"""  
   [NAx, NAy, NAz] = np.shape(A);
   [NBx, NBy, NBz] = np.shape(B);
   #Degree +1 of resulting polynomial, with NAz-1 and NBz-1 being the degree of the...
   Deg = NAz + NBz -1;
   C = np.zeros((NAx,NBy,Deg));
   #Convolution of matrices:
   for n in range(0,(Deg)):
       for m in range(0,n+1):
           if ((n-m)<NAz and m<NBz):
               C[:,:,n] = C[:,:,n]+ np.dot(A[:,:,(n-m)],B[:,:,m]);
   return C
   



    
