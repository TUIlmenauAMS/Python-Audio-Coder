#Folding matrix F implemented with 3 lifting steps, including rounding
#for the IntMDCT
#Gerald Schuller, Sep. 2018

import numpy as np

def LiftingFmat(fb):
   #produces the 3 lifting matrices F0, L0, L1, 
   #whose product results in the F folding matrix for the IntMDCT.
   #Usage: F0,L0,L1=LiftingFmat(fb) 
   #Argument fb: The 1.5N coefficients for the normal F matrix of the MDCT

   N=int(len(fb)/1.5)
   alpha=np.arcsin(fb[:N//2])
   #print("alpha=", alpha)
   #Lifting1:
   F0=np.zeros((N,N))
   #third quadrant anti-diagonal:
   F0[int(N/2):N,0:int(N/2)] =(np.eye(int(N/2)))
   #2nd quadrant anti-diagonal:
   F0[0:int(N/2),int(N/2):N] =(np.eye(int(N/2)))
   #4th quadrant:
   F0[int(N/2):N,int(N/2):N]=np.flipud(np.diag((np.cos(alpha)-1)/np.sin(alpha),k=0))
   
   #Lifting2:
   L0=np.zeros((N,N))
   #2nd quadrant anti-diagonal:
   L0[0:int(N/2),int(N/2):N] =np.fliplr(np.eye(int(N/2)))
   #3rd quadrant anti-diagonal:
   L0[int(N/2):N,0:int(N/2)] =np.fliplr(np.eye(int(N/2)))
   #4th quadrant:
   L0[int(N/2):N,int(N/2):N]=(np.diag(np.sin(alpha),k=0))
   
   #Lifting3:
   L1=np.zeros((N,N))
   #third quadrant anti-diagonal:
   L1[int(N/2):N,0:int(N/2)] =np.fliplr(np.eye(int(N/2)))
   #2nd quadrant anti-diagonal:
   L1[0:int(N/2),int(N/2):N] =np.fliplr(np.eye(int(N/2)))
   #4th quadrant:
   L1[int(N/2):N,int(N/2):N]=np.diag((np.cos(alpha)-1)/np.sin(alpha),k=0)
   return F0, L0, L1
   
#Testing:
if __name__ == '__main__':
   #Rotation
   #np.matrix([[cos(alpha), -sin(alpha)],[sin(alpha), cos(alpha)]])
   #Fmatrix: Rotation matrix with rows flipped:
   #np.matrix([[sin(alpha), cos(alpha)],[cos(alpha), -sin(alpha)]])
   #Lifting implementation: rows of first matrix F0 flipped, L1 can also be flipped 
   #when L0 has its columns flipped. In this way we get all 1's on the anti-diagonal,
   #like for the zero-delay matrices.
   #For N=2:
   """
   alpha=0.1
   F0=np.matrix([[0,1],[1,(np.cos(alpha)-1)/np.sin(alpha)]])
   L0=np.matrix([[0,1],[1,np.sin(alpha)]]) 
   L1=np.matrix([[0,1],[1,(np.cos(alpha)-1)/np.sin(alpha)]])
   Fmatrix=F0*L0*L1
   print("N=2: flipped rotation matrix from lifting=\n", Fmatrix)
   print("As comparison, flipped rotation matrix:\n", np.matrix([[np.sin(alpha), np.cos(alpha)],[np.cos(alpha), -np.sin(alpha)]]))
   """
   #With sympy:
   import sympy 
   alpha=sympy.symbols('alpha') 
   F0=sympy.Matrix([[0,1],[1,(sympy.cos(alpha)-1)/sympy.sin(alpha)]])
   L0=sympy.Matrix([[0,1],[1,sympy.sin(alpha)]])                     
   L1=F0  
   print("Using sympy: flipped lifting steps:")
   #print(sympy.latex(F0)) #for LaTeX
   sympy.pprint(F0)
   sympy.pprint(L0)
   sympy.pprint(L1)
   print("Their product is a flipped rotation matrix:")
   sympy.pprint(sympy.simplify(F0*L0*L1))  
   
   N=4 #number of subbands
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   
   F0,L0,L1=LiftingFmat(fb)
   print("Lifting matrices for N=4:")
   print("F0=\n", F0)
   print("L0=\n", L0)
   print("L1=\n", L1)
   
   Fmat=np.dot(F0,L0)
   Fmat=np.dot(Fmat,L1)
   print("N=4: Fmat from lifting=\n", Fmat)
   
   #comparison to the usual symmetric F matrix:
   from MDCTfb import symFmatrix
   Fa=symFmatrix(fb)
   print("for comparison: usual Fa= \n", Fa[:,:,0])
   #The inverse is the transpose.
   #Hence lifting steps then are in reverse order and each transposed.
   
   
   
