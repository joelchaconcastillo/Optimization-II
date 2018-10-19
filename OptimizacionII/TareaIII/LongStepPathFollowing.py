import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from scipy.sparse import diags, hstack, vstack
from scipy.optimize import linprog





def LongStepPath(A, c, b, s, x, llambda):
   n=3 #nmero de variables
   m=3 #nmero de restricciones
   sigma = np.ones(n)*0.01
   I = identity(n, dtype=float)
   alpha = 0.001
   for i in range(0,100):
     S = diags(s)
     X = diags(x) 
     rs = A.dot(llambda) + s - c# C.data
     rb = A.dot(x) - b
     Mu = np.ones(n)*(np.inner(s, x)/n)
     XSe = np.multiply(s, x) #- np.multiply(sigma,Mu)
     K1 = hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,I])
     K2 = hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
     K3 = hstack([S, csc_matrix( (n,m)), X])
     K = vstack([K1, K2, K3])
     bb = np.hstack((rs, rb, XSe))
   #  print bb
     xx = scipy.sparse.linalg.spsolve(K, -bb)
     x = x + alpha*xx[0:n]
     s = s + alpha*xx[n:2*n]
     llambda = llambda + 0.01*xx[2*n: 2*n+m+1]
   return x


#A = csc_matrix([[1, 1], [2, 0.5]], dtype=float)
A = csc_matrix( [ [1, 4, 8], [40,30,20], [3,2,4]] , dtype=float)


#C = csc_matrix([-3, -2], dtype=float)
c = np.array([70, 80, 85])
b = np.array([4500, 36000,2700])


s = np.array([1.0,1.0, 1.0])
x = np.array([1, 1.429, 1])
llambda = np.array([0.0,0.0, 0.0])
##b = csc_matrix([[5], [8]], dtype=float)
#
print LongStepPath(A, c, b, s, x, llambda)
#
#print(x)

res = linprog(c, A_ub = A.toarray(),b_ub= b, bounds=(0, None))
print(res)
