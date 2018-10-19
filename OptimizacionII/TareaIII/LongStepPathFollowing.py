import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from scipy.sparse import diags, hstack, vstack
n=2 #nmero de variables
m=2 #nmero de restricciones

A = csc_matrix([[1, 1], [2, 0.5]], dtype=float)
C = csc_matrix([-3, -2], dtype=float)
b = np.array([5,8])

s = np.array([5,8])
x = np.array([5,8])
llambda = np.array([9,8])
#b = csc_matrix([[5], [8]], dtype=float)

I = identity(n, dtype=float)
S = diags(s)
X = diags(x)

rs = A.dot(llambda) + s -  C.data
rb = A.dot(x) - b
XSe = np.multiply(s, x)
#XSe = 
K1 = hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,I])
K2 = hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
K3 = hstack([S, csc_matrix( (n,m)), X])
K = vstack([K1,K2,K3])
bb = np.hstack((rs, rb, XSe))
x = scipy.sparse.linalg.spsolve(K, bb)
print(x)
