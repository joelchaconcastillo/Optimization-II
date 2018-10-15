## mwtodos de utilidad...
###from numpy import matrix
###A = matrix( [[1,2,3],[11,12,13],[21,22,23]]) # Creates a matrix.
###x = matrix( [[1],[2],[3]] )                  # Creates a matrix (like a column vector).
###y = matrix( [[1,2,3]] )                      # Creates a matrix (like a row vector).
###print A.T                                    # Transpose of A.
###print A*x                                    # Matrix multiplication of A and x.
###print A.I                                    # Inverse of A.
###print linalg.solve(A, x)     # Solve the linear equation system.
from numpy import linalg
from numpy import matrix
from numpy.linalg import inv
import numpy as np
import numpy.matlib
###Standar matrix A 


#iB = [2, 3]
#iN = [0, 1]
#SN = matrix([ [-1, -1]])
#XN = np.matlib.zeros(2).T

def Simplex(A, b, C, iB, iN, XB):
   m,n = A.shape
   B = A[:,iB]
   N = A[:, iN]
   CN = C[:,iN].T
   CB = C[:,iB].T
   Binv = B.I
   XB = Binv*b
   XN = np.matlib.zeros(m).T
   while 1 :
      lam= linalg.solve(B.T, CB)
      SN = CN - N.T*lam
      if (SN>=0).all():
      #   print("Optimal point found...")
      # #  print lam
      #   print(XB)
      #   print(iB)
         return (iB,iN ,XB, A, C)
      q_r, q_c = np.where(SN<0)
      q = q_r[0,0] ## get the first negative row component
      t= linalg.solve(B, A[:,iN[q]])
      if (t <= 0).all():
         print("Problem is unbounded...")
         exit(0)
      t2 = t.copy()
      t2[t2<=0] = np.nan
      p = np.nanargmin(XB/t2)
      Xq = XB[p]/t[p]
      XB = XB - t*Xq
      XB[p] = Xq
      tmp_out = iB[p] 
      iB[p] = iN[q]
      iN[q] = tmp_out
   
      B = A[:,iB]
      N = A[:, iN]
      CN = C[:, iN].T
      CB = C[:, iB].T


def FirstPhaseSimplex(A, b, C):
  #reformulate the problem of the form:
  #    min e.T z, subject to Ax + Ez = b, (x,z) >= 0
  nrows, ncols = A.shape
  Z = np.eye(nrows)
  rows, cols = np.where(b<0)
  Z[rows, rows] = -1
  A = np.hstack((A,Z))
  iB = range(ncols, ncols+nrows)#np.linspace(ncols, ncols+nrows-1, nrows)
  iN = range(0,ncols)
#def Simplex(A, b, C, iB, iN):
  C = np.zeros((1,ncols+nrows))
  C[0,ncols:ncols+nrows]=1
  return Simplex(A, b, C, iB, iN, np.zeros((nrows,1)))
  


##Change B by adding q and removing p...

#A = matrix([[1, 1, 1, 0], [2, 0.5, 0, 1]])
#b = matrix([[5], [8]])
#C = matrix([-3, -2, 0, 0])
#
#A = matrix([[1, 1, 1, 1, 0, 0], [2, 1, -1, 0, -1, 0], [0, -1, 1, 0, 0, -1]])
#b = matrix([[40], [10], [10]])
#C = matrix([2, 3, 1, 0, 0, 0])
#Diet problem...
A = matrix([[106.6, 500.2, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0 ],[106.6, 500.2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ], [72.2, 121.2, 65, 0,0,0,-1,1,0,0,0,0],[72.2, 121.2, 65, 0,0,0,0,0,1,0,0,0],[1, 0, 0, 0,0,0,0,0,0,1,0,0], [0, 1, 0, 0,0,0,0,0,0,0,1,0], [0, 0, 1, 0,0,0,0,0,0,0,0,1] ])
b = matrix([[5000], [50000], [2000], [2250], [10], [10], [10]])
C = matrix([0.18, 0.23, 0.05, 0,1000,0,1000,0,0,0,0,0])
#A = matrix([[2, 1, 1, 1, 0], [1, -1, -1, 0, 1]])
#b = matrix([[2], [-1]])
#C = matrix([3, 1, 1, 0, 0])

#A = matrix([[3, 2, 1, 1, 0, 0], [2, 5, 3, 0, 1, 0], [1, 9, -1, 0, 0, -1]])
#b = matrix([[10], [15], [4]])
#C = matrix([-2, -3, -4, 0, 0, 0])

#A = matrix([[1, 1, 1,0], [2, 1, 0, 1]])
#b = matrix([[4], [5]])
#C = matrix([-3, -4, 0, 0])
#
#A = matrix([[1, 2, 1,0], [3, 2, 0, 1]])
#b = matrix([[6], [12]])
#C = matrix([-2, 1, 0, 0])

#A = matrix([[1, 2, 1,0], [3, 2, 0, 1]])
#b = matrix([[4], [3]])
#C = matrix([2, 5, 0, 0])
#A = matrix([[-1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [2, 5, 0, 0, 1]])
#b = matrix([[11], [27], [90]])
#C = matrix([4, 6, 0, 0, 0])
##Generate random matrix constraints
#n = 10
#m = n/2
#A = matrix(np.random.randint(10, size=(m,n))) 
#np.fill_diagonal(A,1)
#b = matrix(np.random.randint(40000,size=(m,1)))
#C = matrix(np.random.randint(40000,size=(1,n)))
#print A
#print b
#print C
#First phase simplex
r,c = A.shape
(iB,iN ,XB, A, C) = FirstPhaseSimplex(A, b, C)
iN = range(0, c)
iN = list(set(iN) - set(iB))
# 
(iB,iN ,XB, A, C) =Simplex(A, b, C, iB, iN, XB)
#
X_sol = np.zeros(c+r)
X_sol[iB] = XB
print XB
print X_sol
