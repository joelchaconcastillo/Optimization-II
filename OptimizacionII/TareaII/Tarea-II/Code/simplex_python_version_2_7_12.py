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
import time
from numpy.linalg import inv
import numpy as np
import numpy.matlib
import operator as op
#from scipy.special import comb
###Standar matrix A 
def comb(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def Simplex(A, b, C, iB, iN, XB):
   m,n = A.shape
   B = A[:,iB]
   N = A[:, iN]
   CN = C[:,iN].T
   CB = C[:,iB].T
   #Binv = B.I
   #XB = Binv*b
   XN = np.matlib.zeros(m).T
   maxite = max(comb(n, m),1000000) ## The number of iterations is bounded by the nCm
   ite = 0
   while ite < maxite :
      lam= linalg.solve(B.T, CB)
      SN = CN - N.T*lam
      if (SN>=0).all():
      #   print("Optimal point found...")
      # #  print lam
      #   print(XB)
      #   print(iB)
         return (iB,iN ,XB, A)
      q_r, q_c = np.where(SN<0)
      q = q_r[0] ## get the first negative row component
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
      ite = ite + 1


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
  C_t = np.zeros((1,ncols+nrows))
  C_t[0,ncols:ncols+nrows]=1
  #return Simplex(A, b, C, iB, iN, np.zeros((nrows,1)))
  return Simplex(A, b, C_t, iB, iN, np.absolute(b))
  
def General_Simplex(A, b, C):
   #First phase simplex
   r,c = A.shape
   (iB,iN ,XB, A) = FirstPhaseSimplex(A, b, C)
   iN = range(0, c)
   iN = list(set(iN) - set(iB))
   # 
   C = np.hstack((C, np.zeros((1, c))))
   (iB,iN ,XB, A) =Simplex(A, b, C, iB, iN, XB)
   #
   #X_sol = np.zeros(c+r)
   #X_sol[iB] = XB
  # return X_sol
   return XB


def generatorrandomMatrix(n, m):
 #A = matrix(np.random.randint(2, size=(m,n))) 
 
 x = matrix(np.random.randint(3,size=(m,1)))
 A = x*x.T
 A = np.hstack((A, 3*A[:,n-m-1]))
# np.fill_diagonal(A,1)
 b = matrix(np.random.randint(10,size=(m,1)))
 C = matrix(np.random.randint(20,size=(1,n)))
 return A, b, C

##Change B by adding q and removing p...

A = matrix([[1, 1, 1, 0], [2, 0.5, 0, 1]])
b = matrix([[5], [8]])
C = matrix([-3, -2, 0, 0])
#Diet problem...
A = matrix([[107, 500, 0], [-107, -500, 0], [72, 121, 65],[-72, -121, -65]])
b = matrix([[5000], [-50000], [20000], [-2250]])
C = matrix([0.18, 0.23, 0.05])

#A = matrix([[2, 1, 1,1,0,0],[4, 2, 3,0,1,0], [2, 5, 5, 0,0,1]])
#b = matrix([[14], [28],[30]])
#C = matrix([1, 2, -1])
##Generate random matrix constraints
##n=4
##m=n/2
##A, b, C =generatorrandomMatrix(n,m)
print General_Simplex(A, b, C)

###print A
#for i in range(1, 50):
# n = 200*i
# Total = []
# for k in range(1,30):
#  m = n/2
#  A, b, C =generatorrandomMatrix(n,m)
#  start = time.clock()
#  General_Simplex(A, b, C)
#  end = time.clock()
#  Total.append(end-start)
# print str(n)+" "+str(np.mean(Total))+" "+str(np.std(Total))
# 
#print "n =2000 y m = {1..9}" 
#for m in range(1, 10):
# n = 2000
# Total = []
# for k in range(1,30):
#  m = 200*i
#  A, b, C =generatorrandomMatrix(n,m)
#  start = time.clock()
#  General_Simplex(A, b, C)
#  end = time.clock()
#  Total.append(end-start)
# print str(n)+" "+str(np.mean(Total))+" "+str(np.std(Total))

