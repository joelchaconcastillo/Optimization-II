import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.sparse import diags, hstack, vstack
from scipy.linalg import solve_banded
from scipy.optimize import linprog

def InitialPoints(A, b, c):
   AT = csc_matrix.transpose(A)
   AATI = inv(sparse.csc_matrix(A).dot(AT))
   x = sparse.csc_matrix(AT).dot(AATI).dot(b)
   llambda = sparse.csc_matrix(AATI).dot(A).dot(c)
   #llambda = sparse.csc_matrix(AATI).dot(b)
   s =  (c - sparse.csc_matrix(AT).dot(llambda))
   
   deltas= max(0, -(3.0/2.0)*min(s))
   s = s+deltas
   deltax= max(0, -(3.0/2.0)*min(x))
   x = x+deltax
   x = x + (1.0/2.0)*(np.dot(x, s)/(np.sum(s)))
   s = s + (1.0/2.0)*(np.dot(x, s)/(np.sum(x)))
   (m, n) = np.shape(A)
#   x = np.ones(n)
#   s = np.ones(n)
   return (x, llambda, s)

def  PrimalDual(A, b, c, ite):
   (m, n) = np.shape(A)
   (x, llambda, s) = InitialPoints(A,b,c) 
   sigma = 0.1
   for i in range(0,ite):
     alpha = np.random.uniform(0.1, 0.5, 1)
     S = diags(s)
     X = diags(x) 
     rs = csc_matrix.transpose(A).dot(llambda) + s - c
     rb = A.dot(x) - b
     #print np.linalg.norm(rs)
     if np.linalg.norm(rs) < 1e-2 and np.linalg.norm(rb) < 1e-2:
	break
     Mu = (np.dot(s, x)/n)
     XSe = np.multiply(s, x) #- (sigma*Mu)*np.ones(n)
     K1 = sparse.hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,identity(n, dtype=float) ])
     K2 = sparse.hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
     K3 = sparse.hstack([S, csc_matrix( (n,m)), X])
     K = csc_matrix(sparse.vstack([K1, K2, K3]))
     bb = np.concatenate((rs, rb, XSe))
     #xx = np.linalg.solve(K.toarray(), -bb)
     xx = spsolve(K, -bb)
     while (x + alpha*xx[0:n] <= 0.0).any() or (s + alpha*xx[n+m:2*n+m] <= 0.0).any() :     
	alpha = alpha/2.0
     x = x + alpha*xx[0:n]
     llambda = llambda + alpha*xx[n:(n+m)]
     s = s + alpha*xx[(n+m): 2*n+m]
   return x, llambda, s

def LongStepPath(A, b, c, ite, errorgrad, errorfit, sigmamin, sigmamax, gamma):
   (m, n) = np.shape(A)

   ##Se utiliza el algoritmo Primal Dual para encontrar un punto inicial en la vecindad...
   (x, llambda, s) = PrimalDual(A,b,c, 100) 
   I = identity(n, dtype=float)
   fitness = 1e100
   antnorm = 1e100
   for i in range(0,ite):
     sigma = np.random.uniform(sigmamin, sigmamax, 1)
     S = diags(s)
     X = diags(x) 
     rs = csc_matrix.transpose(A).dot(llambda) + s - c
     rb = A.dot(x) - b
     Mu = (np.dot(s, x)/n)
     XSe = np.multiply(s, x) - np.ones(n)*(sigma*Mu)
     K1 = sparse.hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,I])
     K2 = sparse.hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
     K3 = sparse.hstack([S, csc_matrix( (n,m)), X])
     K = csc_matrix(sparse.vstack([K1, K2, K3]))
     bb = np.concatenate((rs, rb, XSe))
     #xx = np.linalg.solve(K.toarray(), -bb)
     #print Mu
     xx = spsolve(K, -bb)
     alpha = 1e-100
     while (( x + alpha*xx[0:n] >= 0).all() and ( s + alpha*xx[(n+m): 2*n+m] >= 0).all() and np.linalg.norm(A.dot( x + alpha*xx[0:n] ) - b) < 1e-2) and np.linalg.norm( csc_matrix.transpose(A).dot(llambda + alpha*xx[n:(n+m)] )+ s + alpha*xx[(n+m): 2*n+m] -c) < 1e-2 and  np.multiply(x + alpha*xx[0:n],s + alpha*xx[(n+m): 2*n+m]    >=  Mu*gamma).any() :

	alpha = alpha*2.0
     alpha = alpha/2.0
     alpha = min(1.0, alpha) 


     if (A.dot( x + alpha*xx[0:n]  ) - b).dot((A.dot(x + alpha*xx[0:n]) - b)) < rb.dot(rb) and (x + alpha*xx[0:n]).dot(c) <= x.dot(c) :
      x = x + alpha*xx[0:n]
     if (csc_matrix.transpose(A).dot( llambda + alpha*xx[n:(n+m)] ) + alpha*xx[(n+m): 2*n+m] - c).dot(csc_matrix.transpose(A).dot( llambda + alpha*xx[n:(n+m)] ) + alpha*xx[(n+m): 2*n+m] - c) < rs.dot(rs):
      llambda = llambda + alpha*xx[n:(n+m)]
      s = s + alpha*xx[(n+m): 2*n+m]
      #print xx[0:n].dot(xx[0:n])
      if xx[0:n].dot(xx[0:n]) < errorgrad:
	break
#     if abs(antnorm - xx.dot(xx)) < errorgrad:
#        break
#     antnorm = xx.dot(xx)
#     if abs(fitness - c.dot(x)) < errorfit:
#        break
     fitness = c.dot(x)

   return x, np.transpose(c).dot(x), i

#A = csc_matrix( [ [3,1,1,0], [1, 1,0,1]] , dtype=float)
#c = np.array([1, -4, 0,0])
#b = np.array([6, 4])

###b = csc_matrix([[5], [8]], dtype=float)
#print "solution"
#print LongStepPath(A, b, c, 100, 1e-240, 1e-240)
###
#x0_bnds = (0, None)
#x1_bnds = (0, None)
#x2_bnds = (0, None)
#x3_bnds = (0, None)
##res = linprog(c, A_eq=A.toarray(), b_eq=b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
#res = linprog(c, A_eq=A.toarray(), b_eq=b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
##res = linprog(c, A_ub = A.toarray(),b_ub= b, bounds=(0, None))
#print(res)
