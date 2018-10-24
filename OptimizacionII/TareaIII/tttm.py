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

   (m, n) = np.shape(A)
   #Phase one primal part...
   xini = np.ones(n)
   A_ =  csc_matrix(sparse.hstack([A, diags(b - A.dot(xini))]))
#   A_ =  csc_matrix(sparse.hstack([A, identity(m)]))
   c_ =  np.concatenate( (np.zeros(n), np.ones(m) ))
   res = linprog(c_, A_eq=A_.toarray(), b_eq=b)#, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
   x = res.x
   print x
   x[x==0]= x[x==0] + 0.00001
   x = x[0:n]
   #Phase one dual part...
   AA1_ = csc_matrix(sparse.hstack([identity(n),csc_matrix( (n,m))]))
   AA2_ = csc_matrix(sparse.hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A)]))
#   print AA1_.toarray()
 #  print AA2_.toarray()
   AA_ = csc_matrix(sparse.vstack([AA1_, AA2_]))
   cc =  np.concatenate( (np.ones(n), np.zeros(m) ))
   print AA_.toarray()
   res = linprog(cc  , A_eq=AA_.toarray(), b_eq=np.concatenate((c,np.zeros(n))))#, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
   print res 
#   AATI = inv(sparse.csc_matrix(A).dot( csc_matrix.transpose(A)))
#   llambda = sparse.csc_matrix(AATI).dot(A).dot(c)
#   s = c - csc_matrix.transpose(A).dot(llambda)
#   s[s==0]= s[s==0] + 0.00001

   print np.linalg.norm( csc_matrix.transpose(A).dot(llambda)+s-c)
   print np.linalg.norm(A.dot(x) - b)


#   (m, n) = np.shape(A)
#   x = x[0:n]
##   x[n/2:n] =  b - A[:,0:n/2].dot(x[0:n/2])

#   (m, n) = np.shape(A)
#   A2 = sparse.vstack([A, identity(n)])
#   b2 = np.concatenate((b, np.ones(n)*0.1 ))
#
#   (m, n) = np.shape(A2)
#   #Phase one
#   xini = np.ones(n)
#   A_ =  sparse.hstack([A2, diags(b2 - A2.dot(xini))])
#   #A_ =  sparse.hstack([A2, identity(m)])
#   c_ = np.concatenate( (np.zeros(n), np.ones(m) ))
#   res = linprog(c_, A_eq=A_.toarray(), b_eq=b2)#, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
#   x = res.x
#   (m, n) = np.shape(A)
#   x = x[0:n]
#   x[n/2:n] =  b - A[:,0:n/2].dot(x[0:n/2])
#   s = 0.1/x
#
#   AT = csc_matrix.transpose(A)
#   llambda = np.linalg.solve( sparse.csc_matrix(A).dot(AT).toarray(), A.dot(c-s))
#   llambda = inv(sparse.csc_matrix(A)).dot(llambda)

#   AATI = inv(sparse.csc_matrix(A).dot( csc_matrix.transpose(A)))
#   #s = c
#   llambda = sparse.csc_matrix(AATI).dot(A).dot(c-s)

   #llambda = -sparse.csc_matrix(AATI).dot(A).dot(c)
   #s = c - csc_matrix.transpose(A[:,n/2]).dot(llambda)
   print x
   print llambda
   print s
   return (x, llambda, s)

def LongStepPath(A, b, c, ite, errorgrad, errorfit):
   (m, n) = np.shape(A)
   (x, llambda, s) = InitialPoints(A,b,c) 
   sigma = 0.001# np.ones(n)*0.1 ##sigmaaa!!
   I = identity(n, dtype=float)
   alpha = 0.01;
   fitness = 1e100
   antnorm = 1e100
   gamma = 1e-3
   for i in range(0,ite):
     S = diags(s)
     X = diags(x) 
     rs = csc_matrix.transpose(A).dot(llambda) + s - c
     rb = A.dot(x) - b
     Mu = (np.dot(s, x)/n)
     XSe = np.multiply(s, x) - np.ones(n)*(sigma*Mu)
     K1 = sparse.hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,I])
     K2 = sparse.hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
     K3 = sparse.hstack([S, csc_matrix( (n,m)), X])
     K = sparse.vstack([K1, K2, K3])
     bb = np.concatenate((rs, rb, XSe))
     xx = np.linalg.solve(K.toarray(), -bb)
 
     print np.linalg.norm( csc_matrix.transpose(A).dot(llambda)+s-c)
     print np.linalg.norm(A.dot(x) - b)
     alpha = 0.01
     print gamma*Mu
#     while ((x2 >= gamma*Mu).all() and ( s2 >= gamma*Mu).all() and np.linalg.norm(A.dot(x2) - b) < 1e-15 and np.linalg.norm( csc_matrix.transpose(A).dot(llambda2)+s2-c) < 1e-15 ) :  

#	alpha = alpha*2.0
# 	x2 = x + alpha*xx[0:n]
#     	s2 = s + alpha*xx[n+m:2*n+m]
#     	llambda2 = llambda + alpha*xx[n:(n+m)]
#     print alpha

     x = x + alpha*xx[0:n]
     llambda = llambda + alpha*xx[n:(n+m)]
     s = s + alpha*xx[(n+m): 2*n+m]
     if abs(antnorm - xx.dot(xx)) < errorgrad:
        break
     antnorm = xx.dot(xx)
     if abs(fitness - c.dot(x)) < errorfit:
        break
     fitness = c.dot(x)
     print "iterations"
     print i
   return x

A = csc_matrix( [ [3,1,1,0], [1, 1,0,1]] , dtype=float)
c = np.array([1, -4, 0,0])
b = np.array([6, 4])

##b = csc_matrix([[5], [8]], dtype=float)
#
print "solution"
print LongStepPath(A, b, c, 100, 1e-10, 1e-10)
##
x0_bnds = (0, None)
x1_bnds = (0, None)
x2_bnds = (0, None)
x3_bnds = (0, None)
#res = linprog(c, A_eq=A.toarray(), b_eq=b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
res = linprog(c, A_eq=A.toarray(), b_eq=b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
#res = linprog(c, A_ub = A.toarray(),b_ub= b, bounds=(0, None))
print(res)
