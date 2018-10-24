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
   AATI = scipy.sparse.linalg.inv(sparse.csc_matrix(A).dot(AT))
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
   return (x, llambda, s)

def Mehrotra(A, b, c, ite, errorgrad, errorfit, eta):
   (m, n) = np.shape(A)
   (x, llambda, s) = InitialPoints(A,b,c) 
   I = identity(n, dtype=float)
   fitness = 1e100
   antnorm = 1e100
   for i in range(0,ite):
     S = diags(s)
     X = diags(x) 
     rs = csc_matrix.transpose(A).dot(llambda) + s - c
     rb = A.dot(x) - b
     Mu = (np.dot(s, x)/n)
     print i, ' ', Mu
     XSe = np.multiply(s, x)
     K1 = sparse.hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,I])
     K2 = sparse.hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
     K3 = sparse.hstack([S, csc_matrix( (n,m)), X])
     K = csc_matrix(sparse.vstack([K1, K2, K3]))
     bb = np.concatenate((rs, rb, XSe))
     deltaxaffin = spsolve(K, -bb)
     #deltaxaffin = np.linalg.solve(K.toarray(), -bb)
#     deltaxcorr = np.linalg.solve(K.toarray(), np.concatenate((np.zeros(n+m), -np.multiply(deltaxaffin[0:n],deltaxaffin[n+m:2*n+m]) )) )
     deltaxcorr = spsolve(K, np.concatenate((np.zeros(n+m), -np.multiply(deltaxaffin[0:n],deltaxaffin[n+m:2*n+m]) )) )
     delta = deltaxaffin + deltaxcorr 
     #calculo de alphas...
     deltax = delta[0:n] 
     deltalambda = delta[n:n+m] 
     deltas = delta[n+m:2*n+m] 
     
     if any(deltax<0): 
       alpha_primal_max  = min(-x[deltax<0]/deltax[ deltax < 0 ])
     if any(deltas<0):
       alpha_dual_max  = min(-s[deltas<0]/deltas[ deltas < 0 ])
     alphaprimal = min(1.0, eta*alpha_primal_max)	 
     alphadual= min(1.0, eta*alpha_dual_max)	 
     if c.dot(x) < fitness:
      x = x + alphaprimal*deltax
      llambda = llambda + alphadual*deltalambda
      s = s + alphadual*deltas
      fitness = c.dot(x)
#     if abs(antnorm - x.dot(x)) < errorgrad:
#        break
#     antnorm = x.dot(x)
#     if abs(fitness - c.dot(x)) < errorfit:
#        break
   return x, np.transpose(c).dot(x), i

##Generate matrix
#n = 1000
#m = 800
#span = 2
#tt = csc_matrix(np.random.randint(1,span, size=(m, n)))
#tt = csc_matrix.transpose(tt).dot(tt)
#M = csc_matrix(np.random.randint(0,4, size=(m, n))) + csc_matrix(hstack([identity(m), csc_matrix((m,n-m))]))
##print M
#A = csc_matrix(sparse.hstack([ M, identity(m)]))
#b = np.random.randint(1,span, size=(m))
#b = np.ones(m)
#c =np.concatenate((np.ones(n), np.zeros(m)))
#x, fitnes, ite = Mehrotra(A, b, c, 10000, 1e-3, 1e-3, 0.9)
#print ite
###
#res = linprog(c, A_eq=A.toarray(), b_eq=b)
#print(res.nit)
