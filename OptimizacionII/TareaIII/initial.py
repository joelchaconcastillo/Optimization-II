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
from scipy.linalg import lu
def solution(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)] 

def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def InitialPoints(A, b, c):
   (m, n) = np.shape(A)
   #Obtener las columnas que son una base de A
   U = lu(A.toarray())[2]
   lin_indep_columns = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])]
   kk = A.toarray()
   kk = kk[:,lin_indep_columns]
   AT = csc_matrix.transpose(A)
   AAT = sparse.csc_matrix(A).dot( csc_matrix.transpose(A))
   E=sparse.vstack([A,csc_matrix( (n-m,n)) ])
   print E.toarray()
  
#   eigen_vects, eigen_vals = np.linalg.eig(E.toarray())
#   print A.toarray()
#   print eigen_vals
#   print eigen_vects
#   A2 = np.atleast_2d(AAT.toarray())
#   u, s, vh = np.linalg.svd(kk.toarray())
   q, r = np.linalg.qr(kk)
   y = np.dot(q.T, b) # Let y=Q'.B using matrix multiplication
   print r
   xf = np.linalg.solve(r, y) # Solve Rx=y
#   print xf
   #Resolver el sistema para obtener una solucin del conjunto base
   xx = np.linalg.solve( kk, b)

   ##Rellenar la solucin...
   xx = np.concatenate((xx, np.zeros(n-m)))
   ##Obtener un vector del espacio nulo
   xxnull = (nullspace(E.toarray())[:,0])
   print nullspace(E.toarray())
   print xxnull
   ##Calcular una proporcin para hacer positivo los valores de X
   delta = -xx[ min(xx+xxnull)]/(xxnull[min(xx+xxnull)])
 #  print delta
   print xx
#   print xxnull
#   print xx+xxnull
#   print xx/xxnull
#   print xx+delta*xxnull
   xx =  (xx+xxnull)
   print xx
   print A.dot(xx )
   AT = csc_matrix.transpose(A)
   AAT = sparse.csc_matrix(A).dot( csc_matrix.transpose(A))
   AATI = inv(sparse.csc_matrix(A).dot(AT))
   #calcular x
  #xx = np.linalg.solve(AAT.dot(inv(AT)).toarray(), b)


   x = sparse.csc_matrix(AT).dot(AATI).dot(b)
   print "xxx"
   print x
#   x = sparse.csc_matrix(AT).dot(AATI).dot(b)
   llambda = sparse.csc_matrix(AATI).dot(A).dot(c)
#   #llambda = sparse.csc_matrix(AATI).dot(b)
   s =  (c - sparse.csc_matrix(AT).dot(llambda))
#   
#   #deltas= max(0, -(3.0/2.0)*min(s))
#   #s = s+deltas
#   #deltax= max(0, -(3.0/2.0)*min(x))
#   #x = x+deltax
#   #x = x + (1.0/2.0)*(np.dot(x, s)/(np.sum(s)))
#   #s = s + (1.0/2.0)*(np.dot(x, s)/(np.sum(x)))
#   #x = np.ones(n)
#   #s = np.ones(n)
##s = np.array([0.0, 0.0])# c - sparse.csc_matrix(AT).dot(llambda)
   print np.linalg.norm( csc_matrix.transpose(A).dot(llambda)+s-c)
   print np.linalg.norm(A.dot(xx) - b)

   return (x, llambda, s)

def LongStepPath(A, b, c, ite, errorgrad, errorfit):
   (m, n) = np.shape(A)
   (x, llambda, s) = InitialPoints(A,b,c) 
   sigma = 0.01# np.ones(n)*0.1 ##sigmaaa!!
   I = identity(n, dtype=float)
   alpha = 0.1;
   fitness = 1e100
   antnorm = 1e100
   for i in range(0,ite):
     alpha = 0.5#np.random.normal(0.9, 0.01, 1)#! 0.01##Alpha!!
     #alpha = np.random.uniform(0.0, 1.0, 1)#! 0.01##Alpha!!
     #sigma= np.random.uniform(0.0, 1.0, 1)#! 0.01##Alpha!!
     S = diags(s)
     X = diags(x) 
     rs = csc_matrix.transpose(A).dot(llambda) + s - c# C.data
     rb = A.dot(x) - b
     Mu = (np.dot(s, x)/n)
     XSe = np.multiply(s, x) - (sigma*Mu)*np.ones(n)
     K1 = sparse.hstack([csc_matrix( (n,n)) , csc_matrix.transpose(A) ,I])
     K2 = sparse.hstack([A, csc_matrix( (m,m))  , csc_matrix( (m,n)) ])
     K3 = sparse.hstack([S, csc_matrix( (n,m)), X])
     K = sparse.vstack([K1, K2, K3])
     bb = np.concatenate((rs, rb, XSe))
     xx = np.linalg.solve(K.toarray(), -bb)
      
     while (x + alpha*xx[0:n] <= 0.0).any() or (s + alpha*xx[n+m:2*n+m] <= 0.0).any() :     
	alpha = alpha/2.0
     x = x + alpha*xx[0:n]
     llambda = llambda + alpha*xx[n:(n+m)]
     s = s + alpha*xx[(n+m): 2*n+m]
     print x
     print s 

     if abs(antnorm - xx.dot(xx)) < errorgrad:
        break
     antnorm = xx.dot(xx)
     if abs(fitness - c.dot(x)) < errorfit:
        break
     fitness = c.dot(x)
     print "iterations"
     print i
   return x

#A = csc_matrix([[1, 1], [2, 0.5]], dtype=float)
A = csc_matrix( [ [-3,-1,1,0], [1, 1,0,1]] , dtype=float)
#A = csc_matrix( [ [-3, 1, 1, 0], [1, 2, 0, 1]] , dtype=float)
c = np.array([-1, -4, 0,0])
#b = np.array([6, 40])
b = np.array([6, 4])

##b = csc_matrix([[5], [8]], dtype=float)
#
print "solution"
print LongStepPath(A, b, c, 10000, 1e-10, 1e-10)
##
x0_bnds = (0, None)
x1_bnds = (0, None)
x2_bnds = (0, None)
x3_bnds = (0, None)
res = linprog(c, A_eq=A.toarray(), b_eq=b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
#res = linprog(c, A_ub = A.toarray(),b_ub= b, bounds=(0, None))
print(res)
