import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import timeit
import sys
import logging
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.sparse import diags, hstack, vstack
from scipy.linalg import solve_banded
from scipy.optimize import linprog
from PredictorCorrectorQP import InitialPoints, PredictorCorrectorQPSolver
import numpy as np
import matplotlib.pyplot as plt
#from cvxopt import matrix, solvers
#logging.basicConfig(format='%(message)s')
##Generate matrix

def GenerateProblemPrimalSVM(N):
 I = [[0.1, 0.5], [1, 1]]
 mu1 = [-5, -5]
 mu2 = [5, 5]
 d = 2
 x1 = np.random.multivariate_normal( mu1,I , N)
 x2 = np.random.multivariate_normal( mu2,I, N)
 X = np.row_stack((x1, -x2))
 epsilon = 1e-2
 K = np.identity(2*N+2)*epsilon
 ##compute Grahm-matrix
 for i in range(0,2*N):
   for j in range(0,2*N):
     K[i,j] = 0.5*np.dot(X[i,:], X[j,:])
 G = csc_matrix(K)
 Y = np.ones(2*N+2)
 Y[N:2*N+1] = Y[N:2*N+1]*-1
 A = csc_matrix(Y)
 b = np.zeros(1)
 c = np.zeros(2*N+2)
 c[0:2*N] = -1
 print "IIIIIIII"
 print G.toarray()
 print A.toarray()
 print b
 print c
 return G, A, b, c, np.row_stack((x1, x2)), np.concatenate((np.ones(N), -np.ones(N)))
n = 2
m = 2
span = 10
Maxite = 50
errorgradiente = 1e-3
mindeltafitness = 1e-100
sigma = 0.01



for k in range(1,2):
 print 'range ',n
 
 #for i in range(0,29):
 #logging.warning(i,k)
 #print >> sys.stderr, (i, k)
# A = csc_matrix(np.random.randint(1,span, size=(m, n))) + csc_matrix(hstack([identity(m), csc_matrix((m,n-m))]))
# #G = csc_matrix(np.random.randint(1,span, size=(n, n))) + csc_matrix(hstack([identity(n)]))
# G = csc_matrix(hstack([identity(n)]))
#
# #A = M#csc_matrix(sparse.hstack([ M, identity(m)]))
# #b = np.random.randint(1,span, size=(m))
# b = np.ones(m)
## c =np.concatenate((np.ones(n), np.zeros(m)))
# c = np.ones(n)
# c = np.array([1, 1])
# G = csc_matrix([[2,0.5],[0.5, 1]])
# A = csc_matrix([1,1])
# b = np.array([ 1])


# x, fitnes, ite = PredictorCorrectorQPSolver(G, A, b, c, Maxite, errorgradiente, mindeltafitness, eta=0.95)
# sol=solvers.qp(cvxopt.matrix(c), cvxopt.matrix(A),cvxopt.matrix(b))
 #sol=solvers.qp(np.asmatrix(G),np.asmatrix(c), np.asmatrix(A),np.asmatrix(b))
# print x
# print fitnes
# print ite
# #####Performance runing.....
 start = timeit.default_timer()
 N=2
 [G, A, b, c, X, Y] = GenerateProblemPrimalSVM(N)
 alpha, fitnes, ite = PredictorCorrectorQPSolver(G, A, b, c, Maxite, errorgradiente, mindeltafitness, eta=0.95)
 stop = timeit.default_timer()
 print "OOOOOO"
 print fitnes
 print alpha[0:-2].dot(Y)
 #print X
# print X*Y[:, np.newaxis]

 W = alpha[0:-1].dot( X*Y[:, np.newaxis] ) 
### print W
####find b* bias...
 B= np.average(Y - W.dot(X.T))
### print B
 W = np.append(W,B)
### x = W
### d1 = X[0:N]
### d2 = X[N:2*N]
### plt.plot(d1[:,0], d1[:,1],'.', markersize=12)
### plt.plot(d2[:,0], d2[:,1],'.',markersize=12 )
### alldata = np.concatenate((d1, d2))
### #x = W
### #values = d1[ d1.dot(x)== min(d1.dot(x))]
### #plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
### #values = d2[ d2.dot(x)== max(d2.dot(x))]
### #plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
### xmin = min( alldata[:,0])
### xmax = max( alldata[:,0])
### plt.plot( [xmin, xmax], [ (1-x[2]-xmin*x[0])/x[1] ,  (1-x[2]-xmax*x[0])/x[1] ] , linestyle='solid')
### plt.show()
 print x[0:-2].dot(x[0:-2])
