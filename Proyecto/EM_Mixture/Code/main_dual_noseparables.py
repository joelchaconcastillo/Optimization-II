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
#np.random.seed(2)
def GenerateProblemPrimalSVM(N):
 I = [[0.5, 0], [0, 0.5]]
 mu1 = [-1, -1]
 mu2 = [1, 1]
 d = 2
 np.random.seed(2)
 x1 = np.random.multivariate_normal( mu1,I , N)
 x2 = np.random.multivariate_normal( mu2,I, N)
 X = np.row_stack((x1, -x2))
 C = 0.1
 epsilon = 1e-10
 K = np.identity(2*N+1)*epsilon
 ##compute Grahm-matrix
 for i in range(0,2*N):
   for j in range(0,2*N):
     K[i,j] = 0.5*np.dot(X[i,:], X[j,:])
 G = csc_matrix(K)
#first constraint...
 Y = np.zeros(2*N+1)
 Y[0:N] = 1
 Y[N:2*N] = -1
 Y[2*N] = -1
#second constraint....
 Y2 = np.zeros(2*N+1)
 Y2[2*N]=1
#third constraint...
 I = np.identity(2*N)
# Y3 = csc_matrix(-I)
 Y3 = csc_matrix(sparse.hstack([-I, csc_matrix( (2*N,1))]))
 Y4 = csc_matrix(sparse.hstack([I, csc_matrix( (2*N,1))]))

#ensable sparse matrix...
 A = csc_matrix(sparse.vstack([Y,Y2,Y3, Y4]))

 #tmp =  np.c_[ np.identity(2*N), np.zeros(2*N) ] 
 
 #A = np.vstack((Y, tmp))
# A = np.vstack((Y, np.identity(4*N+1)))
# A = csc_matrix(A)
 b = np.zeros(2+4*N)
 b[2:2+2*N]=-C
 c = np.zeros(2*N+1)
 c[0:2*N] = -1
 return G, A, b, c, np.row_stack((x1, x2)), np.concatenate((np.ones(N), -np.ones(N)))
n = 2
m = 2
span = 10
Maxite = 1000
errorgradiente = 1e-10
mindeltafitness = 1e-100
sigma = 0.01



for k in range(1,2):
 print 'range ',n
 

# #####Performance runing.....
 start = timeit.default_timer()
 N=100
 [G, A, b, c, X, Y] = GenerateProblemPrimalSVM(N)
 alpha, fitnes, ite, y, llambda = PredictorCorrectorQPSolver(G, A, b, c, Maxite, errorgradiente, mindeltafitness, eta=0.95)
 stop = timeit.default_timer()
 print alpha
 W = alpha[0:2*N].dot( X*Y[:, np.newaxis] ) ##En alpha no se considera la variable slack..
 B = Y - W.dot(X.T) 
 B= np.mean(B)
 W = np.append(W,B) ##Se agrega el biass al vector omega...
 x = W
 d1 = X[0:N]
 d2 = X[N:2*N]
 plt.plot(d1[:,0], d1[:,1],'.', markersize=12)
 plt.plot(d2[:,0], d2[:,1],'.',markersize=12 )
 alldata = np.concatenate((d1, d2))
 values = d1[ d1.dot(W[0:-1])== min(d1.dot(W[0:-1]))]
 plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
 values = d2[ d2.dot(W[0:-1])== max(d2.dot(W[0:-1]))]
 plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
 xmin = min( alldata[:,0])
 xmax = max( alldata[:,0])
 plt.plot( [xmin, xmax], [ (-x[2]-xmin*x[0])/x[1] ,  (-x[2]-xmax*x[0])/x[1] ] , linestyle='solid')
 plt.show()
 print "Valor objetivo"
 print W[1:-1].dot(W[1:-1])*0.5
 print "Tiempo"
 print stop-start
 print "Iteraciones"
 print ite
# print X*Y[:, np.newaxis]
# W = alpha[0:-1-2*N].dot( X*Y[:, np.newaxis] ) 
# print (W).dot(X.T)
# B= np.average(-W.dot(X.T))
# W = np.append(W,B)
# x = W
# d1 = X[0:N]
# d2 = X[N:2*N]
# plt.plot(d1[:,0], d1[:,1],'.', markersize=12)
# plt.plot(d2[:,0], d2[:,1],'.',markersize=12 )
# alldata = np.concatenate((d1, d2))
## values = d1[ d1.dot(x)== min(d1.dot(x))]
## plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
## values = d2[ d2.dot(x)== max(d2.dot(x))]
## plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
# xmin = min( alldata[:,0])
# xmax = max( alldata[:,0])
# plt.plot( [xmin, xmax], [ (-x[2]-xmin*x[0])/x[1] ,  (-x[2]-xmax*x[0])/x[1] ] , linestyle='solid')
# plt.show()
# print x
# print x[0:-1].dot(x[0:-1])
