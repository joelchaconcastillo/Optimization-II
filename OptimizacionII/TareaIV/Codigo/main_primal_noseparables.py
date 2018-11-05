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
 mu1 = [10, 10]
 mu2 = [15, 15]
 d = 2
 epsilon = 1e-5
 x1 = np.random.multivariate_normal( mu1,I , N)
 x2 = np.random.multivariate_normal( mu2,I, N)
 
 G = np.identity(d+2*N+1)
 G[d:d+2*N+1, d:d+2*N+1]=1e-5
 G = csc_matrix(G)
 x1 = np.column_stack((x1, np.ones(N)))
 x2 = np.column_stack((x2, np.ones(N)))
 A = csc_matrix(np.concatenate((x1, -x2), axis=0))
 A = csc_matrix(sparse.hstack([A, np.identity(2*N)]))
 print np.shape(A.toarray())
 b = np.ones(d+2*N+1)
 print np.shape(b)
 c = np.zeros(d+2*N+1)
 c[d+1:d+1+2*N]=1
 print c
 return G, A, b, c, x1, x2
n = 2
m = 2
span = 10
Maxite = 100
errorgradiente = 1e-100
mindeltafitness = 1e-100
sigma = 0.01



for k in range(1,2):
 print 'range ',n
 
# #####Performance runing.....
 start = timeit.default_timer()
 [G, A, b, c, d1, d2] = GenerateProblemPrimalSVM(3)
 x, fitnes, ite, llambda, y = PredictorCorrectorQPSolver(G, A, b, c, Maxite, errorgradiente, mindeltafitness, eta=0.95)
 stop = timeit.default_timer()
 Y=-np.ones(200)
 Y[100:200]=1 
 print y
 print llambda.dot(y)
 print fitnes
 print x
 plt.plot(d1[:,0], d1[:,1],'.', markersize=12)
 plt.plot(d2[:,0], d2[:,1],'.',markersize=12 )
 alldata = np.concatenate((d1, d2))
 values = d1[ d1.dot(x)== min(d1.dot(x))]
 plt.plot(values[:,0], values[:,1],'.r',markersize=12 )
 values = d2[ d2.dot(x)== max(d2.dot(x))]
 plt.plot(values[:,0], values[:,1],'.r',markersize=12 )

 xmin = min( alldata[:,0])
 xmax = max( alldata[:,0])
 plt.plot( [xmin, xmax], [ (-x[2]-xmin*x[0])/x[1] ,  (-x[2]-xmax*x[0])/x[1] ] , linestyle='solid')
 plt.show()
