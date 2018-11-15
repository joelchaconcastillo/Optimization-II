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
 np.random.seed(2)
 I = [[0.1, 0.5], [1, 1]]
 mu1 = [10, 10]
 mu2 = [15, 15]
 I = [[0.1, 0.5], [1, 1]]
 mu1 = [-5, -5]
 mu2 = [5, 5]
 d = 2
 x1 = np.random.multivariate_normal( mu1,I , N)
 x2 = np.random.multivariate_normal( mu2,I, N)
 G = np.identity(d+1)
 G[d, d]=1e-10
# G = G+np.identity(d+1)
 G = csc_matrix(G)
 x1 = np.column_stack((x1, np.ones(N)))
 x2 = np.column_stack((x2, np.ones(N)))
 A = csc_matrix(np.concatenate((x1, -x2), axis=0))
 b = np.ones(2*N)
 c = np.zeros(d+1)
 return G, A, b, c, x1, x2
n = 2
m = 2
span = 10
Maxite = 100
errorgradiente = 1e-10
mindeltafitness = 1e-100
sigma = 0.01



for k in range(1,2):
 print 'range ',n
 
 #for i in range(0,29):
 #logging.warning(i,k)
 #print >> sys.stderr, (i, k)
 A = csc_matrix(np.random.randint(1,span, size=(m, n))) + csc_matrix(hstack([identity(m), csc_matrix((m,n-m))]))
 #G = csc_matrix(np.random.randint(1,span, size=(n, n))) + csc_matrix(hstack([identity(n)]))
 G = csc_matrix(hstack([identity(n)]))

# #####Performance runing.....
 start = timeit.default_timer()
 N = 100
 [G, A, b, c, d1, d2] = GenerateProblemPrimalSVM(N)
 x, fitnes, ite, llambda, y = PredictorCorrectorQPSolver(G, A, b, c, Maxite, errorgradiente, mindeltafitness, eta=0.95)
 stop = timeit.default_timer()
 Y=-np.ones(N)
 Y[N:2*N]=1 
 
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
 print "Iteraciones"
 print ite
 print "Tiempo"
 print stop-start
 print "Valor objetivo"
 print fitnes
