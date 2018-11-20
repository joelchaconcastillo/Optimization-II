import numpy as np
import scipy.sparse
import scipy.sparse.linalg
#import timeit
#import sys
#import logging
from scipy.stats import multivariate_normal
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
from MixSQP import mixSQP

n = 50
mycov = [[0.01, 0.0], [0.0, 0.01]] ##Covariance matrix...
mycov2 = [[0.01, 0.0], [0.0, 0.01]] ##Covariance matrix...
mu1 = [0, 10]
mu2 = [0, 90]
d = 2
#np.random.seed(20)
##Generating the data...
x1 = np.random.multivariate_normal( mu1, mycov, n*10)
x2 = np.random.multivariate_normal( mu2, mycov, n*10)
plt.plot(x1[:,0], x1[:,1],'r.', markersize=2)
plt.plot(x2[:,0], x2[:,1],'r.',markersize=2 )
#plt.show()

data = np.vstack((x1, x2))
N, d = np.shape(data)
##Generating the table of normal distributions...
m = 20 # the number  of densities too consider
L = np.zeros((N, m))
#creating the bag density functions...
minX, maxX, minY, maxY = 0., 20000., 10000., 50000.

for i in range (0, N):
  for j in range(0,m):
    L[i, j] = multivariate_normal.pdf(data[i,:], mean= [j*0,j*10] , cov=mycov);
n, m = np.shape(L)
X = np.ones(m)*(1.0/m)
#X = np.zeros(m)
#X[0]=1
Weights = mixSQP(L, X, maxite=50)
print Weights
for j in range(0,m):
   for k in range(1,10):
    if np.random.uniform(0,1) < Weights[j]:
     x1 = np.random.multivariate_normal( [j*0, j*10], mycov, N)#N*Weights[j])
     plt.plot(x1[:,0], x1[:,1],'b.', markersize=2)
plt.show()
