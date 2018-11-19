import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import timeit
import sys
import logging
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

n = 500
mycov = [[0.1, 0.0], [0.0, 0.1]] ##Covariance matrix...
mu1 = [10, 10]
mu2 = [80, 80]
d = 2
np.random.seed(2)
##Generating the data...
x1 = np.random.multivariate_normal( mu1, mycov, n)
x2 = np.random.multivariate_normal( mu2, mycov, n)
plt.plot(x1[:,0], x1[:,1],'.', markersize=12)
plt.plot(x2[:,0], x2[:,1],'.',markersize=12 )
plt.show()

data = np.vstack((x1, x2))
N, d = np.shape(data)
##Generating the table of normal distributions...
m = 10 # the number  of densities too consider
L = np.zeros((N, m))
#creating the bag density functions...
for i in range (0, N):
  for j in range(0,m):
    L[i, j] = multivariate_normal.pdf(data[i,:], mean= [j*10,j*10] , cov=mycov);
print L
n, m = np.shape(L)
X = np.ones(m)*(1.0/m)
#X = np.zeros(m)
#X[0]=1
print mixSQP(L, X)
