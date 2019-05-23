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
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
from MixSQP import mixSQP

n = 100
d = 0.2
mycov = [[1, 0.9], [0.9, 15]] ##Covariance matrix...
mycov2 = [[d, 0.1], [0.1, d]] ##Covariance matrix...
mu1 = [5, 5]
mu2 = [89, 89]
mu3 = [50, 8]
#np.random.seed(20)
##Generating the data...
x1 = np.random.multivariate_normal( mu1, mycov, n)
x2 = np.random.multivariate_normal( mu2, mycov, n)
mycov = [[10, 5.9], [5.9, 6]] ##Covariance matrix...
x3 = np.random.multivariate_normal( mu3, mycov, n)
plt.plot(x1[:,0], x1[:,1],'r.', markersize=1)
plt.plot(x2[:,0], x2[:,1],'r.',markersize=1 )
#plt.plot(x3[:,0], x3[:,1],'r.',markersize=1 )
#plt.show()

data = np.vstack((x1, x2))
#data = np.vstack((x1, x2, x3))
N, d = np.shape(data)
##Generating the table of normal distributions...
#creating the bag density functions...
minX, maxX, minY, maxY = 0., 100., 0., 100.
m = 300#np.shape(x)[0]# 20 # the number  of densities too consider
x = np.linspace(minX, maxX, np.sqrt(m))#, (maxX-minX).+1)
y = np.linspace(minY, maxY, np.sqrt(m))#, (maxY-minY)/200.+1)
#print x
L = np.zeros((N, m))
sqrtm = int(np.sqrt(m))
for i in range (0, N):
  for j in range(0,sqrtm):
   for k in range(0,sqrtm):
    L[i, j*sqrtm + k] = multivariate_normal.pdf(data[i,:], mean= [x[j], y[k]] , cov=mycov2);

n, m = np.shape(L)
#print L
X = np.ones(m)*(1.0/m)
X = np.zeros(m)#*(1.0/m)
Weights = mixSQP(L, X, maxite=50)
#print x[Weights>0.01]
#print Weights[Weights>0.01]
print Weights
#
for j in range(0, sqrtm):
   for k in range(0, sqrtm):
    for h in range(1,10):
     if np.random.uniform(0,1) < Weights[j*sqrtm+k]:
      x1 = np.random.multivariate_normal( [x[j], y[k]], mycov2, int(Weights[j*sqrtm+k]*N))
      plt.plot(x1[:,0], x1[:,1],'b.', markersize=2)
plt.show()
