import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import timeit
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.sparse import diags, hstack, vstack
from scipy.linalg import solve_banded
from scipy.optimize import linprog
from Mehrotra import InitialPoints, Mehrotra
from LongStepPathFollowing import LongStepPath
from PrimalDualPathFollowing import PrimalDual
##Generate matrix
n = 30
m = 80
span = 2
Maxite = 100
errorgradiente = 1e-4
mindeltafitness = 1e-4
sigma = 0.01

MeanitePrimal = np.zeros(30)
MeantimePrimal = np.zeros(30)
MeanfitnessPrimal = np.zeros(30)
 
MeaniteLong = np.zeros(30)
MeantimeLong = np.zeros(30)
MeanfitnessLong = np.zeros(30)

MeaniteMehrotra = np.zeros(30)
MeantimeMehrotra = np.zeros(30)
MeanfitnessMehrotra = np.zeros(30)

MeaniteSimplex = np.zeros(30)
MeantimeSimplex = np.zeros(30)
MeanfitnessSimplex = np.zeros(30)


for k in range(1,10):
 n = k*100
 print 'range ',n
 for i in range(0,29):
  A = csc_matrix(np.random.randint(0,span, size=(m, n))) + csc_matrix(hstack([identity(m), csc_matrix((m,n-m))]))
#  A = M#csc_matrix(sparse.hstack([ M, identity(m)]))
  b = np.random.randint(1,span, size=(m))
  #b = np.ones(m)
 # c =np.concatenate((np.ones(n), np.zeros(m)))
  c = np.ones(n)
  
  #####Performance runing.....
  #####Primal dual...
 start = timeit.default_timer()
 x, fitnes, ite = PrimalDual(A, b, c, Maxite, errorgradiente, mindeltafitness,alphamean=0.5, sigma=sigma)
 stop = timeit.default_timer()
 MeanitePrimal[i] = ite
 MeantimePrimal[i] = stop-start
 MeanfitnessPrimal[i] = fitnes

 start = timeit.default_timer()
 x, fitnes, ite = LongStepPath(A, b, c, Maxite, errorgradiente, mindeltafitness, sigmamin=0.001, sigmamax=0.01, gamma = 1e-3)
 stop = timeit.default_timer()
 MeaniteLong[i] = ite
 MeantimeLong[i] = stop-start
 MeanfitnessLong[i] = fitnes
  
 start = timeit.default_timer()
 x, fitnes, ite = Mehrotra(A, b, c, Maxite, errorgradiente, mindeltafitness, eta=0.9)
 stop = timeit.default_timer()
 MeaniteMehrotra[i] = ite
 MeantimeMehrotra[i] = stop-start
 MeanfitnessMehrotra[i] = fitnes

 start = timeit.default_timer()
 res = linprog(c, A_eq=A.toarray(), b_eq=b)
 ite = res.nit
 fitnes = res.fun
 stop = timeit.default_timer()
 MeaniteSimplex[i] = ite
 MeantimeSimplex[i] = stop-start
 MeanfitnessSimplex[i] = fitnes


print 'PrimalDual'
print np.mean(MeanitePrimal), ' ', np.mean(MeantimePrimal), ' ', np.mean(MeanfitnessPrimal)

print 'LongStepPath'
print np.mean(MeaniteLong), ' ', np.mean(MeantimeLong), ' ', np.mean(MeanfitnessLong)


print 'Mehrotra'
print np.mean(MeaniteMehrotra), ' ', np.mean(MeantimeMehrotra), ' ', np.mean(MeanfitnessMehrotra)


print 'Simplex'
print np.mean(MeaniteSimplex), ' ', np.mean(MeantimeSimplex), ' ', np.mean(MeanfitnessSimplex)

print 'PrimalDual'
print np.std(MeanitePrimal), ' ', np.std(MeantimePrimal), ' ', np.std(MeanfitnessPrimal)

print 'LongStepPath'
print np.std(MeaniteLong), ' ', np.std(MeantimeLong), ' ', np.std(MeanfitnessLong)


print 'Mehrotra'
print np.std(MeaniteMehrotra), ' ', np.std(MeantimeMehrotra), ' ', np.std(MeanfitnessMehrotra)


print 'Simplex'
print np.std(MeaniteSimplex), ' ', np.std(MeantimeSimplex), ' ', np.std(MeanfitnessSimplex)


