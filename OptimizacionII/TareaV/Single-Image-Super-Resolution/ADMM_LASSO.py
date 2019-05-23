import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import linprog
def softthresholding(a, lambdak):
  simb = np.abs(a) - lambdak
  if np.any(simb<0):
    simb[simb<0] = 0
  return np.multiply(np.sign(a),simb)

def ADDM_LASSO(A, b, ite, lambdak):
   [m, n] = np.shape(A)
   zk = np.ones((n,1))*1.0
   vk = np.ones((n,1))*1.0
   xk = np.ones((n,1))*1.0
   muk= 1e-2
   for i in range(0, ite):
     prod1 = np.linalg.inv(np.dot(A.T,A) + muk*np.identity(n))
     prod2 = np.dot(A.T,b) + muk*(zk + vk)
     xk = np.dot(prod1, prod2)
     zk = softthresholding(xk - vk, lambdak/muk)
     vk = vk - (xk - zk)
     muk = muk+0.01
   return xk
