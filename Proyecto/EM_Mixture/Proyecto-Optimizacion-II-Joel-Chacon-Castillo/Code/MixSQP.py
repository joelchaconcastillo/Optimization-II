import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as pt
from PredictorCorrectorQP import InitialPoints, PredictorCorrectorQPSolver
from cvxopt import matrix, spmatrix
from scipy.sparse import csc_matrix
from cvxopt.solvers import options, qp
from scipy.optimize import linprog
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))
def F(L, X):
   n, m = np.shape(L) 
   return -(1.0/n)*(np.sum(np.log(np.matmul(L, X)+1e-50))+np.sum(X))
def mixSQP(L, X, Psi=0.5, Rho=0.5, maxite=100, Epsilon=1e-18):
  Method = 'PI'
  eps = 1e-6
  n, m = np.shape(L)
  Q, R, P = la.qr(L, pivoting=True)
  Lapprox = np.matmul(Q,R)[:,P]
  Pt = np.zeros(m)
  for i in range(0, maxite):
    #dt = 1.0/(np.matmul(Lapprox,X)+eps) ##means element-wise division
    dt = 1.0/(np.matmul(L,X)+eps) ##means element-wise division
    #gt = -(1.0/n)*( np.matmul(Lapprox.T,dt)) + np.ones(m)  ##compute approximate gradient
    gt = -(1.0/n)*( np.matmul(L.T,dt)) + np.ones(m)  ##compute approximate gradient
    d2t = np.diag(np.power(dt, 2))
    #Ht = (1.0/n)*( np.matmul(np.matmul(Lapprox.T, d2t),Lapprox)) + np.eye(m)*eps ##compute approximate Hessian
    Ht = (1.0/n)*( np.matmul(np.matmul(L.T, d2t),L)) + np.eye(m)*eps ##compute approximate Hessian
    #y = X + Pt
    at = -Ht.dot(X) + gt
    I = np.diag(np.ones(m))
    G = -I
    h = np.zeros(m)
    if Method == 'PI':
      y = X + Pt
      y = PredictorCorrectorQPSolver(csc_matrix(Ht),csc_matrix(np.diag(np.ones(m))),np.zeros(m),at,y, 100, Epsilon, Epsilon, 0.3)[0] #... #solve quadratic subproblem... either interior point method or active set method
    else:
      y = cvxopt_solve_qp(Ht, at, G, h ) #... #solve quadratic subproblem... either interior point method or active set method
    Pt = y-X
    if np.abs(min(gt)) < Epsilon and la.norm(Pt) < Epsilon :
	break;
    alphat = 1
    while F(L, X + alphat*Pt) > F(L, X) + alphat*Psi*np.matmul(Pt.T,gt):
    #while F(Lapprox, X + alphat*Pt) > F(Lapprox, X) + alphat*Psi*np.matmul(Pt.T,gt):
      alphat = Rho*alphat
    print F(L, X + alphat*Pt)
    X = X + alphat*Pt
    X = X/np.sum(X)
  return X
