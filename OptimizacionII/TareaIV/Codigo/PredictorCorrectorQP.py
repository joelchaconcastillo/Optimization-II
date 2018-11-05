import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.sparse import diags, hstack, vstack
from scipy.linalg import solve_banded
from scipy.optimize import linprog



def InitialPoints(G, A, b, c):
     (m, n) = np.shape(A)
     x = np.ones(n)
     y = np.ones(m)
     llambda = np.zeros(m)
     I = identity(m, dtype=float)
     Y = diags(y)
     La = diags(llambda) 
     rd = G.dot(x) - csc_matrix.transpose(A).dot(llambda) + c
     rp = A.dot(x) - y - b
     LaYe = np.multiply(llambda, y)
     K1 = sparse.hstack([G, csc_matrix( (n,m)) , -csc_matrix.transpose(A)])
     K2 = sparse.hstack([A, -I  , csc_matrix( (m,m)) ])
     K3 = sparse.hstack([csc_matrix( (m,n)), La, Y])
     K = csc_matrix(sparse.vstack([K1, K2, K3]))
     bb = np.concatenate((-rd,-rp, -LaYe))
     deltaAffin = spsolve(K, bb)
     deltaxAffin = deltaAffin[0:n] 
     deltayAffin = deltaAffin[n:n+m]
     deltalambdaAffin = deltaAffin[n+m:2*m+n] 
     y = np.maximum(np.ones(m),np.abs(y+deltayAffin)) 
     llambda = np.maximum(np.ones(m),np.abs(llambda+deltalambdaAffin)) 
     return (x, llambda, y)

def PredictorCorrectorQPSolver(G, A, b, c, ite, errorgrad, errorfit, eta):
   (m, n) = np.shape(A)
   (x, llambda, y) = InitialPoints(G,A,b,c) 
   I = identity(m, dtype=float)
   fitness = 1e100
   antnorm = 1e100
   for i in range(0,ite):
     ###1
     Y = diags(y)
     La = diags(llambda) 
     rd = G.dot(x) - csc_matrix.transpose(A).dot(llambda) + c
     rp = A.dot(x) - y - b
     LaYe = np.multiply(llambda, y)
     K1 = sparse.hstack([G, csc_matrix( (n,m)) , -csc_matrix.transpose(A)])
     K2 = sparse.hstack([A, -I  , csc_matrix( (m,m)) ])
     K3 = sparse.hstack([csc_matrix( (m,n)), La, Y])
     K = csc_matrix(sparse.vstack([K1, K2, K3]))
     bb = np.concatenate((-rd,-rp, -LaYe))
     deltaAffin = spsolve(K, bb)
     deltaxAffin = deltaAffin[0:n] 
     deltayAffin = deltaAffin[n:n+m]
     deltalambdaAffin = deltaAffin[n+m:2*m+n] 
     ###2
     Mu = (np.dot(y, llambda)/m)
     ###3
     alphay_affin = 1.0
     alphalambda_affin = 1.0
     if any(deltayAffin<0): 
       alphay_affin  = min(-y[deltayAffin<0]/deltayAffin[ deltayAffin < 0 ])
     if any(deltalambdaAffin<0): 
       alphalambda_affin  = min(-llambda[deltalambdaAffin<0]/deltalambdaAffin[ deltalambdaAffin < 0 ])
     
     alphay_affin = min(1.0, alphay_affin)
     alphalambda_affin = min(1.0, alphalambda_affin)
     alpha = min(alphay_affin, alphalambda_affin)
     ###4
     Muaff = (np.dot(y+alpha*deltayAffin, llambda+alpha*deltalambdaAffin)/m)
     ###5
     Sigma = (Muaff/Mu)*(Muaff/Mu)*(Muaff/Mu)
     ###6
     bb = np.concatenate((-rd,-rp, -LaYe - np.multiply(deltalambdaAffin, deltayAffin) + Sigma*Mu))
     deltacorr = spsolve(K, bb )
     ###7
     deltaxcorr = deltacorr[0:n] 
     deltaycorr = deltacorr[n:n+m]
     deltalambdacorr = deltacorr[n+m:2*m+n] 
     tao = 0.2
     alpha_tao_pri = 1  
     alpha_tao_dual = 1
     if any(deltaycorr<0):
      ytmp = y-(1.0-tao)*y
      alpha_tao_pri = min(-ytmp[deltaycorr<0]/deltaycorr[deltaycorr<0])
     if any(deltalambdacorr<0):
      llambdatmp = llambda-(1.0-tao)*llambda
      alpha_tao_dual = min(-llambdatmp[deltalambdacorr<0]/(deltalambdacorr[deltalambdacorr<0])  )
     alpha_tao_pri = min(1, alpha_tao_pri)
     alpha_tao_dual = min(1, alpha_tao_dual)

     alpha = min(alpha_tao_pri, alpha_tao_dual)
     x2 = x + alpha*deltaxcorr
     if x2.dot(x2) < errorgrad:
  	break    
#     if (x.T).dot(G.dot(x))*0.5 + c.dot(x) < (x2.T).dot(G.dot(x2))*0.5 + c.dot(x2):
     x = x2#x + alpha*deltaxcorr
     # fitness = (x2.T).dot(G.dot(x2))*0.5 + c.dot(x2)
     y = y + alpha*deltaycorr
     llambda = llambda + alpha*deltalambdacorr
     #print fitness
   return x, (x.T).dot(G.dot(x))*0.5 + c.dot(x), i, y, llambda
