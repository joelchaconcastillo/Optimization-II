from __future__ import print_function
import numpy as np
import math
import scipy.stats as ss
from sklearn import linear_model, preprocessing
from pgm import pgmread, pgmwrite
from ADMM_LASSO import ADDM_LASSO
import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
def solveProblem(Y, D):
   clf = linear_model.Lasso(alpha=2)
   clf.fit(D, Y)
   return clf.coef_
A = np.array([[0,0], [1, 1], [2, 12]])
b = np.array([[0],[1],[20]])
print(ADDM_LASSO(A, b, 100000, 0.3))

clf = linear_model.Lasso(alpha=0.3, fit_intercept=False)
print(clf.fit([[0,0], [1, 1], [2, 12]], [0, 1, 20]))
print(clf.coef_)
