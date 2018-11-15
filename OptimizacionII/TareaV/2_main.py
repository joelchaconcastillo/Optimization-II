from __future__ import print_function
import numpy as np
import math
import scipy.stats as ss
from sklearn import linear_model, preprocessing
from pgm import pgmread, pgmwrite
import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
def  buildDictionary(inImage, Nw, Lw):
   D = np.zeros((Lw*Lw,Nw))
   for i in range(0,Nw):
      pivot1 = np.random.randint(np.shape(inImage)[0]-Lw)
      pivot2 = np.random.randint(np.shape(inImage)[1]-Lw)
      submatrix = inImage[pivot1:pivot1+Lw, pivot2:pivot2+Lw].flatten()
      submatrix = (submatrix - np.mean(submatrix)) / np.std(submatrix)
      submatrix = (submatrix-min(submatrix))/(max(submatrix)-min(submatrix))
      D[:,i] =  submatrix #inImage[range(pivot1,pivot1+Lw), range(pivot2,pivot2+Lw)]
   return D       
def solveProblem(Y, D):
   clf = linear_model.Lasso(alpha=1.5)
   clf.fit(D, Y)
   return clf.coef_

def processImage(D, inImage):
   Hi = np.shape(inImage)[0] ## Row number image
   Wi = np.shape(inImage)[1]  ##Column number image
   outImage = np.zeros((Hi, Wi))
   Lw = int(math.sqrt(np.shape(D)[0])) ##size of window
   Nw = np.shape(D)[1] ##number of windows
   for i in range(0,Hi-Lw):
     for j in range(0,Wi-Lw):
       flatwindow = inImage[i:i+Lw,j:j+Lw].flatten()
       alpha = solveProblem(flatwindow, D)     
       flatwindow = np.reshape(D.dot(alpha),(Lw,Lw))
     #  if j == 0:
     #    outImage[i:i+Lw,j:j+Lw] = np.reshape(flatwindow,(Lw, Lw))
       if j > 0:
        outImage[i:i+Lw,j:j+Lw] = (flatwindow[0:Lw,0:Lw] + outImage[i:i+Lw,j-1:j+Lw-1]) /2.0
        outImage[i:i+Lw,j+Lw] = (flatwindow[:,Lw-1])
       outImage[i:i+Lw,j+Lw] = (outImage[i:i+Lw,j+Lw] + inImage[i:i+Lw,j+Lw])/2.0
       #outImage[i:i+Lw,j:j+Lw] = np.reshape(flatwindow+np.mean(flatwindow),(Lw, Lw))
     eprint(i)
   #outImage = (255*preprocessing.normalize(outImage)).astype(np.int);
   print (outImage)
   outImage = 255*(outImage - np.min(outImage))/(np.max(outImage)-np.min(outImage))
   return outImage 

inImage = pgmread("ardilla.pgm")
inImagef = inImage[0].astype(np.float)
inImagef = inImagef[100:200,100:200]
Nw = 100 #Dictionary size..
Lw = 3 #width size of each window
D = buildDictionary(inImagef, Nw, Lw)
outImage = processImage(D, inImagef)
pgmwrite(outImage, "out.pgm")
