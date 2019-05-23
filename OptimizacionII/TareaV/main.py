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
def  buildDictionary(inImage, ix, jx, Lw, Nw):
   D = np.zeros((Lw*Lw,Nw))
   ratio = 0.7
   for i in range(0,Nw):
      window = inImage[ix:ix+Lw,jx:jx+Lw]
      rowmask = np.ones(Lw*Lw) 
      rowmask[0:int(Lw*Lw*ratio)]=0
      rowmask = np.random.permutation(rowmask)
      #print(rowmask)
      mask =np.reshape( rowmask , (Lw, Lw) )
      window =  window*mask
      submatrix = (window).flatten()
      submatrix = (submatrix - np.mean(submatrix)) #/ np.std(submatrix)
      #submatrix = (submatrix-min(submatrix))/(max(submatrix)-min(submatrix))
      D[:,i] =  submatrix #inImage[range(pivot1,pivot1+Lw), range(pivot2,pivot2+Lw)]
   return D       
def solveProblem(Y, D):
   return ADDM_LASSO(np.array(D), np.array(Y), 100, 2)
   clf = linear_model.Lasso(alpha=2)
   clf.fit(D, Y)
   return clf.coef_

def processImage(inImage, Nw, Lw):
   Hi = np.shape(inImage)[0] ## Row number image
   Wi = np.shape(inImage)[1]  ##Column number image
   outImage = np.zeros((Hi, Wi))
   for i in range(0,Hi-Lw):
     for j in range(0,Wi-Lw):
       flatwindow = inImage[i:i+Lw,j:j+Lw].flatten()
       
       D = buildDictionary(inImage, i, j, Lw, Nw)
       alpha = solveProblem(flatwindow, D)     
       flatwindow = np.reshape(D.dot(alpha),(Lw,Lw))
       if j > 0:
        outImage[i:i+Lw,j:j+Lw] = (flatwindow[0:Lw,0:Lw] + outImage[i:i+Lw,j-1:j+Lw-1]) /2.0
        #outImage[i:i+Lw,j:j+Lw] = (flatwindow[0:Lw,0:Lw] + np.mean(inImage[i:i+Lw,j+Lw]) + outImage[i:i+Lw,j-1:j+Lw-1]) /2.0
        outImage[i:i+Lw,j+Lw] = (flatwindow[:,Lw-1])
       if j == 0:
        outImage[i:i+Lw,j:j+Lw] = flatwindow[0:Lw,0:Lw] ##+ np.mean(inImage[i:i+Lw,j+Lw])
       #outImage[i:i+Lw,j+Lw] = outImage[i:i+Lw,j+Lw] + np.mean(inImage[i:i+Lw,j+Lw])
       #outImage[i:i+Lw,j+Lw] = (outImage[i:i+Lw,j+Lw] + inImage[i:i+Lw,j+Lw])/2.0
       #outImage[i:i+Lw,j:j+Lw] = np.reshape(flatwindow+np.mean(flatwindow),(Lw, Lw))
     eprint(i)
   #outImage = (outImage + inImage)/2.0
   #outImage = (255*preprocessing.normalize(outImage)).astype(np.int);
   print (outImage)
   outImage = 255*(outImage - np.min(outImage))/(np.max(outImage)-np.min(outImage))
   return outImage 

#inImage = pgmread("mri_NOISE.pgm")
#inImage = pgmread("glassware_NOISE.pgm")
inImage = pgmread("ardilla_NOISE.pgm")
inImagef = inImage[0].astype(np.float)
#inImagef = inImagef[100:200,100:200]
Nw = 5 #Dictionary size..
Lw = 2 #width size of each window
#D = buildDictionary(inImagef, Nw, Lw)
outImage = processImage(inImagef, Nw, Lw)
pgmwrite(outImage, "out.pgm")
