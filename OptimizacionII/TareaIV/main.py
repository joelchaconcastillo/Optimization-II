import numpy as np
import matplotlib.pyplot as plt
I = [[0.5, 0], [0, 0.5]]
mu1 = [0, 0]
mu2 = [10, 10]
d1 = np.random.multivariate_normal( mu1,I , 100)
d2 = np.random.multivariate_normal( mu2,I, 100)
print d1
print d2
plt.plot(d1[:,0], d1[:,1],'.')
plt.plot(d2[:,0], d2[:,1],'.')
plt.show()
