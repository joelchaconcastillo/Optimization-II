"""
============
Contour Demo
============

Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also the :doc:`contour image example
</gallery/images_contours_and_fields/contour_image>`.
"""
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
def graph(formula, x_range, y_range):  
    x = np.array(x_range)  
    y = np.array(y_range)  
    z = eval(formula)
    plt.plot(x, y)
    plt.show()

delta = 0.025
minx = -4.0
maxx = 4.0
miny = -4.0
maxy = 4.0
x = np.arange( minx, maxx, delta)
y = np.arange( miny, maxy, delta)
gradientc1 = [-3, -1]
gradientc2 = [0, 1]
gradient = [-2, 1]
lambdax = [2.0/3.0, 5.0/3.0]
point = [0, 1]
X, Y = np.meshgrid(x, y)
Z1 = X #np.exp(-X**2 - Y**2)
Z2 = Y #np.exp(-(X - 1)**2 - (Y - 1)**2)

Z = (-2*Z1 + Z2) 


###############################################################################
# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=100, fontsize=10)
ax.set_title('Optimizacion con restricciones $min -2x_1 + x_2$ s.a. $C_1: (1-x_1)^3 - x_2 \geq 0 \quad C_2:0.25x_1^2 + x_2 -1 \geq 0$')



#Constrain 1:
#graph('(1-x)^3 - y', range(, 1))
s = np.linspace(minx, maxx)
yc1 = (1-s)**3

plt.plot(s, yc1, lw=2, marker='.', color='blue', alpha=0.7, label='$C_1(x^*)$')
plt.ylim(miny, maxy)

yc2 = 1 - 0.25*(s**2)
plt.plot(s, yc2, lw=2, color='b', label='$C_2(x^*)$')
plt.ylim(miny, maxy)

# plot the possible (s, t) pairs
pairs = [(s, t) for s in np.linspace(minx, maxx,200)
                for t in np.linspace(miny, maxy, 200)
                if (1-s)**3- t >= 0
                and ( 0.25* s**2  + t -1) >=0]
ss, ts = np.hsplit(np.array(pairs), 2)

# plot the results
#plt.scatter(ss, ts,color='gray' , cmap='jet', label='$Region \quad factible$', zorder=3, alpha=0.3,edgecolors='none' )
plt.plot( point[0], point[1], marker='o', color="red", label='$x^* = [0, 1]^T$',markersize= 5 )

###Conjunto factible
pairs = [(d1, d2) for d2 in np.linspace(miny, maxy,200)
                for d1 in np.linspace( minx , maxx, 200)
                if d2 >=0]
ss, ts = np.hsplit(np.array(pairs), 2)
# plot the results
plt.scatter(ss+point[0], ts+point[1], color='green', cmap='jet', label='$\\nabla C_1(x^*)^T D \geq 0$', zorder=3, alpha=0.1, edgecolors='none' )


pairs = [(d1, d2) for d2 in np.linspace(miny, maxy,200)
                for d1 in np.linspace( minx , maxx, 200)
                if -3.0*d1 - d2   >=0 ]
ss, ts = np.hsplit(np.array(pairs), 2)
# plot the results
plt.scatter(ss+point[0], ts+point[1], color='yellow', cmap='jet', label='$\\nabla C_2(x^*)^T D \geq 0$', zorder=3, alpha=0.3, edgecolors='none' )
pairs = [(d1, d2) for d2 in np.linspace(0, maxy,200)
                for d1 in np.linspace( minx , maxx, 200)
                if d1 <= -d2/3.0]
ss, ts = np.hsplit(np.array(pairs), 2)
# plot the results
plt.scatter(ss+point[0], ts+point[1], color='black', cmap='jet', label='$  \\sum _i \\nabla C_i(x^*)^T D \geq 0 $', zorder=3, alpha=0.1, edgecolors='none' )





#Gradientes..
style="Simple,tail_width=1.9,head_width=4,head_length=8"
kw1 = dict(arrowstyle=style, color="b", label='$\\nabla C_1(x^*)^T \\lambda_1$')
kw2 = dict(arrowstyle=style, color="y", label='$\\nabla C_2(x^*)^T \\lambda_2$')
kw3 = dict(arrowstyle=style, color="r", label='$\\nabla f(x^*)$')
#kw4 = dict(arrowstyle=style, color="g", label='$\\lambda^* \quad Lagrangiano$')

a1 = patches.FancyArrowPatch( (point[0], point[1]), (point[0]+gradientc1[0]*lambdax[0], point[1] + gradientc1[1]*lambdax[0]),**kw1 )

a2 = patches.FancyArrowPatch( (point[0], point[1]), (point[0]+gradientc2[0]*lambdax[1], point[1] + gradientc2[1]*lambdax[1]),**kw2 )

a3 = patches.FancyArrowPatch( (point[0], point[1]), (point[0]+gradient[0], point[1] + gradient[1]),**kw3 )

#a4 = patches.FancyArrowPatch( (point[0], point[1]), (point[0]+lambdax[0], point[1] + lambdax[1]),**kw4 )

for a in [a1,a2,a3]:
    plt.gca().add_patch(a)

plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$x_2$', fontsize=16)
#plt.legend(['My label','aa','aa','oo','aa'])
plt.legend(fontsize=14)
#plt.savefig('Problema1.eps', format='eps', dpi=1000)
#plt.savefig('Problema1.png', format='png', dpi=1000)
plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions and methods is shown
# in this example:

#import matplotlib
#matplotlib.axes.Axes.contour
#matplotlib.pyplot.contour
#matplotlib.figure.Figure.colorbar
#matplotlib.pyplot.colorbar
#matplotlib.axes.Axes.clabel
#matplotlib.pyplot.clabel
#matplotlib.axes.Axes.set_position
#matplotlib.axes.Axes.get_position
