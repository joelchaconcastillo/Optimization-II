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
minx = -2.0
maxx = 2.0
miny = -2.0
maxy = 2.0
x = np.arange( minx, maxx, delta)
y = np.arange( miny, maxy, delta)
X, Y = np.meshgrid(x, y)
Z1 = X #np.exp(-X**2 - Y**2)
Z2 = Y #np.exp(-(X - 1)**2 - (Y - 1)**2)

Z = (Z1 * Z2) 
xoptimal = [-1.0/np.sqrt(2.0), -1.0/np.sqrt(2.0)]
gradientf = [xoptimal[1], xoptimal[0]]
gradientc1 = [-2.0/np.sqrt(2.0), -2.0/np.sqrt(2.0)]
lambda1 = [1.0/2.0, 1.0/2.0]
###############################################################################
# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=100, fontsize=10)
ax.set_title('Optimizacion con restricciones $min \quad x_1 x_2 \quad s.a. \quad x_1^2 + x_2^2 \leq 1$')


###Conjunto factible
pairs = [(d1, d2) for d2 in np.linspace(miny, maxy,200)
                for d1 in np.linspace( minx , maxx, 200)
                if d1**2 + d2**2  <= 1 ]
ss, ts = np.hsplit(np.array(pairs), 2)
# plot the results
plt.scatter(ss, ts, color='orange', cmap='jet', label='$Conjunto \quad factible$', zorder=3, alpha=0.2, edgecolors='none' )


#Constraint 1:

#s = np.linspace(minx, maxx)
#yc1 = np.sqrt(1-(s)**2)
#
#plt.plot(s, yc1, lw=2, marker='.', color='blue', alpha=0.7, label='$C_1(x^*)$')
plt.ylim(miny, maxy)

# plot the possible (s, t) pairs
pairs = [(s, t) for s in np.linspace(minx, maxx,500)
                for t in np.linspace(miny, maxy, 500)
                if np.sqrt(1-(s**2)) == 0]
ss, ts = np.hsplit(np.array(pairs), 2)

# plot the results
#plt.scatter(ss, ts,color='gray' , cmap='jet', label='$Region \quad factible$', zorder=3, alpha=0.1,edgecolors='none' )
plt.plot( xoptimal[0], xoptimal[1], marker='o', color="red", label='$x^* = [ \\frac{1}{\\sqrt{2}}, \\frac{1}{\\sqrt{2}}]^T$',markersize= 5 )


pairs = [(d1, d2) for d2 in np.linspace(miny-np.abs(xoptimal[1]), maxy,100)
                for d1 in np.linspace( minx , maxx+np.abs(xoptimal[0]), 100)
                if  d1>= d2]
ss, ts = np.hsplit(np.array(pairs), 2)
# plot the results
plt.scatter(ss+xoptimal[0], ts+xoptimal[1], color='black', cmap='jet', label='$Direcciones \quad linealizadas $', zorder=3, alpha=0.1, edgecolors='none' )





#Gradientes..
style="Simple,tail_width=1.9,head_width=4,head_length=8"
kw1 = dict(arrowstyle=style, color="b", label='$\\nabla C_1(x^*)$')
kw3 = dict(arrowstyle=style, color="r", label='$\\nabla f(x^*)$')
kw4 = dict(arrowstyle=style, color="g", label='$\\lambda^* \quad Lagrangiano$')

a1 = patches.FancyArrowPatch((xoptimal[0], xoptimal[1]), (xoptimal[0]+ gradientc1[0], xoptimal[1]+gradientc1[1]),**kw1 )

a3 = patches.FancyArrowPatch((xoptimal[0], xoptimal[1]), (xoptimal[0]+ gradientf[0], xoptimal[1]+gradientf[1]) ,**kw3 )

a4 = patches.FancyArrowPatch((xoptimal[0], xoptimal[1]), (xoptimal[0]+ lambda1[0], xoptimal[1]+lambda1[1]) ,**kw4 )

for a in [a1,a3, a4]:
    plt.gca().add_patch(a)

plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$x_2$', fontsize=16)
#plt.legend(['My label','aa','aa','oo','aa'])
plt.legend(fontsize=14)
#plt.savefig('Problema1.eps', format='eps', dpi=1000)
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
