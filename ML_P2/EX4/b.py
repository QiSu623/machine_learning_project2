# (b) Implement function perceptron.
# This should contain a loop that calls perceptronUpdate until it converges or the maximum iteration count, 100, has been reached.
# Make sure you randomize the order of the training data on each iteration.

import numpy as np
from matplotlib import *
from pylab import *
import sys
import matplotlib.pyplot as plt
import time

# <GRADED>
def perceptron(xs, ys):
    """
    function w=perceptron(xs,ys);

    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)

    Output:
    w : weight vector (1xd)
    b : bias term
    """

    assert (len(xs.shape) == 2), "The first input to Perceptron must be a _matrix_ of row input vecdtors."
    assert (len(ys.shape) == 1), "The second input to Perceptron must be a _vector_ of n labels (try ys.flatten())."

    n, d = xs.shape  # so we have n input vectors, of d dimensions each

    ## fill in code ...
    ## ... until here

    # initialization
    w = [0] * d
    b = [0]
    learning_rate = 0.1

    # iteration
    for i in range(n):
        if sign(ys[i]*(np.dot(w, xs[i])+b)) <= 0:
            w += ys[i]*xs[i]
            b += ys[i]

    return (w, b)

# </GRADED>

# You can use the following script to test your code and visualize your perceptron on linearly separable data in 2 dimensions. Your classifier should find a separating hyperplane on such data.
# number of input vectors
N = 100

# generate random (linarly separable) data
xs = np.random.rand(N, 2)*10-5
# defining random hyperplane
w0 = np.random.rand(2)
b0 = rand()*2-1;

# assigning labels +1, -1 labels depending on what side of the plane they lie on
ys = np.sign(xs.dot(w0)+b0)

# call perceptron to find w from data
w, b = perceptron(xs.copy(), ys.copy())

# test if all points are classified correctly
assert (all(np.sign(ys*(xs.dot(w)+b)) == 1.0))  # yw'x should be +1.0 for every input
print("Looks like you passed the Perceptron test! :o)")

# we can make a pretty visualizxation
from helperfunctions import visboundary
visboundary(w,b,xs,ys)


def onclick(event):
    global w, b, ldata, ax, line, xydata

    pos = np.array([[event.xdata], [event.ydata]])
    if event.key == 'shift':  # add positive point
        color = 'or'
        label = 1
    else:  # add negative point
        color = 'ob'
        label = -1
    ax.plot(pos[0], pos[1], color)
    ldata.append(label);
    xydata = np.vstack((xydata, pos.T))

    # call Perceptron function
    w, b = perceptron(xydata, np.array(ldata).flatten())

    # draw decision boundary
    q = -b / (w ** 2).sum() * w;
    if line == None:
        line, = ax.plot([q[0] - w[1], q[0] + w[1]], [q[1] + w[0], q[1] - w[0]], 'b--')
    else:
        line.set_data([q[0] - w[1], q[0] + w[1]], [q[1] + w[0], q[1] - w[0]])


xydata = rand(0, 2)
ldata = []
w = zeros(2)
b = 0
line = None

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(0, 1)
plt.ylim(0, 1)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
title('Use shift-click to add negative points.')

w,b=perceptron(xydata,np.array(ldata).flatten())
xydata

# w,b=perceptron(Xdata,np.array(ldata))
q=-b/(w**2).sum() *w;
line, = ax.plot([q[0]-w[1],q[0]+w[1]],[q[1]+w[0],q[1]-w[0]],'b--')
line,=ax.plot([0.2,0.2],[0.8,0.8])