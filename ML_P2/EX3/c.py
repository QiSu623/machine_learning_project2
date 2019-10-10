# (c) Implement classifyLinear that applies the weight vector and bias to the input vector.
# (The bias is an optional parameter. If it is not passed in, assume it is zero.) Make sure that the predictions returned are either 1 or -1.

import numpy as np
from matplotlib import *
from pylab import *
import sys
import matplotlib.pyplot as plt
import time


# <GRADED>
def classifyLinear(xs, w, b):
    """
    function preds=classifyLinear(xs,w,b)

    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)

    Output:
    preds: predictions (1xn)
    """
    w = w.flatten()
    predictions = np.zeros(xs.shape[0])
    ## fill in code ...
    ## ... until here
    predictions=np.sign(xs.dot(w) + b)
    return predictions
# </GRADED>

# test classifyLinear code:
xs=rand(1000,2)-0.5 # draw random data
w0=np.array([0.5,-0.3]) # define a random hyperplane
b0=-0.1 # with bias -0.1
ys=np.sign(xs.dot(w0)+b0) # assign labels according to this hyperplane (so you know it is linearly separable)
assert (all(np.sign(ys*classifyLinear(xs,w0,b0))==1.0))  # the original hyperplane (w0,b0) should classify all correctly
print("Looks like you passed the classifyLinear test! :o)")