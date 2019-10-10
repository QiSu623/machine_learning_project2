#(a) Implement the process of updating the weight vector in the following function.
#<GRADED>
import numpy as np
from matplotlib import *
#matplotlib.use('PDF')
from pylab import *
from retry import retry
#</GRADED>
import sys
import matplotlib.pyplot as plt
import time

# add p02 folder
sys.path.insert(0, './p02/')

print('You\'re running python %s' % sys.version.split(' ')[0])

# <GRADED>

def perceptronUpdate(x, y, w):
    """
    function w=perceptronUpdate(x,y,w);

    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions

    Output:
    w : weight vector after updating (d)
    """
    assert (y in {-1, 1})
    assert (len(w.shape) == 1), "At the update w must be a vector not a matrix (try w=w.flatten())"
    assert (len(x.shape) == 1), "At the update x must be a vector not a matrix (try x=x.flatten())"

    ## fill in code ...
    ## ... until here
    if sign(y*(np.dot(x,w))) <= 0:
        w += y*x
    return w.flatten()
# </GRADED>


# test the update code:
x = rand(5)  # random weight vector
w = rand(5)  # random feature vector
y = -1  # random label
wnew = perceptronUpdate(x, y, w.copy())  # do a perceptron update
assert(norm(wnew-w+x) < 1e-10), "perceptronUpdate didn't pass the test : ("  # if correct, this should return 0
print("Looks like you passed the update test : )")

