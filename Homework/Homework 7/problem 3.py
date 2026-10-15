import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

def driver():
    xvals = np.linspace(-1, 1, 1001)
    f = 1 / (1 + (10*xvals)**2) # real function

    N = 30 # number of data points

    xnodes = np.zeros(N) # creating the data points
    for j in range(N):
        val = (2*(j+1) - 1) * np.pi / (2*N)
        xnodes[j] = np.cos(val)
    ynodes = 1 / (1 + (10*xnodes)**2)

    w = np.zeros(N) # wj function
    for j in range(N):
        w[j] = 1 / np.prod(xnodes[j] - xnodes[np.arange(N) != j])

    diff = xvals[:, None] - xnodes[None, :] # x - xj for the phi function
    phi = np.prod(diff, axis=1) # phi function

    p = 0 # creating p(x)
    for j in range(N):
        p += w[j] * ynodes[j] / (xvals - xnodes[j])
    p *= phi

    plt.plot(xvals, f, label = 'f(x)')
    plt.plot(xnodes, ynodes, 'o', label = 'Data Points')
    plt.plot(xvals, p, label = 'interpolation')
    plt.legend()
    plt.show()

driver()