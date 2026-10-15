import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def driver():
    xvals = np.linspace(-1, 1, 1001)
    f = 1 / (1 + (10*xvals)**2) # real function

    N = 15

    xnodes = np.linspace(-1, 1, N) # interpolation data points
    ynodes = 1 / (1 + (10*xnodes)**2)

    w = np.zeros(N) # wj function
    for j in range(N):
        w[j] = 1 / np.prod(xnodes[j] - xnodes[np.arange(N) != j])

    diff = xvals[:, None] - xnodes[None, :]
    phi = np.prod(diff, axis=1)

    p = 0
    for j in range(N):
        p += w[j] * ynodes[j] / (xvals - xnodes[j])
    p *= phi

    plt.plot(xvals, f, label = 'f(x)')
    plt.plot(xnodes, ynodes, 'o', label = 'Data Points')
    plt.plot(xvals, p, label = 'interpolation')
    plt.legend()
    plt.show()

driver()