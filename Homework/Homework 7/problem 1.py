import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def driver():
    xvals = np.linspace(-1, 1, 1001)
    f = 1 / (1 + (10*xvals)**2) # real function

    N = 19
    h = 2/(N - 1)
    x = np.zeros(N)
    for i in range(N):
        # I alluded the i-1 term because i indexes 0 -> N-1
        x[i] = -1 + (i)*h # interpolation points

    # Vandermonde Matrix
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            V[i, j] = x[i] ** j

    # Y Values
    y = 1 / (1 + (10*x)**2)

    # coefficient vector
    c = np.linalg.inv(V) @ y

    p = np.zeros(1001)
    for i in range(N):
        p += c[i] * xvals**i

    plt.plot(x, y, 'o', label = 'Data Points') # data points
    plt.plot(xvals, p, label = 'Interpolation')
    plt.plot(xvals, f, label = 'f(x)')
    plt.legend()
    plt.show()



driver()