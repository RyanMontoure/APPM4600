import numpy as np
from numpy import polynomial
import math

def driver():
    n = 10
    for x in [2, 4, 6, 8, 10]:
        f = lambda t: t**(x-1)

        # w is the weights
        # t is the points to plug into the gamma function
        t, w = polynomial.laguerre.laggauss(n)

        sum = 0
        for i in range(n):
            sum += w[i] * f(t[i])

        real = math.factorial(x-1)
        print('x =', x)
        print('Real =', real)
        print('Laggauss =', sum)
        print('Error =', np.abs(real-sum),'\n')


driver()