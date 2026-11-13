import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import special
import math

def driver():
    N = 1000
    a = 0
    b = 60

    for x in [2, 4, 6, 8, 10]:
        real_gamma = math.factorial(x-1)

        trap_gamma = eval_composite_trap_gamma(a, b, x, N)
        trap_err = np.abs(real_gamma - trap_gamma)

        scipy_gamma = special.gamma(x)
        scipy_err = np.abs(real_gamma - scipy_gamma)

        print('X =', x, 'Real Value =', real_gamma)
        print('Trapezoid:', trap_gamma, 'Error:', trap_err)
        print('SCIPY:', scipy_gamma, 'Error:', scipy_err)
        print('\n')



# Evaluates Composite Tapezoidal Integration
# interval [a, b], function f(x), N intervals
def eval_composite_trap_gamma(a, b, x, N):
    h = (b-a)/N
    integrand = lambda t: t**(x-1) * np.exp(-t)

    sum = 0

    for j in range(1, N, 1):
        tj = a + (h*j)
        sum += integrand(tj)

    gamma = (h/2) * (integrand(a) + integrand(b) + (2*sum))
    return gamma


driver()