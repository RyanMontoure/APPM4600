import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def driver():
    a = -5
    b = 5
    f = lambda x: 1 / (1+x**2)
    Ntrap = 1291
    Nsimp = 108

    intTrap = eval_composite_trap(a, b, f, Ntrap)
    intSimp = eval_composite_simpsons(a, b, f, Nsimp)
    intScipy6, _, info6 = quad(f, a, b, full_output=True)
    intScipy4, _, info4 = quad(f, a, b, epsabs=1e-4, epsrel=1e-4, full_output=True)

    print('Trapezoid:', intTrap, 'evaluations:',Ntrap+1)
    print("Simpson's:",intSimp, 'evaluations:', Nsimp+1)
    print("SCIPY Quad (10^-4):", intScipy4, 'evaluations:', info4['neval'])
    print('SCIPY Quad (10^-6):', intScipy6, 'evaluations:', info6['neval'])

# Evaluates Composite Tapezoidal Integration
# interval [a, b], function f(x), N intervals
def eval_composite_trap(a, b, f, N):
    h = (b-a)/N
    sum = 0

    for j in range(1, N, 1):
        xj = a + (j*h)
        sum += f(xj)

    integral = (h/2) * (f(a) + f(b) + (2*sum))
    return integral

# Evaluates Composite Simpson's Integration
# interval [a, b], function f(x), N intervals
def eval_composite_simpsons(a, b, f, N):
    # ensures an even number of intervals
    if N%2 != 0:
       N += 1

    h = (b-a)/N
    sum1 = 0

    for j in range(2, N, 2):
        x2j = a + (j*h)
        sum1 += f(x2j)

    sum2 = 0
    for j in range(1, N, 2):
        x2j1 = a + (j*h)
        sum2 += f(x2j1)

    integral = (h/3) * (f(a) + 2*sum1 + 4*sum2 + f(b))
    return integral



driver()