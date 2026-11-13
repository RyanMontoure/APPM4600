import numpy as np
import matplotlib.pyplot as plt


def driver():
    a = 0
    b = 1

    N = 4

    f = lambda t: 0.0 if t == 0 else t * np.cos(1/t)

    approx = eval_composite_simpsons(a, b, f, N)
    print('Composite Simpsons:', approx)

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