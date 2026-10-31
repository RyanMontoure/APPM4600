import matplotlib.pyplot as plt
import numpy as np

def driver():
    f = lambda x: np.sin(x)

    p33 = lambda x: (x - (7/60)*x**3) / (1 + (1/20)*x**2)
    p42 = lambda x: x / (1 + (1/6)*x**2 + (7/360)*x**4)
    taylor = lambda x: x - (x**3)/6 + (x**5)/120

    xvals = np.linspace(0, 5, 1000)
    fvals = f(xvals)
    p33vals = p33(xvals)
    p42vals = p42(xvals)
    taylorvals = taylor(xvals)

    plt.plot(xvals, fvals, label = 'Sin(x)')
    plt.plot(xvals, p33vals, label = 'P33 and P42')
    plt.plot(xvals, p42vals, label = 'P24')
    plt.plot(xvals, taylorvals, label = 'Taylor Series')
    plt.legend()
    plt.show()


driver()