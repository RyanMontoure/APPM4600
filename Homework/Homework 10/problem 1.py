import numpy as np
import matplotlib.pyplot as plt

def driver():
    a = 0
    b = np.pi
    f = lambda x: np.exp(x)*np.cos(x)
    Ntrap = 1
    Nsimp = 2 # starts at 2 since it needs an even number
    I = -12.070346316389632

    # quad using trapezoidal doubling the intervals until the relative error is satisfied
    # creates an array of the h and error values
    trapezoid, ht = eval_composite_trap(a, b, f, Ntrap)
    htrap = np.array([ht])
    error = np.abs(I - trapezoid)
    errortrap = np.array([error])
    while (error > 1e-10):
        Ntrap *= 2
        trapezoid, ht = eval_composite_trap(a, b, f, Ntrap)
        htrap = np.append(htrap, ht)
        error = np.abs(I - trapezoid)
        errortrap = np.append(errortrap, error)

    # quad using simpsons doubling the intervals until the relative error is satisfied
    # creates an array of the h and error values
    simpsons, hs = eval_composite_simpsons(a, b, f, Nsimp)
    hsimp = np.array([hs])
    error = np.abs(I - simpsons)
    errorsimp = np.array([error])
    while (error > 1e-10):
        Nsimp *= 2
        simpsons, hs = eval_composite_simpsons(a, b, f, Nsimp)
        hsimp = np.append(hsimp, hs)
        error = np.abs(I - simpsons)
        errorsimp = np.append(errorsimp, error)

    # plotting h vs. error on a loglog timescale for both quad methods
    plt.loglog(htrap, errortrap, label = 'Trapezoid')
    plt.loglog(hsimp, errorsimp, label = 'Simpsons')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    # calculating slopes of the lines on log-log scale
    slope_trap, _ = np.polyfit(np.log(htrap), np.log(errortrap), 1)
    slope_simp, _ = np.polyfit(np.log(hsimp), np.log(errorsimp), 1)

    print(f"Trapezoidal Rule Slope: {slope_trap:.4f}")
    print(f"Simpson's Rule Slope:   {slope_simp:.4f}")
        
    
    


# Evaluates Composite Tapezoidal Integration
# interval [a, b], function f(x), N intervals
def eval_composite_trap(a, b, f, N):
    h = (b-a)/N
    sum = 0

    for j in range(1, N, 1):
        xj = a + (j*h)
        sum += f(xj)

    integral = (h/2) * (f(a) + f(b) + (2*sum))
    return (integral, h)

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
    return (integral, h)



driver()