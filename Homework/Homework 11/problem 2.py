import matplotlib.pyplot as plt
import numpy as np


def driver():

    f = lambda x: np.sin(x)
    a = 0
    b = np.pi

    # exact integral
    I_ex = 2

    nrefine = 5

# number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0, nrefine + 1)

    Rom_simp = rombergSimp(a, b, nrefine, f)
    Rom_trap = rombergTrap(a, b, nrefine, f)

# column j is only filled from row j down
    print('--- Romberg-Simpson Results ---')
    for j in range(0, nrefine + 1):
        print('absolute error in column', j, '= ', abs(I_ex - Rom_simp[j:, j]))

    err_simp = abs(I_ex - np.diag(Rom_simp))
    err_trap = abs(I_ex - np.diag(Rom_trap))

    print('\nabsolute error of best approximation (Romberg-Simpson)    = ', err_simp)
    print('absolute error of best approximation (Romberg-Trapezoidal) = ', err_trap)

    # Function evaluations: n+1 for Trapezoidal, 2n+1 for Simpson
    evals_trap = ntest + 1
    evals_simp = 2 * ntest + 1

    # ax1.semilogy(ntest, err_trap, 'o-', label='Romberg-Trapezoidal')
    # ax1.semilogy(ntest, err_simp, 's--', label='Romberg-Simpson')
    # ax1.set_xlabel('Number of panels (n)')
    # ax1.set_ylabel('Absolute error')
    # ax1.set_title('Error vs. Panel Count')
    # ax1.legend()
    # ax1.grid(True, which='both', linestyle=':')

    # ax2.semilogy(evals_trap, err_trap, 'o-', label='Romberg-Trapezoidal')
    # ax2.semilogy(evals_simp, err_simp, 's--', label='Romberg-Simpson')
    # ax2.set_xlabel('Number of function evaluations')
    # ax2.set_ylabel('Absolute error')
    # ax2.set_title('Error vs. Function Evaluations')
    # ax2.legend()
    # ax2.grid(True, which='both', linestyle=':')

    # plt.tight_layout()
    # plt.show()


def rombergSimp(a, b, nrefine, f):
    # number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0, nrefine + 1)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    # first column: composite Simpson's rule with n panels (2n subintervals)
    for j in range(0, nn):
        Rom[j, 0] = CompSimp(a, b, ntest[j], f)

    # remaining columns: Richardson extrapolation
    # Column 0 error is O(h^4), so column j eliminates O(h^(2j+2)) via 4^(j+1) - 1
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j - 1] + (Rom[i, j - 1] - Rom[i - 1, j - 1]) / (4**(j + 1) - 1)

    return Rom


def CompSimp(a, b, n, f):
    # n panels = 2n subintervals of width h = (b - a)/(2n)
    h = (b - a) / (2 * n)
    xnode = a + np.arange(0, 2 * n + 1) * h

    I_simp = f(xnode[0]) + f(xnode[2 * n])

    for j in range(1, n):
        I_simp = I_simp + 2 * f(xnode[2 * j])
    for j in range(1, n + 1):
        I_simp = I_simp + 4 * f(xnode[2 * j - 1])

    I_simp = I_simp * h / 3
    return I_simp


def rombergTrap(a, b, nrefine, f):
    ntest = 2**np.arange(0, nrefine + 1)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    for j in range(0, nn):
        Rom[j, 0] = CompTrap(a, b, ntest[j], f)

    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j - 1] + (Rom[i, j - 1] - Rom[i - 1, j - 1]) / (4**j - 1)

    return Rom


def CompTrap(a, b, n, f):
    h = (b - a) / n
    xnode = a + np.arange(0, n + 1) * h

    I_trap = h * f(xnode[0]) * 0.5
    for j in range(1, n):
        I_trap = I_trap + h * f(xnode[j])
    I_trap = I_trap + 0.5 * h * f(xnode[n])

    return I_trap


driver()