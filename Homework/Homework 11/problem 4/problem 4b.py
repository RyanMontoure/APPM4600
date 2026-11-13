import numpy as np
import scipy.integrate as spi
import math


def driver():
    N = 1000
    a = 0
    b = 60
    tol = 1e-10

    for x in [2, 4, 6, 8, 10]:
        real_gamma = math.factorial(x - 1)
        integrand = lambda t: t**(x - 1) * np.exp(-t)

        trap_gamma, _, trap_evals = eval_composite_trap_gamma(a, b, x, N)
        trap_err = abs(real_gamma - trap_gamma)

        adap_gamma, _, _, adap_evals = adaptive_quad(a, b, x, tol, N, eval_composite_trap_gamma)
        adap_err = abs(real_gamma - adap_gamma)

        print('x =', x)
        print('Real:', real_gamma)
        print('Trapezoid:', trap_gamma, 'Error:', trap_err, 'Evaluations:', trap_evals)
        print('Adaptive:', adap_gamma, 'Error:', adap_err, 'Evaluations:', adap_evals)


# Evaluates Composite Tapezoidal Integration
# interval [a, b], N intervals
def eval_composite_trap_gamma(a, b, x, N):
    h = (b - a) / N
    tj = np.linspace(a, b, N + 1)
    integrand = lambda t: t**(x - 1) * np.exp(-t)

    y = integrand(tj)
    gamma = (h / 2) * (y[0] + y[-1] + 2 * np.sum(y[1:-1]))
    neval = N + 1
    return gamma, tj, neval


# adaptive quad I altered in lab 11

def adaptive_quad(a, b, x, tol, M, method, maxit=500):
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros(maxit)
    left_p[0], right_p[0] = a, b

    s[0], xv, total_evals = method(a, b, x, M)
    X = [xv]

    j = 1
    I = 0
    nsplit = 1
    while j < maxit:
        c = 0.5 * (left_p[j - 1] + right_p[j - 1])
        s1, xv, ev1 = method(left_p[j - 1], c, x, M)
        X.append(xv)
        s2, xv, ev2 = method(c, right_p[j - 1], x, M)
        X.append(xv)
        total_evals += ev1 + ev2

        if np.max(np.abs(s1 + s2 - s[j - 1])) > tol:
            left_p[j] = left_p[j - 1]
            right_p[j] = c
            s[j] = s1
            left_p[j - 1] = c
            s[j - 1] = s2
            j += 1
            nsplit += 1
        else:
            I += s1 + s2
            j -= 1
            if j == 0:
                j = maxit

    return I, np.unique(np.concatenate(X)), nsplit, total_evals


driver()