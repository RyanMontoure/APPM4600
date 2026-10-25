import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    f = lambda x: 1 / (1 + x**2)
    fp = lambda x: -2 * x / ((1 + x**2)**2)  # Derivative of f(x) for clamped BCs
    a = -5
    b = 5
    
    ''' number of intervals'''
    for Nint in [5, 10, 15, 20]:
        ''' create Chebyshev interpolation nodes on [a, b] '''
        k = np.arange(1, Nint + 2)  #
        x_hat = np.cos((2*k - 1) * np.pi / (2*(Nint + 1)))  # Chebyshev nodes on [-1, 1][cite: 3]
        xint = 0.5 * (a + b) + 0.5 * (b - a) * x_hat       # Map to interval [a, b][cite: 3]
        xint = np.sort(xint)                                # Sort into ascending order!
        yint = f(xint)

        ''' create points you want to evaluate at'''
        Neval = 100
        xeval = np.linspace(xint[0], xint[Nint], Neval+1)
    
        # Endpoint derivative values for clamped boundary condition
        fp_a = fp(a)
        fp_b = fp(b)

        (M, C, D) = create_clamped_spline(yint, xint, Nint, fp_a, fp_b)
    
        yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)
    
        ''' evaluate f at the evaluation points'''
        fex = f(xeval)
        
        nerr = norm(fex - yeval)
    
        plt.figure()    
        plt.plot(xeval, fex, label='exact function')
        plt.plot(xeval, yeval, label='clamped spline')
        plt.title(f'N = {Nint}')
        plt.legend()
    plt.show()
     

def create_clamped_spline(yint, xint, N, fp_a, fp_b):
    # Step sizes h
    h = np.diff(xint)

    # Right hand side vector b
    b = np.zeros(N+1)

    # Interior RHS entries
    for i in range(1, N):
        b[i] = (yint[i+1] - yint[i]) / h[i] - (yint[i] - yint[i-1]) / h[i-1]

    # Clamped RHS boundary entries
    b[0] = (yint[1] - yint[0]) / h[0] - fp_a
    b[N] = fp_b - (yint[N] - yint[N-1]) / h[N-1]

    # Linear system matrix A
    A = np.zeros((N+1, N+1))

    # Row 0: Clamped BC at x_0
    A[0][0] = h[0] / 3.0
    A[0][1] = h[0] / 6.0

    # Interior rows
    for j in range(1, N):
        A[j][j-1] = h[j-1] / 6.0
        A[j][j] = (h[j] + h[j-1]) / 3.0
        A[j][j+1] = h[j] / 6.0

    # Row N: Clamped BC at x_N
    A[N][N-1] = h[N-1] / 6.0
    A[N][N] = h[N-1] / 3.0

    Ainv = inv(A)
    M = Ainv.dot(b)

    # Linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6.0
        D[j] = yint[j+1] / h[j] - h[j] * M[j+1] / 6.0

    return (M, C, D)
       

def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    hi = xip - xi
    yeval = (Mi * (xip - xeval)**3 + (xeval - xi)**3 * Mip) / (6 * hi) \
            + C * (xip - xeval) + D * (xeval - xi)
    return yeval 
    
    
def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        atmp = xint[j]
        btmp = xint[j+1]
        
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j+1], C[j], D[j])
        yeval[ind] = yloc

    return yeval


driver()