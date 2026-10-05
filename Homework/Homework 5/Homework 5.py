import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from numpy.linalg import norm

# driver for question 1
def driver1():
    x0 = np.array([1, 1])
        
    Nmax = 100
    tol = 1e-10

    [xstar,ier,its] =  iteration2a(x0, tol, Nmax)

    print('Normal iteration:')
    print(xstar)
    print('The error message reads:', ier)
    print('Number of iterations is:', its)

    print('Newton:')
    [xstar1, ier1, its1] = Newton(x0, tol, Nmax)
    print(xstar1)
    print('Newton: The error message reads:', ier1)
    print('Number of iterations is:', its1)

# driver for question 3
def driver3():
    x0 = np.array([1.0, 1.0, 1.0])
    tol = 1e-10
    Nmax = 100

    p_hist, xstar, ier, its = iteration_ellipsoid(x0, tol, Nmax)

    print(xstar)
    print('f(xstar):', eval_f_ellipsoid(xstar))
    print('Error flag (ier):', ier)
    print('Number of iterations:', its)

    p_norms = np.array([norm(pt) for pt in p_hist])

    alpha, lambdA = orderofConvergence(p_norms, x0, ier, tol, Nmax)

    print('Alpha:', alpha)
    print('Lambda:', lambdA)
    if alpha == 2:
        print('Converges QUADRATICALLY')

# the iteration given to me in question 3
def iteration_ellipsoid(x0, tol, Nmax):
    """
    Generalized surface projection Newton iteration:
    x_{n+1} = x_n - d_n * grad f(x_n)
    where d_n = f(x_n) / ||grad f(x_n)||^2
    """
    x = x0.copy().astype(float)
    p = [x.copy()]
    
    for its in range(Nmax):
        f_val = eval_f_ellipsoid(x)
        grad_f = eval_grad_ellipsoid(x)
        norm_grad_sq = np.dot(grad_f, grad_f)
        
        d = f_val / norm_grad_sq
        x_next = x - d * grad_f
        p.append(x_next.copy())
        
        if norm(x_next - x) < tol:
            xstar = x_next
            ier = 0
            return p, xstar, ier, its + 1
            
        x = x_next
        
    xstar = x
    ier = 1
    return p, xstar, ier, Nmax

# function from question 3
def eval_f_ellipsoid(x):
    """ Evaluates f(x, y, z) = x^2 + 4y^2 + 4z^2 - 16 """
    return x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16.0

# gradient of the function for question 3
def eval_grad_ellipsoid(x):
    """ Evaluates grad f(x, y, z) = [2x, 8y, 8z] """
    return np.array([2*x[0], 8*x[1], 8*x[2]])

# order of convergence function I made in lab 3
def orderofConvergence(p, x0, ier, tol, Nmax):
    if (ier == 1):
        return ('bad fixedpt approximation')

    # np.diff creates a new vector with the difference of consecutive terms
    epsilon = np.abs(np.diff(p))
    epsilon = epsilon[epsilon > 1e-10] # to avoid divide by zero errors
    if (len(epsilon) < 3):
        return (2, 0.0) # returns quadratic if function converges super fast

    # calculates alpha
    numerator = np.log(epsilon[2:] / epsilon[1:-1])
    denominator = np.log(epsilon[1:-1] / epsilon[:-2])

    alphaV = numerator / denominator
    alpha = round(alphaV[-1])

    # calculate lambda using alpha
    lambdaV = epsilon[1:] / epsilon[:-1] ** alpha
    lambdA = lambdaV[-1]

    # determine convergence
    # if ((alpha == 1) and (lambdA < 1)):
    #     print ('The sequence converges LINEARLY')
    #     return(alpha, lambdA)
    # elif (alpha == 2):
    #     print ('The sequence converges QUADRATICALLY')
    #     return(alpha, lambdA)
    # print ('The sequence did not converge?')
    # return
    return(alpha, lambdA)


def evalF(x): 
# vector function that you want to find the roots of

    F = np.zeros(2)
    
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1
 
    return F
    
def evalJ(x): 
# Jacobian of the vector function you want to find the roots of
    J = np.array([[6*x[0], -2*x[1]],
                  [6*x[1] - 3*x[0]**2, 6*x[0]]])

    return J

def iteration2a(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = np.array([[1/6, 1/18],
                  [0, 1/6]])
    
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - J.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   

def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

#driver1()
driver3()