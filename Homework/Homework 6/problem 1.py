import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 
import matplotlib.pyplot as plt

'''-------- DRIVER --------'''
# driver
def driver():
    tol = 1e-4
    Nmax = 100

    initialVals = [np.array([-0.5, 0.25]), 
                   np.array([-1.5, 1.25]), 
                   np.array([-3, 3])]

    # fixed point method
    print('FIXED POINT METHOD:')
    for x0 in initialVals:
        print('X0 =', x0)
        t = time.time()
        for j in range(50):
            [xstar,ier,its] =  fixedSys(x0,tol,Nmax)
        elapsed = time.time()-t
        print(xstar)
        print('Fixed pt: the error message reads:',ier) 
        print('Fixed pt: took this many seconds:',elapsed/50)
        print('Fixed pt: number of iterations is:',its)

    # newton's method
    print('\nNEWTONS METHOD:')
    for x0 in initialVals:
        print('X0 =', x0)
        t = time.time()
        for j in range(50):
            [xstar,ier,its] =  Newton(x0,tol,Nmax)
        elapsed = time.time()-t
        print(xstar)
        print('Newton: the error message reads:',ier) 
        print('Newton: took this many seconds:',elapsed/50)
        print('Netwon: number of iterations is:',its)


'''-------- FUNCTION EVALUATORS --------'''
# g function given
def evalG(x):
    m = np.array([[0.016, -0.17],
                  [0.52, -0.26]])
    
    v = np.array([3*x[0]**2 + 4*x[1]**2 -1, 
                  x[1]**3 - 8*x[0]**3 - 1])
    
    g = np.array([x[0], x[1]]) - m @ v

    return g

# f function given
def evalF(x):
    f = np.array([3*x[0]**2 + 4*x[1]**2 - 1,
                  x[1]**3 - 8*x[0]**3 - 1])
    return f

# jacobian of f function
def evalJ(x):
    j = np.array([[6*x[0], 8*x[1]],
                  [-24*x[0]**2, 3*x[1]**2]])
    return j


'''-------- ROOT FINDING METHODS --------'''
# fixed point iteration for systems
def fixedSys(x0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       
       x1 = evalG(x0)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

# newtons method for a system
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


driver()