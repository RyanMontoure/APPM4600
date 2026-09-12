import numpy as np

def fixedpt(f,x0,tol,Nmax): # fixed point algorithm

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    x = np.zeros((Nmax, 1)) # vector to hold all of the approximations
    count = 0
    while (count <Nmax):
       x[count] = x0  # add the current x to the array
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          x[count] = x1 # add the final approximation to the array
          xstar = x1
          ier = 0
          return [xstar,ier, x]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, x]

f1 = lambda x: 1+0.5*np.sin(x)
Nmax = 100
tol = 1e-6
x0 = 0.0

def testFunction(f, Nmax, tol, x0): # runs a function through  the fixedpt algorithm
    [xstar,ier,approx] = fixedpt(f1,x0,tol,Nmax)
    print('approximations:',approx)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)

#testFunction(f1, Nmax, tol, x0)

def orderofConvergence(f, Nmax, tol, x0):
    [p, ier, pv] = fixedpt(f, x0, tol, Nmax) # pv is the vector of all approximations

    if (ier == 1):
        return ('bad fixedpt approximation')

    