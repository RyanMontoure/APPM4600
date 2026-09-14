import numpy as np

def driver():
    Nmax = 100
    #tol = 1e-6
    #x0 = 0.0

    # f1 = lambda x: 1+0.5*np.sin(x)
    # testFunction(f1, Nmax, tol, x0)
    # orderofConvergence(f1, Nmax, tol, x0)

    # QUESTION 2.2
    g = lambda x: (10 / (x + 4)) ** 0.5
    x0 = 1.5
    tol = 1e-10

    #testFunction(g, Nmax, tol, x0)
    fixedptdata = fixedpt(g, x0, tol, Nmax)
    print ('It took', fixedptdata[2].size, 'iterations to converge')
    lambdA = orderofConvergence(g, Nmax, tol, x0)
    print ('Lambda =', lambdA[1])

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
          return [xstar,ier, x[:count + 1].flatten()] # removes extra 0's from x array
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, x[:count + 1].flatten()]

def testFunction(f, Nmax, tol, x0): # runs a function through  the fixedpt algorithm
    [xstar,ier,approx] = fixedpt(f,x0,tol,Nmax)
    print('approximations:',approx)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f(xstar))
    print('Error message reads:',ier)

#testFunction(f1, Nmax, tol, x0)

def orderofConvergence(f, Nmax, tol, x0):
    [p, ier, pv] = fixedpt(f, x0, tol, Nmax) # pv is the vector of all approximations

    if (ier == 1):
        return ('bad fixedpt approximation')

    # np.diff creates a new vector with the difference of consecutive terms
    epsilon = np.abs(np.diff(pv))

    # calculates alpha
    numerator = np.log(epsilon[2:] / epsilon[1:-1])
    denominator = np.log(epsilon[1:-1] / epsilon[:-2])

    alphaV = numerator / denominator
    alpha = round(alphaV[-1])

    # calculate lambda using alpha
    lambdaV = epsilon[1:] / epsilon[:-1] ** alpha
    lambdA = lambdaV[-1]

    # determine convergence
    if ((alpha == 1) and (lambdA < 1)):
        print ('The sequence converges LINEARLY')
        return(alpha, lambdA)
    elif (alpha == 2):
        print ('The sequence converges QUADRATICALLY')
        return(alpha, lambdA)
    print ('The sequence did not converge?')
    return


driver()