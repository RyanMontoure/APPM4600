import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# driver for question 2
def driver2():
    # QUESTION 2a
    # constants given in question
    t = 5184000
    Ts = -15.0
    Ti = 20.0
    alpha = 0.138e-6

    xvals = np.linspace(0, 2, 500)
    # this goes inside erf function, given in problem
    erfVar = xvals / (2 * np.sqrt(alpha * t))

    fVector = erf(erfVar) * (Ti - Ts) + Ts # f(x) as a vector

    plt.plot(xvals, fVector)
    plt.xlabel('Depth (meters)')
    plt.ylabel('Temperature (degrees Celcius)')
    plt.grid(True)
    plt.show()

    # QUESTION 2b
    tol = 1e-13

    f = lambda x: erf(x / (2 * np.sqrt(alpha  *t))) * (Ti - Ts) + Ts # f(x) as a lambda function

    [astar, ier] = bisection(f, 0, 2, tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

    # QUESTION 2c
    print('\n')
    Nmax = 500

    fp = lambda x: ((Ti - Ts) / np.sqrt(np.pi * alpha * t)) * np.exp((-x**2) / (4 * alpha * t)) # f'(x)

    (p, pstar, info, it) = newton(f, fp, 0.01, tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

# driver for question 3
def driver3():
    x0 = 1
    tol = 1e-10
    Nmax = 1000

    fa = lambda x: x * (1 + (7 - x**5) / x**2)**3
    fb = lambda x: x - (x**5 - 7) / x**2
    fc = lambda x: x - (x**5 - 7) / (5 * x**4)
    fd = lambda x: x - (x**5 - 7) / 12

    functions = [fa, fb, fc, fd]

    for f in functions:
        [xstar, ier, count] = fixedpt(f, x0, tol, Nmax)

        if (ier == 1):
           print('Failed / Diverged')
        else:
            print('the approximate fixed point is:',xstar)
            print('f1(xstar):', f(xstar))
            print(count,'iterations')
        print('\n')

def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    print('count = ', count)
    return [astar, ier]

def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
        count = count + 1
        # added this because some functions diverged way faster than others
        try:
            x1 = f(x0)
        except OverflowError:
            print('Diverged due to numerical overflow.')
            return None, 1, 0
        
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier, count]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, count]

#driver2()
driver3()