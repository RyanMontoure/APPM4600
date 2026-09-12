import numpy as np
import matplotlib.pyplot as plt

# driver for question 2b
def driverQ2():
    print('Question 2b:')
    A = 0.5 * np.array([[1.0, 1.0],
                       [1 + 1e-10, 1 - 1e-10]])
    # returns singular values of A in descending order
    singularvals = np.linalg.svd(A, compute_uv = False) 
    k = singularvals[0] / singularvals[1]

    print('Singular Values:', singularvals)
    print('Condition Number:', k)
    
# driver for question 4c
def driverQ4():
   print('Question 4c:')
   f = lambda x: 2*x - 1 - np.sin(x)
   a = 0
   b = 1
   tol = 1e-8

   [astar,ier] = bisection(f,a,b,tol)
   print('the approximate root is',astar)
   print('the error message reads:',ier)
   print('f(astar) =', f(astar))

# driver for question 5b
def driverQ5():
    print('Question 5b:')
    f = lambda x: x**3 + x - 4
    a = 1
    b = 4
    tol = 1e-3
   
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

# driver for question 6a
def driverQ6a():
    print('Question 6a:')
    x = np.linspace(-2, 8, 1000)
    y = x - 4*np.sin(2*x) - 3

    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('$x-4sin(2x)-3$')
    plt.axhline(0, color = 'black', linestyle = '--') # graphs the x axis
    plt.show()

# driver for question 6b
def driverQ6b():
   print('Question 6b:')
   f = lambda x: -np.sin(2*x) + 5*x/4 - 3/4

   xguesses = [-1.0, -0.5, 1.7, 3.0, 4.5] # guesses by what each root looks like
   tol = 1e-10
   Nmax = 500
   root = 1

   for x0 in xguesses:
        print ('Root', root, ':')
        [xstar,ier] = fixedpt(f,x0,tol,Nmax)
        print('the approximate fixed point is:',xstar)
        print('f2(xstar):',f(xstar))
        print('Error message reads:',ier)
        root+=1
   
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

    count = 1
    d = 0.5*(a+b)
    while (abs(d-a) > tol):
      fd = f(d)
      if (fd == 0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count + 1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    print('count = ', count)
    return [astar, ier]

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count < Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) / abs(x1) < tol):  # checking if relative error is within the tolerance
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]


driverQ4()
driverQ5()
driverQ6a()
driverQ6b()
driverQ2()