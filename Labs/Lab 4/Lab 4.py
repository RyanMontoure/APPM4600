import numpy as np

def driver():
    f = lambda x: np.e ** (x**2 + 7*x - 30) - 1 # f(x)
    fp = lambda x: (2*x + 7) * (f(x) + 1) # f'(x)
    f2p = lambda x: 2*(f(x) + 1) + (2*x + 7)*fp(x) # f''(x)
    a = 2
    b = 4.5
    tol = 1e-10
    Nmax = 100


    [bRoot, bError, bCount] = bisection(f, a, b, tol) # bisection root
    [z1, nRoot, nCount, z2] = newton(f, fp, b, tol, Nmax) # newton root
    # z variables because I only care about the root
    [hRoot, hError, hCount] = bisectionNewton(f, fp, f2p, a, b, tol, Nmax) # hybrid root

    print ('Bisection:', bRoot, bCount)
    print('Newton:', nRoot, nCount)
    print('Hybrid:', hRoot, hCount)

def bisectionNewton(f, fp, f2p, a, b, tol, Nmax):
    
#    Inputs:
#     f, fp, f2p, a,b  - function, first and 2nd derivative, and endpoints of initial interval
#     tol, Nmax        - minimum error for approximation, maximum iterations
#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    count = 0
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier = 0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    
    d = 0.5*(a+b)
    fd = f(d) # f(d)
    fdp = fp(d) # f'(d)
    fd2p = f2p(d) # f''(d)
    while (abs((fd*fd2p)/(fdp**2)) >= 1): # loops while d is not in the basin of convergence
      if (fd == 0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      fd = f(d)
      fdp = fp(d)
      fd2p = f2p(d)
      count = count +1
      
    astar = d
    ier = 0

    #print('Bisection approximated:', astar)

# Newton's Method Starts Here
    if (ier == 1):
       print('Did not converge')
       return [astar, ier, count]

    p0 = astar
    p = np.zeros(Nmax+1)
    p[0] = p0 # midpoint from bisection is initial guess
    for it in range(Nmax):
        count += 1
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [pstar, ier, count]
        p0 = p1
    pstar = p1
    info = 1
    #print(p)
    return [pstar, ier, count]


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
    count = 0
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]
    
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
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
    #print('count = ', count)
    return [astar, ier, count]

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
  count = 0
  for it in range(Nmax):
      count += 1
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,count,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,count,it]


driver()