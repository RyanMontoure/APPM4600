from gauss_legendre import *
import numpy as np

def driver():
    a = 0.1
    b = 2
    f = lambda x: np.sin(1/x)
    tol = 1e-3
    N = 5
    
    intTrap, xTrap, nTrap = adaptive_quad(a, b, f, tol, N, eval_composite_trap)
    intSimp, xSimp, nSimp = adaptive_quad(a, b, f, tol, N, eval_composite_simpsons)
    intGuass, xGuass, nGuass = adaptive_quad(a, b, f, tol, N, eval_gauss_quad)

    print ('COMPOSITE TRAPEZOIDAL:')
    print('Integral =', intTrap)
    print('# of Intervals:', nTrap, '\n')

    print ('COMPOSITE SIMPSONS:')
    print('Integral =', intSimp)
    print('# of Intervals:', nSimp, '\n')

    print ('COMPOSITE GUASSIAN:')
    print('Integral =', intGuass)
    print('# of Intervals:', nGuass, '\n')


# Evaluates Composite Tapezoidal Integration
# interval [a, b], function f(x), N intervals
def eval_composite_trap(a, b, f, N):
    h = (b-a)/N
    x = np.linspace(a,b,N+1)
    sum = 0

    for j in range(1, N, 1):
        xj = a + (j*h)
        sum += f(xj)

    integral = (h/2) * (f(a) + f(b) + (2*sum))
    return (integral, x, None)

# Evaluates Composite Simpson's Integration
# interval [a, b], function f(x), N intervals
def eval_composite_simpsons(a, b, f, N):
    # ensures an even number of intervals
    if N%2 != 0:
       N += 1

    h = (b-a)/N
    x = np.linspace(a,b,N+1)
    sum1 = 0

    for j in range(2, N, 2):
        x2j = a + (j*h)
        sum1 += f(x2j)

    sum2 = 0
    for j in range(1, N, 2):
        x2j1 = a + (j*h)
        sum2 += f(x2j1)

    integral = (h/3) * (f(a) + 2*sum1 + 4*sum2 + f(b))
    return (integral, x, None)

def eval_gauss_quad(a,b,f,M):
  r"""
  Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
  Input:
    M - number of quadrature nodes
    a,b - interval [a,b]
    f - function to integrate
  
  Output:
    I_hat - approx integral
    x - quadrature nodes
    w - quadrature weights

  Currently uses Gauss-Legendre rule
  """
  x,w = lgwt(M,a,b)
  I_hat = np.sum(f(x)*w)
  return I_hat,x,w

def adaptive_quad(a,b,f,tol,M,method):
  r"""
  Adaptive numerical integrator for \int_a^b f(x)dx
  
  Input:
  a,b - interval [a,b]
  f - function to integrate
  tol - absolute accuracy goal
  M - number of quadrature nodes per bisected interval
  method - function handle for integrating on subinterval
         - eg) eval_gauss_quad, eval_composite_simpsons etc.
  
  Output: I - the approximate integral
          X - final adapted grid nodes
          nsplit - number of interval splits
  """
  # 1/2^50 ~ 1e-15
  maxit = 50
  left_p = np.zeros((maxit,))
  right_p = np.zeros((maxit,))
  s = np.zeros(maxit)
  left_p[0], right_p[0] = a, b
  # initial approx and grid
  s[0], x, _ = method(a,b,f,M)
  # save grid
  X = []
  X.append(x)
  j = 1
  I = 0
  nsplit = 1
  while j < maxit:
    # get midpoint to split interval into left and right
    c = 0.5*(left_p[j-1]+right_p[j-1])
    # compute integral on left and right spilt intervals
    s1,x,_ = method(left_p[j-1],c,f,M); X.append(x)
    s2,x,_ = method(c,right_p[j-1],f,M); X.append(x)
    if np.max(np.abs(s1+s2-s[j-1])) > tol:
      left_p[j] = left_p[j-1]
      right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j] = s1
      left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j-1] = s2
      j = j+1
      nsplit = nsplit+1
    else:
      I = I+s1+s2
      j = j-1
      if j == 0:
        j = maxit
  return I,np.unique(np.concatenate(X)),nsplit



driver()