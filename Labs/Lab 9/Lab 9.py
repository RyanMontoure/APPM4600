import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad



def eval_legendre(n, x):
    p = np.zeros(n + 1)
    p[0] = 1.0
    if n == 0:
        return p
    
    p[1] = float(x)
    for k in range(2, n + 1):
        p[k] = ((2 * k - 1) * x * p[k - 1] - (k - 1) * p[k - 2]) / k
        
    return p

def eval_phifw(x, phi, f, w):
    return phi(x) * f(x) * w(x)

def eval_phiphiw(x, phi, w):
    return (phi(x)**2) * w(x)

def eval_coefficient(phi_j, f, a, b, w):
    numer,_ = quad(eval_phifw, a, b, args=(phi_j, f, w))
    denom,_ = quad(eval_phiphiw, a, b, args=(phi_j, w))
    return numer / denom


def driver():

#  function you want to approximate
    f = lambda x: 1 / (1 + x**2)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()
    
      
    

def eval_legendre_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
  p = eval_legendre(n, x)
  # initialize the sum to 0 
  pval = 0.0    
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_legendre(j,x)[j]
      # make a function handle for evaluating phi_j^2(x)*w(x)
      # use the quad function from scipy to evaluate coeffs
      aj = eval_coefficient(phi_j, f, a, b, w)
      # accumulate into pval
      pval = pval+aj*p[j]
       
  return pval
    
driver()