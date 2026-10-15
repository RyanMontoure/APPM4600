import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import math
import time

def driver():
    
    f = lambda x: np.exp(x)
    a = 0
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 10
    xeval =  np.linspace(a, b, Neval)
    
    ''' number of intervals'''
    Nint = 2
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    #print(yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
      
    plt.plot(xeval,fex,'ro-',label='exact function')
    for i in range(Neval):
        plt.plot(xeval,yeval[i],'bs--',label='natural spline') 
    plt.legend()
    plt.show()

     
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()

# Constructs a line (f(x)) between two points, evaluates f(alpha) 
def line(x0, y0, x1, y1, alpha):
    m = (y1 - y0) / (x1 - x0)
    f = m*(alpha-x0) + y0

    return f


def eval_lin_spline(xeval, Neval, a, b, f, Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a, b, Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        alpha = np.linspace(a1, b1, Neval//Nint)
        fb1 = f(b1)
        #print(a1, fa1, b1, fb1, xint)

        lines = np.zeros((Neval//Nint, Neval//Nint))
        for index, kk in enumerate(alpha):
           lines[index] = line(a1, fa1, b1, fb1, alpha)
           '''use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
        print(lines)
    return lines

driver()