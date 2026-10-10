import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 
import matplotlib.pyplot as plt

#driver
def driver():
    tol = 1e-10
    Nmax = 100

    x0 = np.array([0.5, 0.5, 0.5])

    # newton's method
    print('\nNEWTONS METHOD:')
    t = time.time()
    for j in range(50):
        [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print('Solution', xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)

    # steepest descent method
    print('\nSTEEPEST DESCENT:')
    t = time.time()
    for j in range(50):
        [xstar,gval,ier,its] = SteepestDescent(x0,tol,Nmax)
    elapsed = time.time()-t
    print("Solution:",xstar)
    print("g evaluated at this point is ", gval)
    print("Error message: ", ier)
    print('Number of iterations:', its)
    print('Seconds:', elapsed/50)

    # hybrid method
    print('\nHYBRID METHOD')
    t = time.time()
    for j in range(50):
        [xstar, ier, its] = hybrid(x0, tol, Nmax)
    elapsed = time.time()-t
    print('Solution: ', xstar)
    print('Error Message: ', ier)
    print('Number of iterations:', its)
    print('Seconds:', elapsed/50)



 
'''-------- FUNCTION EVALUATORS --------'''
# function given
def evalF(x):
    f = np.array([x[0] + np.cos(x[0]*x[1]*x[2]) - 1,
                  (1-x[0])**0.25 + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1,
                  -(x[0]**2) - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1])
    return f

# jacobian of function
def evalJ(x):
    j = np.array([[1 - x[1]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
                  [-0.25*(1-x[0])**-0.75, 1, 0.1*x[2] - 0.15],
                  [-2*x[0], -0.2*x[1] + 0.01, 1]])
    return j

# least squares function to minimize for steepest descent
def evalG(x):
    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

# gradient of least squares function
def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)

    gradg = np.transpose(J).dot(F)
    return gradg




'''-------- ROOT FINDING METHODS --------'''
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

# steepest descent method
def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalG(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalG(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalG(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier,its]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalG(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalG(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier,its]

    #print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier,its]


def hybrid(x0, tol, Nmax):
    [x, gval, ier, itsSD] = SteepestDescent(x0, 5e-2, Nmax)
    if ier == 0:
        [x, ier, itsN] = Newton(x, tol, Nmax)
        return [x, ier, itsSD + itsN]
    return None

driver()