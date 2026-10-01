import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv 
from numpy.linalg import norm 
import math

def driverPreLab():

    x0 = np.array([2, 0.5])
    x1 = np.array([3, 5])
    
    Nmax = 500
    tol = 1e-6

    print('For X0 = (2, 0.5)')

    t = time.time()
    for j in range(50):
        [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
        [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)

    print('\nFor X0 = (3, 5)')
    
    t = time.time()
    for j in range(50):
        [xstar,ier,its] =  Newton(x1,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
         
    t = time.time()
    for j in range(20):
        [xstar,ier,its] =  LazyNewton(x1,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)


def driverLab():
    x0 = np.array([0.1, 0.1, -0.1])
        
    Nmax = 500
    tol = 1e-10
    
    t = time.time() # slacker newton
    for j in range(50):
        [xstar,ier,its] =  slackerNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Slacker Newton: the error message reads:',ier) 
    print('Slacker Newton: took this many seconds:',elapsed/50)
    print('Slacker Netwon: number of iterations is:',its)
    
    t = time.time() # regular newton
    for j in range(50):
        [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)

# def evalF(x): 
# # vector function that you want to find the roots of

#     F = np.zeros(2)
    
#     F[0] = 4*x[0]**2 + x[1]**2 - 4
#     F[1] = x[0] + x[1] - np.sin(x[0] - x[1])
 
#     return F

def evalF(x):
    F = np.zeros(3)

    F[0] = 3*x[0] - np.cos(x[0]*x[2]) - 0.5
    F[1] = x[0] - 81*(x[1]+0.1)**2 + np.sin(x[2]) + 1.06
    F[2] = np.exp(-x[0]*x[1]) + 20*x[2] + (10*np.pi - 3)/3

    return F
    
# def evalJ(x): 
# # Jacobian of the vector function you want to find the roots of
    
#     J = np.array([[8*x[0], 2*x[1]], 
#         [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]])

#     return J

def evalJ(x):
    J = np.array([[3 + np.sin(x[0]*x[2])*x[2], 0, np.sin(x[0]*x[2])*x[0]],
                  [1, -162*(x[1]+0.1), np.cos(x[2])],
                  [-x[1]*np.exp(-x[0]*x[1]), -x[0]*np.exp(-x[0]*x[1]), 20]])

    return J



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
           return[xstar, ier, its+1]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its+1]
           
def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its+1]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its+1]   

def slackerNewton(x0, tol, Nmax):
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
    
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
           
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its+1]
               
        x0 = x1
        # if (np.abs(np.linalg.norm(J)) > 2): # condition when we decide to update J
        #     J = evalJ(x1)
        #     Jinv = inv(J)

        if (its % 2 == 0):
            J = evalJ(x1)
            Jinv = inv(J)
        
    xstar = x1
    ier = 1
    return[xstar, ier, its+1]  

#driverPreLab()
driverLab()