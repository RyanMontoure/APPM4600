"""
 This script utilized the fixed point method to find the fixed
 Point of the nonlinear system of equations.
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 



import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    x0 = np.array([1.8, 1.1])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  fixedSys(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Fixed pt: the error message reads:',ier) 
    print('Fixed pt: took this many seconds:',elapsed/50)
    print('Fixed pt: number of iterations is:',its)

def evalF(x): 

    F = np.zeros(2)
    
    F[0] = x[0]-(x[0]**2+x[1]**2-5)/4
    F[1] = x[1]-(x[0]*x[1]-2)/2
 
    return F
    


def fixedSys(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       
       x1 = evalF(x0)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

driver()       
