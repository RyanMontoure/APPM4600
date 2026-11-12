import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():
    ''' Create an ill-conditioned rectangular matrix '''
    N = 10
    M = 5
    A = create_rect(N,M)     
    b = np.random.rand(N,1)

    # Solving with QR Factorization
    Q, R = np.linalg.qr(A)
    y = np.matmul(np.transpose(Q), b)
    x = scila.solve(R,y)
    print(x)

    # Solving with Normal Equation
    At = np.transpose(A)
    AtA = np.matmul(At, A)
    y = np.matmul(At, b)
    x = scila.solve(AtA, y)
    print(x)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       