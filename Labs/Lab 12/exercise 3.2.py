import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

    ''' create  matrix for testing different ways of solving a square 
    linear system'''

    '''' N = size of system'''
    for N in [100, 500, 1000, 2000, 4000, 5000]:
        print('N =', N,':')

        ''' Right hand side'''
        b = np.random.rand(N,1)
        A = np.random.rand(N,N)

        start = time.perf_counter()
        x = scila.solve(A,b)
        end = time.perf_counter()
    
        test = np.matmul(A,x)
        r = la.norm(test-b)
     
        #print('SCIPY Solve:',r)
        print('SCIPY Solve Time:', end-start)

        ''' solving via LU factorization '''
        start1 = time.perf_counter()
        lu, piv = scila.lu_factor(A)
        end1 = time.perf_counter()

        start2 = time.perf_counter()
        x = scila.lu_solve((lu, piv), b)
        end2 = time.perf_counter()

        test = np.matmul(A,x)
        r = la.norm(test-b)

        #print('LU Solve:',r)
        print('LU Factor Time:', end1-start1)
        print('LU Solve Time:', end2-start2)
        print('LU Total Time:', (end2-start2) + (end1-start1))

    # testing how many b's LU is faster
    # N = 100
    # A = np.random.rand(N,N)
    # b = np.random.rand(N,1)

    # start = time.perf_counter()
    # x = scila.solve(A,b)
    # end = time.perf_counter()
    # time2 = end-start
    
    # start1 = time.perf_counter()
    # lu, piv = scila.lu_factor(A)
    # x = scila.lu_solve((lu, piv), b)
    # end1 = time.perf_counter()
    # time1 = end-start
    
    # count = 1
    # while(time1<time2):
    #     count += 1
    #     b = np.random.rand(N,1)

    #     start = time.perf_counter()
    #     x = scila.solve(A,b)
    #     end = time.perf_counter()
    #     time2 = end-start

    #     start1 = time.perf_counter()
    #     lu, piv = scila.lu_factor(A)
    #     x = scila.lu_solve((lu, piv), b)
    #     end1 = time.perf_counter()
    #     time1 = end-start

    #print (count)





     
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
