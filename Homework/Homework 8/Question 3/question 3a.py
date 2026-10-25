import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

def driver():
    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    for N in [5, 10, 15, 20]:
        ''' interval'''
        a = -5
        b = 5
   
        ''' create Chebyshev interpolation nodes on [a, b] '''
        k = np.arange(1, N + 2)
        x_hat = np.cos((2*k - 1) * np.pi / (2*(N + 1)))  # Chebyshev nodes on [-1, 1]
        xint = 0.5 * (a + b) + 0.5 * (b - a) * x_hat       # Map to interval [a, b]
    
        ''' create interpolation data'''
        yint = np.zeros(N+1)
        ypint = np.zeros(N+1)
        for jj in range(N+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
    
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yevalL = np.zeros(Neval+1)
        yevalH = np.zeros(Neval+1)
        for kk in range(Neval+1):
            yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)

        ''' create vector with exact values'''
        fex = np.zeros(Neval+1)
        for kk in range(Neval+1):
            fex[kk] = f(xeval[kk])
    
    
        plt.figure()
        plt.plot(xeval,fex,label = 'data')
        plt.plot(xeval,yevalH,label='Hermite')
        plt.title(f'N = {N}')
        plt.legend()
        plt.semilogy()
        plt.show()
         
    # errL = abs(yevalL-fex)
    # errH = abs(yevalH-fex)
    # plt.figure()
    # plt.semilogy(xeval,errL,'bs--',label='Lagrange')
    # plt.semilogy(xeval,errH,'c.--',label='Hermite')
    # plt.show()            


def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       
    

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
