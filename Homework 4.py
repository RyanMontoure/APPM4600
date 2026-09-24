import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# driver for question 3
def driver3():
    f = lambda x: (np.exp(x) - 3*x**2)**3 # f(x) factored
    fp = lambda x: 3*((np.exp(x) - 3*x**2)**2) * (np.exp(x) - 6*x) # f'(x) factored
    g = lambda x: (np.exp(x) - 3*x**2) / (3 * (np.exp(x) - 6*x)) # g(x) factored
    gp = lambda x: (np.exp(x)*(x**2 - 4*x + 2) + 6*x**2) / ((np.exp(x)) - 6*x)**2 #g'(x) factored
    p0 = 4
    tol = 1e-10
    Nmax = 1000

    # regular Newton's Method
    [p1, pstar1, info1, it1] = newton(f, fp, p0, tol, Nmax)
    (alpha1, lambdA1) = orderofConvergence(p1, p0, info1, tol, Nmax)
    print('Approximation 1:', pstar1)
    if ((alpha1 == 1) and lambdA1 < 1):
        print('Converges Linearly')
    elif (alpha1 == 2):
        print('Converges Quadratically')
    else:
        print('did not converge')

    # In Class Method (using g(x) = f(x)/f'(x))
    [p2, pstar2, info2, it2] = newton(g, gp, p0, tol, Nmax)
    (alpha2, lambdA2) = orderofConvergence(p2, p0, info2, tol, Nmax)
    print('Approximation 2:', pstar2)
    if ((alpha2 == 1) and lambdA2 < 1):
        print('Converges Linearly')
    elif (alpha2 == 2):
        print('Converges Quadratically')
    else:
        print('did not converge')

    # Method From Problem 1
    m = 3 # multiplicity of root of function
    [p3, pstar3, info3, it3] = newtonmod1(f, fp, p0, m, tol, Nmax)
    (alpha3, lambdA3) = orderofConvergence(p3, p0, info3, tol, Nmax)
    print('Approximation 3:', pstar3)
    if ((alpha3 == 1) and lambdA3 < 1):
        print('Converges Linearly')
    elif (alpha3 == 2):
        print('Converges Quadratically')
    else:
        print('did not converge')


# driver for question 4
def driver4():
    tol = 1e-10
    Nmax = 1000
    f = lambda x: x**6 - x - 1
    fp = lambda x: 6*x**5 - 1
    x0 = 2
    x1 = 1

    [p1, pstar1, info1, it1, error1] = newtonWError(f, fp, x0, tol, Nmax)
    [pstar2, error2, ier2, p2] = secant(f, x0, x1, tol, Nmax)

    df_newton = pd.DataFrame({'Iteration (k)': np.arange(len(error1)), 'Newton Error': error1})
    df_secant = pd.DataFrame({'Iteration (k)': np.arange(len(error2)), 'Secant Error': error2})

    df_table = pd.merge(df_newton, df_secant, on='Iteration (k)', how='outer')

    print("--- Root Approximations ---")
    print(f"Newton Root Approximation: {pstar1:.10f}")
    print(f"Secant Root Approximation: {pstar2:.10f}\n")
    
    print("--- Error Table ---")
    print(df_table.to_string(index=False))

    e_newton = error1[error1 > 1e-15]
    e_secant = error2[error2 > 1e-15]

    newton_x, newton_y = e_newton[:-1], e_newton[1:]
    secant_x, secant_y = e_secant[:-1], e_secant[1:]

    plt.figure(figsize=(8, 6))
    plt.loglog(newton_x, newton_y, label="Newton's Method")
    plt.loglog(secant_x, secant_y, label="Secant Method")
    plt.grid(True)
    plt.legend()
    plt.show()





# order of convergence function I made in lab 3
def orderofConvergence(p, x0, ier, tol, Nmax):
    if (ier == 1):
        return ('bad fixedpt approximation')

    # np.diff creates a new vector with the difference of consecutive terms
    epsilon = np.abs(np.diff(p))
    epsilon = epsilon[epsilon > 1e-10] # to avoid divide by zero errors
    if (len(epsilon) < 3):
        return (2, 0.0) # returns quadratic if function converges super fast

    # calculates alpha
    numerator = np.log(epsilon[2:] / epsilon[1:-1])
    denominator = np.log(epsilon[1:-1] / epsilon[:-2])

    alphaV = numerator / denominator
    alpha = round(alphaV[-1])

    # calculate lambda using alpha
    lambdaV = epsilon[1:] / epsilon[:-1] ** alpha
    lambdA = lambdaV[-1]

    # determine convergence
    # if ((alpha == 1) and (lambdA < 1)):
    #     print ('The sequence converges LINEARLY')
    #     return(alpha, lambdA)
    # elif (alpha == 2):
    #     print ('The sequence converges QUADRATICALLY')
    #     return(alpha, lambdA)
    # print ('The sequence did not converge?')
    # return
    return(alpha, lambdA)
   

# regular Newton's Method
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
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p[:it+2],pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p[:it+2],pstar,info,it] # returns p with non-empty elements


# Modified Newton's From Problem 1
def newtonmod1(f, fp, p0, m, tol, Nmax): # I added m as an input paramater
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
  for it in range(Nmax):
      p1 = p0 - m*f(p0)/fp(p0)  # changed the iteration to Xn - mf(Xn)/f'(Xn)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p[:it+2],pstar,info,it] # returns p with only non-empty elements
      p0 = p1
  pstar = p1
  info = 1
  return [p[:it+2],pstar,info,it] # returns p with only non-empty elements


# Secant Method
def secant(f, x0, x1, tol, Nmax):
    history = np.zeros(Nmax)

    f0 = f(x0)
    f1 = f(x1)
    x2 = x1

    for i in range(Nmax):
        if np.abs(f1 - f0) == 0:
            if f1 == 0:
                ier = 0
                error = np.abs(pstar - history[:i])
                return [x1, error, ier, history[:i]]
            ier = 1
            error = np.abs(x1 - history[:i])
            return [None, error, ier, history[:i]]

        x2 = x1 - f1*(x1-x0)/(f1-f0) # secant update
        history[i] = x2

        if np.abs(x2-x1) < tol: # check convergence
            pstar = x2
            ier = 0
            error = np.abs(pstar - history[:i+1])
            return [pstar, error, ier, history[:i+1]]
        
        x0, f0 = x1, f1 # update values
        x1 = x2
        f1 = f(x1)

    ier = 1
    pstar = x2
    error = np.abs(pstar - history)
    return [pstar, error, ier, history]

# Newton's method with an array for the error
# I made a new one so I didn't have to change my code for problem 3
def newtonWError(f,fp,p0,tol,Nmax):
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
  history = np.zeros(Nmax)
  
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          error = np.abs(pstar - p[:it+2])
          return [p[:it+2],pstar,info,it, error]
      p0 = p1
  pstar = p1
  info = 1
  error = np.abs(pstar-p[:it+2])
  return [p[:it+2],pstar,info,it, error] # returns p with non-empty elements


#driver3()
driver4()