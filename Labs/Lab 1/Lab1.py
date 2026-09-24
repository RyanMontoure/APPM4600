import math
import numpy as np
import matplotlib.pyplot as plt


x = [1, 2, 3]
y = np.array([1, 2, 3])
print('This is 3y', 3*y)

# creates a vector [0 - 2pi] with 100 values
X = np.linspace(0, 2 * np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)

plt.plot(X, Ya)
plt.plot(X, Yb)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = np.linspace(0, 100, 101)
y = np.arange(0, 100)

print('The first 3 entries of x are:', x[0], x[1], x[2])

# w is a vector containing 10^-n where n is integers 1-10
w = 10**(-np.linspace(1,10,10))
x = np.linspace(1, 10, 10)
plt.plot(x, w)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# matrix multiplication:
def driver4_2():
    matrix1 = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,0]])
    matrix2 = np.array([[1,2,3],
                       [4,5,6],
                       [7,8,9]])
    n = 3
    result = matrixMult(matrix1, matrix2, n)
    print('The matrix is: ', result)
    return

def matrixMult(a, b, n):
    result = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            result[i,j] = dotProduct(a[i], b[:, j], n)

    return result

def dotProduct(x,y,n):
#   Computes the dot product of the n x 1 vectors x and y
     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  

driver4_2()