import matplotlib.pyplot as plt
import numpy as np

"""
#1) 
i. Plot P(x) given the polynomial by evaulating int's coefficients
ii. Plot P(x) given it's expression
"""
x = np.arange(1.920, 2.081, 0.001)

p = (x**9)-(18*x**8)+(144*x**7)-(672*x**6)+(2016*x**5)-(4032*x**4)+(5376*x**3)-(4608*x**2)+(2304*x)-512
plt.plot(x,p, label = 'Expanded Function')

f = (x-2.0)**9
plt.plot(x,f, label = '$(X-2)^9$')

plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()

plt.show()

"""
iii. What is the difference? What is causing the discrepancy? Which plot is correct?

The difference in the functions is that one is in factored form and the longer function
is in expanded form. The difference in graphs is that the factored form is a smooth curve
while the expanded form is rough and jagged but still following the same general curve. 

The discrepency is caused by floating point arithmatic error since the expanded function
has 10 terms, all with exponents and multiplication, causing a lot of accuracy to be dropped.
You can see the line is especially rough near y = 0.0. This is where the true y values are 
the smallest and the error is the largest. 

The factored form plot is more numerically accurate since it is not losing as much accuracy
with floating point arithmatic. 
"""