#%%
import numpy
import scipy.optimize as optimization
#from IPython.core.debugger import set_trace

# Data
xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])
# Initial guess.
x0 = numpy.array([0.0, 0.0, 0.0])
# Objective fucntion
def func(x, a, b, c):
    rval = a + b*x + c*x*x
    return rval

result = optimization.curve_fit(func, xdata, ydata, x0)
#set_trace()
print(result[0])


