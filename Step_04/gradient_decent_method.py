'''
Gradient Decent method
x_(k+1) = x_(k) - lr * grad(f(x_(k)))

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1
def f(x):
    return x ** 4 - 3 * x ** 3 + 2

def fprime(x): # forward difference
    h = 0.001
    return (f(x+h) - f(x)) / h

# 2 
k = 0               # iteration #
max_iters = 1000
lr = 0.001          # learning rate
tol = 1e-5

x_old = 0.0
x_new = 4.0             # -2.0
x_list = [x_new]        # list of x_new
x = tf.Variable(x_new, dtype = np.float32) # initial value

while abs(x_old - x_new) > tol and k < max_iters:
    k += 1
    x_old = x.numpy()
    step = lr * fprime(x) # how much to move in gradient's direction
    x.assign_sub(step, read_value = False) # update value by gradient decent method
    x_new = x.numpy() 
    x_list.append(x_new)
    
    # print('k = {}: f({}) = {}'.format(k, x_new, f(x_new)))
print('k = {} : f({}) = {}'.format(k, x_new, f(x_new)))

# 3 : check solutions
print('[f(0), f(9/4), f(-2), f(4)] =', [f(0), f(9/4), f(-2), f(4)])
# [f(0), f(9/4), f(-2), f(4)]: [2, -6.54296875, 42, 66]

# 4 : draw graph
# 4-1 : graph(x)
## x_values = np.linspace(-2.0, 4.0, num = 101) # numpy array
xs = tf.linspace(-2.0, 4.0, num = 101)          # tensor
ys = f(xs)
plt.plot(xs, ys, 'b-')

# 4-2 : f(x_new), updated solutions
## x_list = np.array(x_list)                        # numpy array
x_list = tf.constant(x_list, dtype = np.float32)    # tensor
y_list = f(x_list)
plt.plot(x_list, y_list, 'ro')
plt.show()