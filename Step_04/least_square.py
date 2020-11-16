'''
Least square solution
'''

import tensorflow as tf
import matplotlib.pyplot as plt
# 1 
A = tf.constant([[0,1],
                 [1,1],
                 [2,1]], dtype = tf.float32)

b = tf.constant([[6],
                 [0],
                 [0]], dtype = tf.float32)

At = tf.transpose(A)
C = tf.matmul(At, A)

# 2 solve method
x = tf.linalg.solve(C, tf.matmul(At, b))
print('step 1. solve method')
print('Solution : \n{}'.format(x))

# 3 backward multiply
x2 = tf.matmul(tf.matmul(tf.linalg.inv(C), At), b)
print('step 2. Backward multiply')
print('solution : \n{}'.format(x2))

# 4 : LU decomposition
L_U, p = tf.linalg.lu(C)
x3 = tf.linalg.lu_solve(L_U, p, tf.matmul(At, b))
print('step 3. LU decomposition')
print('solution : \n{}'.format(x3))

# 5 : least square
x4 = tf.linalg.lstsq(A, b)
print('Step 4. Least square')
print('solution : \n{}'.format(x4))

# 6 : draw line
m,c = x.numpy()[:,0]

plt.gca().set_aspect('equal')
plt.scatter(x = A.numpy()[:,0], y = b.numpy())

t = tf.linspace(-1.0, 3.0, num = 51)
b1 = m * t + c

plt.plot(t, b1, 'b-')
plt.axis([-1, 10, -1, 10])

plt.show()