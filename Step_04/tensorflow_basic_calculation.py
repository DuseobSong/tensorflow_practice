import tensorflow as tf

#1 : shape = (2,)
a = tf.constant([1,2])

print(a + 1) # tf.add(a,1) or tf.math.add(a,1)

print(a - 1) # tf.subtract(a, 1)

print(a * 2) # tf.multiply(a, 2)

print(a / 2) # tf.devide(a, 2)


# 2 : shape = (2,)
b = tf.constant([3,4])
print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 3 : shape = (2,2)
a = tf.constant([[1,2],[3,4]])
b = tf.constant([1,2])

print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 4
a = tf.reshape(tf.range(12), shape = (3,4)) 
print(a)

# 5 : minimum value in each direction(0: column, 1: row)
k = tf.reduce_min(a)
print(k)

k = tf.reduce_min(a, axis = 0)
print(k)

k = tf.reduce_min(a, axis = 1)
print(k)

# 6
k = tf.reduce_max(a)
print(k)

k = tf.reduce_max(a, axis = 0)
print(k)

k = tf.reduce_max(a, axis = 1)
print(k)

# 7 : sum of each row or column
k = tf.reduce_sum(a)
print(k)

k = tf.reduce_sum(a, axis = 0)
print(k)

k = tf.reduce_sum(a, axis = 1)
print(k)

# 8 mean/product
k = tf.reduce_prod(a)
print(k)

k = tf.reduce_prod(a, axis = 0)
print(k)

k = tf.reduce_prod(a, axis = 1)
print(k)

# 9 : index of minimum or maximum element in each row or column
a = tf.reshape(tf.random.shuffle(tf.range(12)), shape = (3,4))

print(tf.argmin(a)) # tf.argmin(a, axis = 0)

print(tf.argmin(a, axis = 1))

print(tf.argmax(a)) # default axis : 0

print(tf.argmax(a, axis = 1))

# 10
a = tf.random.shuffle(tf.range(12))

tf.sort(a) # direction = ASCENDING
print(a)

tf.sort(a, direction = "DESCENDING")
print(a)

a = tf.reshape(a, shape = (3,4))
print(a)

tf.sort(a) # tf.sort(a, axis = 1)
print(a)

tf.sort(a, axis = 0)
print(a)


# linear algebra
# 1
a = tf.constant([1,2,3], dtype = tf.float32)

print(tf.norm(a)) # tf.linalg.norm(a): length of 1-dimensional tensor

# 2: transpose
A = tf.constant([[1,2],[3,4]], dtype = tf.float32)
print(tf.linalg.matrix_transpose(A))

# 3: determinant, inverse, dot product
print(tf.linalg.det(A))

B = tf.linalg.inv(A)
print(B)

print(tf.matmul(A, B))