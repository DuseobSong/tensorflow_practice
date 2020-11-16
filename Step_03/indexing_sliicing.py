import tensorflow as tf
a = tf.constant(1) # constant
b = tf.constant([1,2,3,4]) # 1-dimensioinal tensor
c = tf.constant([[1,2,],[3,4]]) # 2-dimensional tensor
d = tf.constant([[[1,2],[3,4]]]) # 3-dimensional tensor

a 
a.dtype
a.ndim, b.ndim, c.ndim, d.ndim # 

a.shape, b.shape, c.shape, d.shape

# indexing, slicing
b[0]   # indexing an element from 1-dimensional tensor
b[:2]  # slicing elements from 1-dimensional tensor
c[0,0] # indexing an element from 2-dimensional tensor
c[:,0] # slicing elements from 2-dimensional tensor

