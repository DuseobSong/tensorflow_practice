# required tensorflow version : 2.x 
# current version : 1.3.1 (2020/11/12)-> upgrade required !!!

import tensorflow as tf

# step 1 Define variables
a = tf.Variable(1)
b = tf.Variable([1,2,3,4])
c = tf.Variable([[1,2],[3,4]])
d = tf.Variable([[[1,2],[3,4]]])

a.dtype

a.shape, b.shape, c.shape, d.shape

# step 2
a

a.trainable

#3 indexing and slicing
b[0]
b[:2]
c[0,0]
c[:,0]

#4 assign, assign_add, assign_sub
id(a) # address of variable 'a'

a.assign(10) # change the vlaue of a to 10
# a.assign(20, read_value = False) # return no value

a.assign_add(20) # add 20 to a

a.assign_sub(10) # subtract 10 from a

id(a) # the address of a is not changed

