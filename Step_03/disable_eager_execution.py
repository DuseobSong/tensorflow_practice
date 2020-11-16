# eager execution is activated automatically with 2.x versions.(default)

from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
print(tf.executing_eagerly())

# eager_execution is disabled. (same as 1.x versions)
disable_eager_execution()
print(tf.executing_eagerly())

#1 graph construction
a = tf.constant(1)
b = tf.constant(2)

c = a + b # c = tf.add(a,b)

print(a)
print(b)
print(c)

# graph execution
sess = tf.compat.v1.Session()
sess.run(a)
sess.run(b)
sess.run(c)

sess.close()

'''
# tensorflow 2.x version : eager_execution is activated

from tensorflow.pytho.framework.ops import enable_eager_execution
import tensorflow as tf
enable_eager_execution()
print(tf.executing_eagerly())

a = tf.constant(1)
b = tf.constant(2)

c = a + b # c = tf.add(a,b)

a
b
c

a.numpy(), b.numpy(), c.numpy()

'''