import tensorflow as tf

print(tf.test.is_gpu_available())
a = tf.constant(2.)
b = tf.constant(4.)
print(a*b)

# import numpy as np
# print(np.array([0, 10]))