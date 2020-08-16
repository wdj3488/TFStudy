import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.test.is_gpu_available())
a = tf.constant(2.)
b = tf.constant(4.)
print(a*b)

# import numpy as np
# print(np.arange(0, 10))
