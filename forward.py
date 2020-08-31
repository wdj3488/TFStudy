import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# a = tf.fill([2, 2], 2.0)
# b = tf.ones([2, 2])
# # 对应位置相乘
# print(a*b)
# print(tf.matmul(a, b))
#
# # ln 以e为底的对数运算
# print(tf.math.log(a))
#
# print(tf.math.exp(a))

(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, tf.float32)/255.
y = tf.convert_to_tensor(y, tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
# sample[0] 为x  sample[1]为标签值
print('batch:', sample[0].shape, sample[1].shape)


w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

learning_rate = 1e-3

for epoch in range(10):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:
            y = tf.one_hot(y, 10)
            h1 = tf.matmul(x, w1) + b1
            h1 = tf.nn.relu(h1)

            h2 = tf.matmul(h1, w2) + b2
            h2 = tf.nn.relu(h2)

            out = tf.matmul(h2, w3) + b3

            loss = tf.square(out - y)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(grads[0]*learning_rate)
        b1.assign_sub(grads[1]*learning_rate)
        w2.assign_sub(grads[2]*learning_rate)
        b2.assign_sub(grads[3]*learning_rate)
        w3.assign_sub(grads[4]*learning_rate)
        b3.assign_sub(grads[5]*learning_rate)

        if step%100 == 0:
            print(epoch, step, 'loss:', float(loss))
