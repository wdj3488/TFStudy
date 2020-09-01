import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def accuracy(output, target, topk=[1, ]):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred)
    target = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target)
    res = []
    for k in topk:
        correct_k = tf.cast(correct[:k], dtype=tf.int32)
        correct_k = tf.reduce_sum(correct_k)
        acc = correct_k/batch_size
        res.append(acc)
    return res


output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], minval=0, maxval=6, dtype=tf.int32)
print('prob:', output)
pred = tf.argmax(output, axis=1)
print('pred:', pred)
print('label:', target)

acc = accuracy(output, target, topk=[1, 2, 3, 4, 5, 6])
print('top-1-6 acc:', acc)
