import tensorflow as tf
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(3.)
w2 = tf.constant(4.)
b2 = tf.constant(5.)

with tf.GradientTape(persistent = True) as tape:
    tape.watch([w1, b1, w2, b2])
    y1 = w1*x + b1
    y2 = w2*y1 + b2
dy2_dy1 = tape.gradient(y2, [y1])[0]
print(type(dy2_dy1))
dy2_dw1 = tape.gradient(y2, [w1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]

print(dy2_dw1, dy2_dy1*dy1_dw1)