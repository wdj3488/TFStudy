import tensorflow as tf
from tensorflow.keras import layers, optimizers

x = tf.random.normal([200, 4, 4, 3], mean=1., stddev=0.5)
net = layers.BatchNormalization(axis=-1, center=True, scale=True, trainable=True)
out = net(x)
print('forward in test mode', net.variables)
out = net(x, training=True)
print('forward in train mode(1 step)', net.variables)

for i in range(100):
    out = net(x, training=True)
print('forward in train mode(100 step)', net.variables)

for i in range(1000):
    out = net(x, training=True)
print('forward in train mode(1000 step)', net.variables)

optimizer = optimizers.SGD(learning_rate=1e-2)
for i in range(20):
    with tf.GradientTape() as tape:
        out = net(x, training=True)
        loss = tf.reduce_mean(tf.pow(out, 2)) - 1
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
print('backward 10 steps:', net.variables)
