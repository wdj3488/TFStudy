import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

[x, y], [x_val, y_val] = datasets.mnist.load_data()
print(type(x), type(y))
x = tf.convert_to_tensor(x, tf.float32)/255.0
y = tf.convert_to_tensor(y, tf.int32)
y = tf.one_hot(y, 10)
print(type(x), type(y))
print(x.shape, y.shape)
train_datasets = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_datasets.batch(200)

model = keras.Sequential([layers.Dense(512, activation='relu'),
                          layers.Dense(256, activation='relu'),
                          layers.Dense(10)])
optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, [-1, 28*28])
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y))/y.shape[0]
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(0, 30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
