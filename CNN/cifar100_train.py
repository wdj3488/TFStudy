import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, optimizers, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVET'] = '2'

conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
]


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
print(x.shape, y.shape)
y = tf.squeeze(y)
y_test = tf.squeeze(y_test)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess)

def main():
    conv_net = Sequential(conv_layers)
    fc_net = Sequential([layers.Dense(256, activation='relu'),
                         layers.Dense(128, activation='relu'),
                         layers.Dense(100)])
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    optimizer = optimizers.Adam(learning_rate=1e-4)
    variable = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(15):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)
                out = tf.reshape(out, [-1, 512])
                logits = fc_net(out)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variable)
            optimizer.apply_gradients(zip(grads, variable))
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss)

        total_num, total_correct = 0, 0
        for (x, y) in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            total_num += correct.shape[0]
            total_correct += tf.reduce_sum(correct)
        acc = total_correct/total_num
        print(epoch, 'Acc:', acc)


if __name__ == '__main__':
    main()
