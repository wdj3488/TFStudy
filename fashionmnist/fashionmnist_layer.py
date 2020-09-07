import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers, metrics

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape)
batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batchsz)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsz)
model = Sequential([layers.Dense(256, activation='relu'),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(10)])
model.build(input_shape=[None, 28*28])
model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)

def main():
    for epoch in range(10):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28*28])
            x = tf.cast(x, dtype=tf.float32)/255.
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            if step % 100 == 0:
                print("epoch:", epoch, "step:", step, "loss:", loss)
        total_correct = 0
        total_num = 0
        for (x, y) in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            x = tf.cast(x, dtype=tf.float32) / 255.
            logits = model(x)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.uint8)
            #print(pred)
            #print(y)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            total_correct += correct
            total_num += pred.shape[0]
        acc = total_correct/total_num
        print("epoch:", epoch, "test acc:", acc)

if __name__ == '__main__':
    main()


