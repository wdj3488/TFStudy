import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
from CNN.ResNet import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0 - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y)
y_test = tf.squeeze(y_test)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(learning_rate=1e-3)
    for epoch in range(500):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                y = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y, out, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 5 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', loss)
        total, total_correct = 0, 0
        for index, (x, y) in enumerate(test_db):
            pred = model(x)
            pred = tf.argmax(pred, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)

            total += correct.shape[0]
            total_correct += tf.reduce_sum(correct)
        acc = total_correct/total
        print('Acc:', acc)


if __name__ == '__main__':
    main()
