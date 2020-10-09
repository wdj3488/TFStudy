import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, optimizers, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batchsz)

optimizer = optimizers.Adam(learning_rate=1e-3)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dropout(0.5),
                      layers.Dense(128, activation='relu'),
                      layers.Dropout(0.5),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)
                      ])
network.build(input_shape=[None, 28*28])
network.summary()

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, [-1, 28*28])
        out = network(x, training=True)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.reduce_sum(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
        loss_regularizaton = []
        for p in network.trainable_variables:
            loss_regularizaton.append(tf.nn.l2_loss(p))
        loss_regularizaton = tf.reduce_sum(loss_regularizaton)
        loss += 0.0001*loss_regularizaton
    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    if step % 100 == 0:
        print(step, 'loss:', loss, 'loss_regularization:', loss_regularizaton)
    if step % 500 == 0:
        total, total_correct = 0, 0
        for index, (x, y) in enumerate(db_val):
            x = tf.reshape(x, [-1, 28*28])
            out = network(x, training=False)
            out = tf.argmax(out, axis=1)
            out = tf.cast(out, dtype=tf.int32)
            pred = tf.cast(tf.equal(out, y), dtype=tf.int32)
            total_correct += tf.reduce_sum(pred)
            total += pred.shape[0]
        print(step, 'Acc with drop:', total_correct/total)

        total, total_correct = 0, 0
        for index, (x, y) in enumerate(db_val):
            x = tf.reshape(x, [-1, 28 * 28])
            out = network(x, training=True)
            out = tf.argmax(out, axis=1)
            out = tf.cast(out, dtype=tf.int32)
            pred = tf.cast(tf.equal(out, y), dtype=tf.int32)
            total_correct += tf.reduce_sum(pred)
            total += pred.shape[0]
        print(step, 'Acc without drop:', total_correct / total)