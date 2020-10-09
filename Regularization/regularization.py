import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential, metrics, datasets


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()
db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)
db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_val = db_val.map(preprocess).batch(batchsz)

network = Sequential([layers.Dense(256, activation=tf.nn.relu),
                      layers.Dense(128, activation=tf.nn.relu),
                      layers.Dense(64, activation=tf.nn.relu),
                      layers.Dense(32, activation=tf.nn.relu),
                      layers.Dense(10)])
network.build(input_shape=[None, 28*28])
network.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)

for step, (x, y) in enumerate(db_train):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, [-1, 28*28])
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.reduce_sum(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
        loss_regularization = []
        for p in network.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(loss_regularization)
        loss = loss + 0.0001 * loss_regularization
    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 100 == 0:
        print(step, 'loss:', float(loss), 'loss_regularization:', float(loss_regularization))

    if step % 500 == 0:
        total, total_correct = 0, 0
        for step, (x, y) in enumerate(db_val):
            x = tf.reshape(x, [-1, 28*28])
            out = network(x)
            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            total_correct += tf.reduce_sum(correct)
            total += correct.shape[0]
        print(step, 'Acc:', total_correct/total)

