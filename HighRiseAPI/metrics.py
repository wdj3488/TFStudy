import tensorflow as tf
from tensorflow.keras import optimizers, datasets, layers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x,y), (x_test, y_test) = datasets.mnist.load_data()

batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(batchsz)

network = Sequential([layers.Dense(256, activation=tf.nn.relu),
                      layers.Dense(128, activation=tf.nn.relu),
                      layers.Dense(64, activation=tf.nn.relu),
                      layers.Dense(32, activation=tf.nn.relu),
                      layers.Dense(10)])
network.build(input_shape=[None, 28*28])
network.summary()

optimizer = optimizers.Adam(learning_rate=1e-2)
acc_metrics = metrics.Accuracy()
loss_metrics = metrics.Mean()

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, [-1, 28*28])
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)
        pred = tf.nn.softmax(out)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, pred))
        loss_metrics.update_state(loss)
    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    if step % 100 == 0:
        print(step, "loss:", loss)  # 第100步的结果
        # 计算100步的平均值？
        print(step, "loss:", loss_metrics.result())
        loss_metrics.reset_states()
    if step % 300 == 0:
        total, total_correct = 0, 0
        acc_metrics.reset_states()
        for step, (x, y) in enumerate(db_test):
            x = tf.reshape(x, [-1, 28*28])
            out = network(x)
            pred = tf.nn.softmax(out)
            pred = tf.argmax(pred, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            result = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            total_correct += tf.reduce_sum(result)
            total += result.shape[0]
            # acc_metrics.reset_states()
            acc_metrics.update_state(y, pred)  # 第0步和第300步准确度的平均值
        print(step, "Evaluate Acc:", total_correct/total, acc_metrics.result().numpy())

