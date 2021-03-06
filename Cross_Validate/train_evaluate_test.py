import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, metrics, optimizers


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()

idx = tf.range(60000)
idx = tf.random.shuffle(idx)
x_train, y_train = tf.gather(x, idx[:50000]), tf.gather(y, idx[:50000])
x_val, y_val = tf.gather(x, idx[-10000:]), tf.gather(y, idx[-10000:])
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(50000).batch(batchsz)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=[None, 28*28])
network.summary()

network.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )
network.fit(db_train, epochs=15, validation_data=db_val, validation_freq=2)
print('Test performance:')
network.evaluate(db_test)

sample = next(iter(db_test))
x = sample[0]
y = sample[1]
pred = network.predict(x)
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)
print(y)
print(pred)