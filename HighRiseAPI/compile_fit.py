import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    x = tf.reshape(x, [28*28])
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess).shuffle(60000).batch(batchsz)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(batchsz)
sample = next(iter(db))
print(sample[0].shape, sample[1].shape)


network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])

network.build(input_shape=[None, 28*28])
network.summary()

network.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # 注意这里的CategoricalCrossentropy是大写
                metrics=['accuracy'])
network.fit(db, epochs=5, validation_data=db_test, validation_freq=2)  # 训练两遍后验证一遍
network.evaluate(db_test)

sample = next(iter(db))
x = sample[0]
y = sample[1]
pred = network(x)
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)
print(pred)
print(y)

