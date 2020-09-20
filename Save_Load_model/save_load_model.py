import tensorflow as tf
from tensorflow.keras import layers, datasets, Sequential, optimizers, metrics
import os

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess)
db = db.shuffle(60000).batch(batchsz)
db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess).batch(batchsz)

# network = Sequential([layers.Dense(256, activation='relu'),
#                       layers.Dense(128, activation='relu'),
#                       layers.Dense(64, activation='relu'),
#                       layers.Dense(32, activation='relu'),
#                       layers.Dense(10)])
# network.build(input_shape=[None, 28*28])
# network.summary()
#
#
# network.compile(optimizer=optimizers.Adam(learning_rate=1e-2),
#                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
# network.fit(db, epochs=3, validation_data=db_val, validation_freq=2)
network = tf.keras.models.load_model('model.pb')
network.evaluate(db_val)
network.save('model.pb')
print('saved total model')
del network
