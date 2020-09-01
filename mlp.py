import tensorflow as tf
from tensorflow import keras


x = tf.random.normal([2, 3])
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2)]
)
model.build(input_shape=[None, 3])
model.summary()

for p in model.trainable_variables:
    print((p.name, p.shape))

