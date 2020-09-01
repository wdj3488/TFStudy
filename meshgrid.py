import tensorflow as tf
import matplotlib.pyplot as plt
import math


def func(x):
    z = tf.math.sin(x[..., 0]) + tf.math.cos(x[..., 1])
    return z


x = tf.linspace(0., 2*math.pi, 500)
y = tf.linspace(0., 2*3.14, 500)
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y], axis=2)
print('points:', points.shape)
z = func(points)
print('z:', z.shape)
# plt.imshow(z, origin='lower', interpolation='none')
plt.contour(point_x, point_y, z)

plt.show()
