import numpy as np


def compute_error_for_line_given_points(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error/len(points)


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = len(points)
    for i in range(0, N):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += 2.0/N * ((w_current*x + b_current) - y)
        w_gradient += 2.0/N * ((w_current*x + b_current) - y) * x
    new_b = b_current - learning_rate*b_gradient
    new_w = w_current - learning_rate*w_gradient
    return [new_b, new_w]


def gradient_descent_runner(points, start_b, start_w, learning_rate, num_iteartions):
    b = start_b
    w = start_w
    for i in range(0, num_iteartions):
        b, w = step_gradient(b, w, points, learning_rate)
    return [b, w]


def run():
    points = np.loadtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print('Bef {0} iteartions b = {1}, w = {2} ,error = {3}'
          .format(num_iterations, initial_b, initial_w, compute_error_for_line_given_points(initial_b, initial_w,points)))
    print("Runing ...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print('After {0} iteartions b = {1}, w = {2} ,error = {3}'
          .format(num_iterations, b, w, compute_error_for_line_given_points(b, w, points)))


if __name__ == '__main__':
    run()
