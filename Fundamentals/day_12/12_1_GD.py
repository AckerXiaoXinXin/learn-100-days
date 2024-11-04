import numpy as np


def f(x):
    return x**2


def df(x):
    return 2*x


def gradient_descent(start_p, iter, lr):
    x0 = start_p
    for _ in range(iter):
        dx = df(x0)
        x0 = x0 - lr * dx

    return x0

start_p = 10
learning_rate = 0.01
iters = 10000

opt_x = gradient_descent(start_p, iters, learning_rate)

print(opt_x)



