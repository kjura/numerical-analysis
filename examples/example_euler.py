# Task: Implement Euler's method for ODE, solve and plot equation -> x' = -x , x(0) = 1
# For h = 0.1 , h = 1.0 i h = 2.2
# Then do the same for classical harmonic oscillator x'' + x = 0, x(0) = 1, x'(0)=0


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from methods.main_papi import PapaEuler

# Solve for x' = -x , x(0) = 1

def fun(t, y):
    dydt = t * y
    return dydt

def model(t_i, t_f, y_i, h, fun):
    interval_start = t_i
    test_list = []

    while t_i <= t_f:
        y = y_i + h * fun(t_i, y_i)
        y_i = y
        t_i += h
        test_list.append(y)

    test_plot_args = np.linspace(interval_start, t_f, len(test_list))

    model_result = (test_plot_args, test_list)

    return model_result

test = PapaEuler()

def md(t, y):
    dydt = -y
    return dydt

t = np.linspace(0, 5, 50)

plt.plot(t, np.exp(-t))
test.sol_euler(1, 0.1, 0, 5, md)