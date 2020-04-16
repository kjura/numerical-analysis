# Task: Implement Euler's method for ODE, solve and plot equation -> x' = -x , x(0) = 1
# For h = 0.1 , h = 1.0 i h = 2.2
# Then do the same for classical harmonic oscillator x'' + x = 0, x(0) = 1, x'(0)=0

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


class PapaEuler:

    def sol_euler(self, y_i, h, ti, tf, fun):
        interval_start = ti
        iteration = 0
        self.y_sol = []
        self.t_span = []


        table_of_values = {"Step": [], "t_i": [], "y": [], "y_true": [], "e_i": []}
        df = pd.DataFrame(data=table_of_values)

        while ti <= tf:
            y = y_i + h * fun(ti, y_i)

            sumarised_data = pd.DataFrame({"Step": [iteration], "t_i": [ti], "y": [y],
                                           "y_true": [np.exp(-ti)], "e_i": [math.fabs(np.exp(ti) - y)]})

            df = df.append(sumarised_data)
            y_i = y
            ti += h
            iteration += 1
            self.y_sol.append(y)



        t = np.linspace(interval_start, tf, len(self.y_sol))
        plt.plot(t, self.y_sol)
        plt.show()
        print(df)



