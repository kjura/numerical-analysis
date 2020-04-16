# Task: Implement Euler's method for ODE, solve and plot equation -> x' = -x , x(0) = 1
# For h = 0.1 , h = 1.0 i h = 2.2
# Then do the same for classical harmonic oscillator x'' + x = 0, x(0) = 1, x'(0)=0

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import pandas as pd




#def ode_eq(t, y):
    #return -y

#def np_method():
    #t_exact = np.linspace(0, 10)
    #sol = solve_ivp(ode_eq, [0, 10], [1], t_eval=np.linspace(0, 10), method="Radau")

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle("Comparison of exact and numerical solution")
    #ax1.plot(sol.t, sol.y[0])
    #ax2.plot(t_exact, np.exp(-t_exact))
    #plt.show()



#   initial condition
#w_i = 1.0

# step size
#h = 0.1
#t_i = 0
# papa euler's in action

#tablica = []




#table_of_values = {"Step": [], "t_i": [], "w_i": [], "y_i": [], "e_i":[]}
#df = pd.DataFrame(data=table_of_values)

#step = 0
#for end in range(0, 10):

    #w_iplus_one = w_i + h * (-w_i)
    #tablica.append(w_iplus_one)

    #sumarised_data = pd.DataFrame({"Step": [step], "t_i": [t_i], "w_i": [w_iplus_one],
                                   #"y_i": [w_i], "e_i":[math.fabs(w_i - w_iplus_one)]}

    #df = df.append(sumarised_data)


    #w_i = w_iplus_one
    #t_i += h
    #step += 1

class PapaEuler:

    def sol_euler(self, y0, h, ti, tf):
        interval_start = ti
        iteration = 0
        self.y_sol = []
        self.t_span = []

        table_of_values = {"Step": [], "t_i": [], "y": [], "y_true": [], "e_i": []}
        df = pd.DataFrame(data=table_of_values)

        while ti <= tf:
            y = y0 + h * (-y0)

            sumarised_data = pd.DataFrame({"Step": [iteration], "t_i": [ti], "y": [y],
                                           "y_true": [np.exp(-ti)], "e_i": [math.fabs(np.exp(ti) - y)]})

            df = df.append(sumarised_data)
            y0 = y
            ti += h
            iteration += 1
            self.y_sol.append(y)



        t = np.linspace(interval_start, tf, len(self.y_sol))
        plt.plot(t, self.y_sol)
        plt.show()
        print(df)

test_instance = PapaEuler()
test_instance.sol_euler(1.0, 0.1, 0, 5)
#t_exact = np.linspace(0, 5, 51)
#plt.plot(t_exact, np.exp(-t_exact))
#plt.show()

