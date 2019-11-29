# odeサンプル

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt


def logistic(t, y, r):
    return r * y * (1.0 - y)

r = .01
t0 = 0
y0 = 1e-5
t1 = 5000.0

backend = 'dopri5'
# backend = 'dop853'
solver = ode(logistic).set_integrator(backend)

sol = []
def solout(t, y):
    sol.append([t, *y])
solver.set_solout(solout)
solver.set_initial_value(y0, t0).set_f_params(r)
solver.integrate(t1)

sol = np.array(sol)

plt.plot(sol[:, 0], sol[:, 1], 'b.-')
plt.show()
