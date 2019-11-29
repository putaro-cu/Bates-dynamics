# Adams-Bashforth-Moulton法による数値解導出(Scipyのodeintを使用)
# うまく収束しない

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


def func(X, a=0.4, b=0.5, c=1, r0=0.5, g0=0.8, P=1, k=50, G=0.005, K1=100, K2=100):
    return [X[0] * (r0 * (1 - X[0] / K1 - c * X[2]) - a * P * ((X[1] / (X[0] + X[1])) ** (k * X[2])))
        , X[1] * (g0 * (1 - X[1] / K2) - b * P * ((X[1] / (X[0] + X[1])) ** (k * X[2])))
        , (-r0 * c - a * k * math.log(X[1] / (X[0] + X[1])) * ((X[1] / (X[0] + X[1])) ** (k * X[2]))) * G
            ]


N0 = 90
M0 = 1
u0 = 0.05
X0 = [N0, M0, u0]
t = np.arange(0, 1000, 0.001)

X = odeint(func, X0, t, args=())

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0, 0].plot(X[:,0])
axes[0, 0].set_title("model", size=14)
axes[0, 0].set_xlabel("time", size=14)
axes[0, 0].set_ylabel("N", size=14)

axes[0, 1].plot(X[:,1])
axes[0, 1].set_title("mimic", size=14)
axes[0, 1].set_xlabel("time", size=14)
axes[0, 1].set_ylabel("M", size=14)

axes[1, 0].plot(X[:,2])
axes[1, 0].set_title("poison", size=14)
axes[1, 0].set_xlabel("time", size=14)
axes[1, 0].set_ylabel("level", size=14)

axes[1, 1].plot(X[:,0], X[:,1])
axes[1, 1].set_title("", size=14)
axes[1, 1].set_xlabel("model", size=14)
axes[1, 1].set_ylabel("mimic", size=14)

fig.show()
