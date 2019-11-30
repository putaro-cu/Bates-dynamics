# 4次Runge-Kutta法による連立微分方程式の数値解シミュレーション

import numpy as np
import matplotlib.pyplot as plt
import math

h = 0.01  # 刻み幅
t = 0.0
t_max = 60.0

N0 = 90
M0 = 1
u0 = 0.05
X = [[N0], [M0], [u0], [0], [M0 / (N0 + M0)]]  # 初期値[N,M,u,t,M/(N+M)]を表す。
a = 0.4
b = 0.5
c = 0.5
r0 = 0.5
g0 = 0.8
P = 1
k = 60
G = 0.1
K1 = 100
K2 = 100


def Batesian(N, M, u, a, b, c, r0, g0, P, k, G, K1, K2):  # 微分方程式の定義
    N_dot = N * (r0 * (1 - N / K1 - c * u) - a * P * ((M / (N + M)) ** (k * u)))  # モデル
    M_dot = M * (g0 * (1 - M / K2) - b * P * ((M / (N + M)) ** (k * u)))  # 擬態種
    u_dot = (-r0 * c - a * k * math.log(M / (N + M)) * ((M / (N + M)) ** (k * u))) * G  # 毒性の進化
    if u <= 0:  # u>0を考慮
        u_dot = 0
    return np.array([N_dot, M_dot, u_dot])


K = np.zeros([3, 5])  # [k1,k2,k3,k4,k] #数字の文字列になっている

while t < t_max:  # 時間がt_maxになるまで計算

    # Batesian関数のx,y,zを指定。[-1]はリストの最後の要素を表す。
    N, M, u = X[0][-1], X[1][-1], X[2][-1]

    K[:, 0] = h * Batesian(N, M, u, a, b, c, r0, g0, P, k, G, K1, K2)
    K[:, 1] = h * Batesian(N + K[0, 0] / 2.0, M + K[1, 0] / 2.0, u + K[2, 0] / 2.0, a, b, c, r0, g0, P, k, G, K1, K2)
    K[:, 2] = h * Batesian(N + K[0, 1] / 2.0, M + K[1, 1] / 2.0, u + K[2, 1] / 2.0, a, b, c, r0, g0, P, k, G, K1, K2)
    K[:, 3] = h * Batesian(N + K[0, 2], M + K[1, 2], u + K[2, 2], a, b, c, r0, g0, P, k, G, K1, K2)
    K[:, 4] = (K[:, 0] + 2.0 * K[:, 1] + 2.0 * K[:, 2] + K[:, 3]) / 6.0

    for j in range(3):
        X[j].append(X[j][-1] + K[j, 4])

    X[3].append(t)
    X[4].append(X[1][-1] / (X[0][-1] + X[1][-1]))
    t += h

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0, 0].plot(X[3], X[0])
axes[0, 0].set_title("model", size=14)
axes[0, 0].set_xlabel("time", size=14)
axes[0, 0].set_ylabel("N", size=14)

axes[0, 1].plot(X[3], X[1])
axes[0, 1].set_title("mimic", size=14)
axes[0, 1].set_xlabel("time", size=14)
axes[0, 1].set_ylabel("M", size=14)

axes[1, 0].plot(X[3], X[2])
u_lim = math.log(-c * r0 / (a * k * P * math.log(X[1][-1] / (X[1][-1] + X[0][-1])))) / (
            k * math.log(X[1][-1] / (X[1][-1] + X[0][-1])))
axes[1, 0].axhline(u_lim, ls="-.", color="magenta")
axes[1, 0].set_title("poison", size=14)
axes[1, 0].set_xlabel("time", size=14)
axes[1, 0].set_ylabel("level", size=14)

axes[1, 1].plot(X[0], X[1])
axes[1, 1].set_title("", size=14)
axes[1, 1].set_xlabel("model", size=14)
axes[1, 1].set_ylabel("mimic", size=14)

fig.show()
