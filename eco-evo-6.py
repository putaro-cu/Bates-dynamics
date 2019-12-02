# 4次Runge-Kutta法による連立微分方程式の数値解シミュレーション
# α,βと共存の関係

import numpy as np
import matplotlib.pyplot as plt
import math


def Batesian(N, M, u, a, b, c, r0, g0, P, k, G, K1, K2):  # 微分方程式の定義
    N_dot = N * (r0 * (1 - N / K1 - c * u) - a * P * ((M / (N + M)) ** (k * u)))  # モデル
    M_dot = M * (g0 * (1 - M / K2) - b * P * ((M / (N + M)) ** (k * u)))  # 擬態種
    u_dot = (-r0 * c - a * k * math.log(M / (N + M)) * ((M / (N + M)) ** (k * u))) * G  # 毒性の進化
    if u <= 0:  # u>0を考慮
        u_dot = 0
    return np.array([N_dot, M_dot, u_dot])


def Solver(N0, M0, u0, a, b, c, r0, g0, P, k, G, K1, K2):
    K = np.zeros([3, 5])  # [k1,k2,k3,k4,k]
    X = [[N0], [M0], [u0]]  # 初期値[N,M,u,t]を表す。
    h = 0.01  # 刻み幅
    t = 0.0
    t_max = 60.0

    while t < t_max:  # 時間がt_maxになるまで計算

        # Batesian関数のN, M, uを指定。[-1]はリストの最後の要素を表す。
        N, M, u = X[0][-1], X[1][-1], X[2][-1]

        K[:, 0] = h * Batesian(N, M, u, a, b, c, r0, g0, P, k, G, K1, K2)
        K[:, 1] = h * Batesian(N + K[0, 0] / 2.0, M + K[1, 0] / 2.0, u + K[2, 0] / 2.0, a, b, c, r0, g0, P, k, G, K1,
                               K2)
        K[:, 2] = h * Batesian(N + K[0, 1] / 2.0, M + K[1, 1] / 2.0, u + K[2, 1] / 2.0, a, b, c, r0, g0, P, k, G, K1,
                               K2)
        K[:, 3] = h * Batesian(N + K[0, 2], M + K[1, 2], u + K[2, 2], a, b, c, r0, g0, P, k, G, K1, K2)
        K[:, 4] = (K[:, 0] + 2.0 * K[:, 1] + 2.0 * K[:, 2] + K[:, 3]) / 6.0

        for j in range(3):
            X[j].append(X[j][-1] + K[j, 4])

        t += h
    return [X[0][-1], X[1][-1], X[2][-1]]


N0 = 0.5
M0 = 0.5
u0 = 0.05
a = 0.3
b = 0.5
c = 0.5
# r0 = 0.5
# g0 = 0.8
P = 1
k = 50
G = 0.1
K1 = 1
K2 = 1

x = np.linspace(0, 3, 50, endpoint=False)
y = np.linspace(0, 3, 50, endpoint=False)
X, Y = np.meshgrid(x, y)

Z = [[0 for i in range(len(x))] for j in range(len(y))]
for i, r0 in enumerate(x):
    for j, g0 in enumerate(y):
        lim = Solver(N0, M0, u0, a, b, c, r0, g0, P, k, G, K1, K2)
        if lim[0] <= 0.001:
            if lim[1] <= 0.001:  # 両方絶滅
                Z[j][i] = 0
            else:
                Z[j][i] = 1  # モデル絶滅
        else:
            if lim[1] <= 0.001:  # ミミック絶滅
                Z[j][i] = 2
            else:
                Z[j][i] = 3  # 共存

plt.pcolormesh(X, Y, Z, cmap="binary", vmin=0, vmax=3)
plt.colorbar()  # カラーバーの表示
plt.xlabel('r0')
plt.ylabel('g0')
plt.title("coexist or extinct")
plt.show()
