# Runge-Kutta法による連立微分方程式の数値解シミュレーション
# 擬態種の個体群サイズと毒性の関係

import numpy as np
import matplotlib.pyplot as plt
import math

h = 0.01  # 刻み幅
t = 0.0
t_max = 100.0
Res = [[], []]  # 結果[limit_mimic,u]を表す。


def Batesian(N, M, u, K2, a=0.2, b=0.5, r0=0.2, g0=0.15, P=1, k=50, G=0.01, K1=100):  # 微分方程式の定義
    N_dot = N * (r0 * (1 - N / K1 - u) - a * P * ((M / (N + M)) ** (k * u)))
    M_dot = M * (g0 * (1 - M / K2) - b * P * ((M / (N + M)) ** (k * u)))
    u_dot = (-r0 - a * k * math.log(M / (N + M)) * ((M / (N + M)) ** (k * u))) * G
    return np.array([N_dot, M_dot, u_dot])


for x in range(1, 100, 2):
    X = [[1], [1], [0.09], [0]]  # 初期値[N,M,u,t]を表す。
    k = np.zeros([3, 5])  # [k1,k2,k3,k4,k] #数字の文字列になっている

    while t < t_max:  # 時間がt_maxになるまで計算

        # Batesian関数のx,y,zを指定。[-1]はリストの最後の要素を表す。
        N, M, u = X[0][-1], X[1][-1], X[2][-1]

        k[:, 0] = h * Batesian(N, M, u, x)
        k[:, 1] = h * Batesian(N + k[0, 0] / 2.0, M + k[1, 0] / 2.0, u + k[2, 0] / 2.0, x)
        k[:, 2] = h * Batesian(N + k[0, 1] / 2.0, M + k[1, 1] / 2.0, u + k[2, 1] / 2.0, x)
        k[:, 3] = h * Batesian(N + k[0, 2], M + k[1, 2], u + k[2, 2], x)
        k[:, 4] = (k[:, 0] + 2.0 * k[:, 1] + 2.0 * k[:, 2] + k[:, 3]) / 6.0

        for j in range(3):
            X[j].append(X[j][-1] + k[j, 4])

        X[3].append(t)
        t += h

    t = 0
    Res[0].append(X[1][-1])
    Res[1].append(X[2][-1])

plt.scatter(Res[0], Res[1])
plt.xlabel("population size of mimic")
plt.ylabel("poison level")
plt.title("population size of mimic & poison")
plt.show()
