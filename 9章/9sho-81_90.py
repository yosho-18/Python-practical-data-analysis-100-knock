# 9章 潜在顧客を把握するための画像認識１０本（ある程度の画像認識技術．実務重視，画像認識ライブラリOpenCV）

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def func(state, t):
    dydt = np.zeros_like(state)
    dydt[0] = state[1]
    dydt[1] = -(G / L) * sin(state[0])
    return dydt


G = 9.8  # 重力加速度
L = 1  # 振り子の長さ

th1 = 30.0  # 角度の初期値[deg]
w1 = 0.0  # 角速度の初期値[deg]

# 初期状態
state = np.radians([th1, w1])
dt = 0.05
t = np.arange(0.0, 20, dt)

sol = odeint(func, state, t)

theta = sol[:, 0]
x = L * sin(theta)
y = - L * cos(theta)

print([1, ] + ["w", "as"] + [2, ])
print(np.random.randint(2 ** 30))
print(np.arange(30) + 1)
# ノック８１：画像データを読み込んでみよう


# ノック８２：


# ノック８３：


# ノック８４：


# ノック８５：


# ノック８６：


# ノック８７：


# ノック８８：


# ノック８９：


# ノック９０：