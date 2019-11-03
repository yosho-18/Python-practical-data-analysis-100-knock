# 7章 ロジスティクスネットワークの最適設計を行う10本ノック（物流ネットワーク全体の最適化）

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import numpy as np
from itertools import product
from pulp import LpVariable, lpSum, value  # 最適化モデルの作成を行う
from ortoolpy import model_min, addvars, addvals  # 目的関数を生成


# ノック６１：輸送最適化問題を解いてみよう
df_tc = pd.read_csv('trans_cost.csv', index_col="工場")
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 初期設定 #
np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)
pr = list(product(range(nw), range(nf)))  # 直積を計算，(W1, F1), (W1, F2), ...

# 数理モデル作成 #
m1 = model_min()  # 最小化を行うモデルを定義


# ノック６２：


# ノック６３：


# ノック６４：


# ノック６５：


# ノック６６：


# ノック６７：


# ノック６８：


# ノック６９：


# ノック７０：