# 7章 ロジスティクスネットワークの最適設計を行う10本ノック（物流ネットワーク全体の最適化）

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import numpy as np
from itertools import product
from pulp import LpVariable, lpSum, value  # 最適化モデルの作成を行う
from ortoolpy import model_min, addvars, addvals  # 目的関数を生成
import networkx as nx


# ノック６１：輸送最適化問題を解いてみよう
df_tc = pd.read_csv('trans_cost.csv', index_col="工場")  # それぞれのリンクでのコスト
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 初期設定 #
np.random.seed(1)
nw = len(df_tc.index)  # 倉庫の個数，3
nf = len(df_tc.columns)  # 工場の個数，4
pr = list(product(range(nw), range(nf)))  # 直積を計算，(W1, F1), (W1, F2), ...，実際には(0, 0), (0, 1), ...

# 数理モデル作成 #
m1 = model_min()  # 最小化を行うモデルを定義
v1 = {(i, j): LpVariable('v%d_%d' % (i, j), lowBound=0) for i, j in pr}  # それぞれのリンクの輸送量，v0_0, v0_1, ...，下限 = 0
m1 += lpSum(df_tc.iloc[i][j] * v1[i, j] for i, j in pr)  # 目的関数をm1に定義
# 制約条件を定義
for i in range(nw):
    m1 += lpSum(v1[i, j] for j in range(nf)) <= df_supply.iloc[0][i]  # 倉庫の供給する部品量（合計を取る）が限界を超えないように
for j in range(nf):
    m1 += lpSum(v1[i, j] for i in range(nw)) >= df_demand.iloc[0][j]  # 工場の製造する製品（合計を取る）が需要を満たすように，求める部品の量以上になるように
m1.solve()  # 変数v1が最適化される

# 総輸送コスト計算 #
df_tr_sol = df_tc.copy()
total_cost = 0
for k, x in v1.items():  # (i, j)とv0_0を返す．k：座標（倉庫，工場），x：v0_0, ...（輸送量）
    i, j = k[0], k[1]
    df_tr_sol.iloc[i][j] = value(x)  # 実際の値を入れていく
    total_cost += df_tc.iloc[i][j] * value(x)

print(df_tr_sol)
print("総輸送コスト：" + str(total_cost))  # 最適な輸送コストが求められる．1296万円（さっきは1433万円）


# ノック６２：最適輸送ルートをネットワークで確認しよう
df_tr = df_tr_sol.copy()  # それぞれのリンクでの輸送量
df_pos = pd.read_csv('trans_route_pos.csv')

G = nx.Graph()
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])  # 頂点作成

num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i == j):
            G.add_edge(df_pos.columns[i], df_pos.columns[j])  # 辺の追加
            if num_pre < len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns) and(df_pos.columns[j] in df_tr.index):  # iが工場で，jが倉庫
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]] * size
                elif (df_pos.columns[j] in df_tr.columns) and(df_pos.columns[i] in df_tr.index):  # jが工場で，iが倉庫
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]] * size
                edge_weights.append(weight)

pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0], df_pos[node][1])

nx.draw(G, pos, with_labels=True, font_size=16, node_size=1000, node_color='k', font_color='w', width=edge_weights)

plt.show()


# ノック６３：最適輸送ルートが制約条件内に収まっているかどうかを確認しよう
# 制約条件計算関数

# 需要側（工場）の制約条件




# ノック６４：生産計画


# ノック６５：


# ノック６６：


# ノック６７：


# ノック６８：


# ノック６９：


# ノック７０：
