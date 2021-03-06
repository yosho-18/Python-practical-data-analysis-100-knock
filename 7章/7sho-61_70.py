# 7章 ロジスティクスネットワークの最適設計を行う10本ノック（物流ネットワーク全体の最適化）

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from pulp import LpVariable, lpSum, value  # 最適化モデルの作成を行う
from ortoolpy import model_min, model_max, addvars, addvals, logistics_network  # 目的関数を生成
import networkx as nx

"""最適化問題の詳細についてはAppendix③参照"""


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
def condition_demand(df_tr, df_demand):
    flag = np.zeros(len(df_demand.columns))  # F1, F2, F3, F4
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])  # F1, F2, ...（縦（倉庫W毎）にsumを取る）
        if (temp_sum >= df_demand.iloc[0][i]):
            flag[i] = 1
    return flag

# 供給側（倉庫）
def condition_supply(df_tr, df_supply):
    flag = np.zeros(len(df_supply.columns))  # W1, W2, W3
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])  # W1, W2, ...（横（工場F毎）にsumを取る），行指定のため.locを使う
        if temp_sum <= df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("需要条件計算結果：" + str(condition_demand(df_tr_sol, df_demand)))  # [1, 1, 1, 1]ならOK
print("供給条件計算結果：" + str(condition_supply(df_tr_sol, df_supply)))  # [1, 1, 1]ならOK

"""線形最適化の定式化できるものは比較的短時間に解を求めることができる"""



# ノック６４：生産計画に関するデータを読み込んでみよう
df_material = pd.read_csv('product_plan_material.csv', index_col="製品")
print(df_material)
df_profit = pd.read_csv('product_plan_profit.csv', index_col="製品")
print(df_profit)
df_stock = pd.read_csv('product_plan_stock.csv', index_col="項目")
print(df_stock)
df_plan = pd.read_csv('product_plan.csv', index_col="製品")
print(df_plan)


# ノック６５：利益を計算する関数を作ってみよう
def product_plan(df_profit, df_plan):
    profit = 0
    for i in range(len(df_profit.index)):  # 製品１，製品２，i = 0, 1，利益
        for j in range(len(df_plan.columns)):  # 生産量，j = 0
            profit += df_profit.iloc[i][j] * df_plan.iloc[i][j]  # 利益 * 生産量
    return profit

print("総利益：" + str(product_plan(df_profit, df_plan)))  # 80万円


# ノック６６：生産最適化問題を解いてみよう
df = df_material.copy()
inv = df_stock

# ノック６１と同じ流れ
m = model_max()  # 最大化
v1 = {(i): LpVariable('v%d'%(i), lowBound=0) for i in range(len(df_profit))}  # v1 = 製品１，製品２の個数v0, v1
m += lpSum(df_profit.iloc[i] * v1[i] for i in range(len(df_profit)))  # 目的関数，利益 * 生産量
for i in range(len(df_material.columns)):  # 原料１，原料２，原料３
    m += lpSum(df_material.iloc[j, i] * v1[j] for j in range(len(df_profit))) \
         <= df_stock.iloc[:, i]  # 制約条件，原料に対して製品が必要な原料を足していく，そのsumが在庫以下であることを条件にする
m.solve()

df_plan_sol = df_plan.copy()
for k, x in v1.items():  # k = 0, 1，x = v0, v1
    df_plan_sol.iloc[k] = value(x)  # v0, v1に関しての新たな生産計画を作る
print(df_plan_sol)
print("純利益：" + str(value(m.objective)))  # value(m.objective)で目的関数の値を出力，95万


# ノック６７：最適生産計画が制約条件内に収まっているかどうかを確認しよう
"""最適化計算を行った結果を「あの手この手で」理解する"""
def condition_stock(df_plan, df_material, df_stock):
    flag = np.zeros(len(df_material.columns))
    for i in range(len(df_material.columns)):  # 原料１，原料２，原料３
        temp_sum = 0
        for j in range(len(df_material.index)):  # 製品１，製品２
            temp_sum = temp_sum + df_material.iloc[j][i] \
                       * float(df_plan.iloc[j])  # 製品jがdf_plan.iloc[j]個でそれに応じた１個当たりの原材料数をかける
            if temp_sum <= float(df_stock.iloc[0][i]):  # 使われている原材料が，在庫を越えていないかチェック
                flag[i] = 1
            print(df_material.columns[i] + " 使用量：" + str(temp_sum) + "，在庫：" + str(float(df_stock.iloc[0][i])))
    return flag
print("制約条件計算結果：" + str(condition_stock(df_plan_sol, df_material, df_stock)))  # 原材料２，原材料３は全て使っている


# ノック６８：ロジスティクスネットワーク問題を解いてみよう
"""輸送コストと製造コストの和を最小化する，制約条件：各商店での販売数が需要数を上回ること"""
製品 = list('AB')
需要地 = list('PQ')
工場 = list('XY')
レーン = (2, 2)

# DataFrameを作っていく

# 輸送費用 #
tbdi = pd.DataFrame(((j, k) for j in 需要地 for k in 工場), columns=['需要地', '工場'])  # (P需要地, X工場), (P, Y), ...
tbdi['輸送費'] = [1, 2, 3, 1]  # X工場からP需要地がコスト1
print(tbdi)

# 需要表 #
tbde = pd.DataFrame(((j, i) for j in 需要地 for i in 製品), columns=['需要地', '工場'])  # (P需要地, A製品), (P, B), ...
tbde['需要'] = [10, 10, 20, 20]  # P需要地はA製品を10個欲している
print(tbde)

# 生産表 #（工場とレーンと製品の組み合わせ）
tbfa = pd.DataFrame(((k, l, i, 0, np.inf) for k, nl in zip(工場, レーン) for l in range(nl) for i in 製品),
                    columns=['工場', 'レーン', '製品', '下限', '上限'])  # (X工場, 0, A製品, 0, np.inf), (X工場, 0, B製品, 0, np.inf)
                    # , ..., (X工場, 1, A製品, 0, np.inf), ..., (Y工場, 0, A製品, 0, np.inf), ... 2*2*2=8個
tbfa['生産費'] = [1, np.nan, np.nan, 1, 3, np.nan, 5, 3]  # 8個
tbfa.dropna(inplace=True)  # 生産費nanのものは切り捨てる
tbfa.loc[4, '上限'] = 10
print(tbfa)

_, tbdi2, _ = logistics_network(tbde, tbdi, tbfa)  # 需要，輸送費，生産，最適設計を行う
print(tbfa)  # ValY：最適生産量
print(tbdi2)  # ValX：最適輸送量


# ノック６９：最適ネットワークにおける輸送コストとその内訳を計算しよう
print(tbdi2)
trans_cost = 0
for i in range(len(tbdi2.index)):
    trans_cost += tbdi2["輸送費"].iloc[i] * tbdi2["ValX"].iloc[i]
print("総輸送コスト：" + str(trans_cost))  # 80万円


# ノック７０：最適ネットワークにおける生産コストとその内訳を計算しよう
print(tbfa)
product_cost = 0
for i in range(len(tbfa.index)):
    product_cost += tbfa["生産費"].iloc[i] * tbfa["ValY"].iloc[i]
print("総生産コスト：" + str(product_cost))  # 120万円

"""最適化計算のライブラリは便利だが，鵜呑みにすると大けがをする．常に疑う姿勢を持つことが大事．"""
