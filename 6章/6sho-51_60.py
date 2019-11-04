# 6章_物流の最適ルートをコンサルティングする１０本ノック（物の流れ，輸送最適化）

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import networkx as nx
import numpy as np

# ノック５１：物流に関するデータを読み込んでみよう
factories = pd.read_csv("tbl_factory.csv", index_col=0)  # 0列目をindex指定
print(factories)

warehouses = pd.read_csv("tbl_warehouse.csv", index_col=0)
print(warehouses)

cost = pd.read_csv("rel_cost.csv", index_col=0)
print(cost.head())

trans = pd.read_csv("tbl_transaction.csv", index_col=0)
print(trans.head())

join_data = pd.merge(trans, cost, left_on=["ToFC", "FromWH"],
                     right_on=["FCID", "WHID"], how="left")  # trans（輸送実績）を主体にする，costを加える
print(join_data.head())

join_data = pd.merge(join_data, factories, left_on="ToFC", right_on="FCID", how="left")  # factoriesも加える
print(join_data.head())

join_data = pd.merge(join_data, warehouses, left_on="FromWH", right_on="WHID", how="left")  # factoriesも加える
join_data = join_data[["TransactionDate", "Quantity", "Cost", "ToFC", "FCName", "FCDemand",
                       "FromWH", "WHName", "WHSupply", "WHRegion"]]  # 見やすいように並び替え，重複したキーを指定しない
print(join_data.head())

kanto = join_data.loc[join_data["WHRegion"] == "関東"]  # 関東は関東に，東北は東北に運んでいるとしている
print(kanto.head())

tohoku = join_data.loc[join_data["WHRegion"] == "東北"]
print(tohoku.head())


# ノック５２：現状の輸送量，コストを確認してみよう
print("関東支社の総コスト：" + str(kanto["Cost"].sum()) + "万円")  # １年間の関東支社の総コスト，2189.3個
print("東北支社の総コスト：" + str(tohoku["Cost"].sum()) + "万円")  # 2060.5個

print("関東支社の総部品輸送個数：" + str(kanto["Quantity"].sum()) + "個")  # １年間の関東支社の総部品輸送個数，49146個
print("東北支社の総部品輸送個数：" + str(tohoku["Quantity"].sum()) + "個")  # 50214個

tmp = (kanto["Cost"].sum() / kanto["Quantity"].sum()) * 10000
print("関東支社の部品1つ当たりの輸送コスト：" + str(int(tmp)) + "円")  # 関東支社の部品1つ当たりの輸送コスト，445円
tmp = (tohoku["Cost"].sum() / tohoku["Quantity"].sum()) * 10000
print("東北支社の部品1つ当たりの輸送コスト：" + str(int(tmp)) + "円")  # 410円

cost_chk = pd.merge(cost, factories, on="FCID", how="left")
print("関東（東京？）支社の平均輸送コスト：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"] == "関東"].mean())
      + "万円")  # 関東（東京？）支社の平均輸送コスト，1.0751万円
print("東北支社の平均輸送コスト：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"] == "東北"].mean())
      + "万円")  # 1.05万円
"""各支社の平均輸送コストはほぼ同じ，関東支社より東北支社の方が「効率よく」部品の輸送が行えている"""


# ノック５３：ネットワークを可視化してみよう
"""「最適化プログラムによって導き出されたプランを可視化するプロセス」
と「条件を実際に満たしているのこと確認するプロセス」が重要"""

# グラフオブジェクトの作成
G = nx.Graph()

# 頂点の設定
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")

# 辺の設定
G.add_edge("nodeA", "nodeB")
G.add_edge("nodeA", "nodeC")
G.add_edge("nodeB", "nodeC")

# 座標の設定
pos = {}
pos["nodeA"] = (0, 0)
pos["nodeB"] = (1, 1)
pos["nodeC"] = (0, 1)

# 絵画
nx.draw(G, pos)  # ここで，"nodeA"と座標をリンクさせる

# 表示
plt.show()


# ノック５４：ネットワークにノード（頂点）を追加してみよう
G.add_node("nodeD")
G.add_edge("nodeA", "nodeD")
pos["nodeD"] = (1, 0)
nx.draw(G, pos, with_labels=True)  # グラフ上に"nodeA"と表示される
plt.show()


# ノック５５：ルートの重みづけを実装しよう
"""リンク（エッジ）の太さを変えられる（重みづけを行う）ので，物流の最適ルートが分かりやすく可視化できる"""
# データ読み込み
df_w = pd.read_csv("network_weight.csv")  # リンクの重み
df_p = pd.read_csv("network_pos.csv")  # リンクの座標

# エッジの重みのリスト化
size = 10
edge_weights = []
for i in range(len(df_w)):
    for j in range(len(df_w.columns)):
        edge_weights.append(df_w.iloc[i][j] * size)  # 横に進んで，縦に行く．「これ」と

# ここからノック５３，ノック５４と同様に行う
# グラフオブジェクトの作成
G = nx.Graph()

# 頂点の設定
for i in range(len(df_w.columns)):
    G.add_node(df_w.columns[i])  # A, B, ...

# 辺の設定
for i in range(len(df_w.columns)):
    for j in range(len(df_w.columns)):
        G.add_edge(df_w.columns[i], df_w.columns[j])  # 全てのノードの組に辺を張る，「これ」を対応させておく，(A, A)，(A, B), ...

# 座標の設定
pos = {}
for i in range(len(df_w.columns)):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0], df_p[node][1])

# 絵画
nx.draw(G, pos, with_labels=True, font_size=16, node_size=1000,
        node_color='k', font_color='w', width=edge_weights)  # ここで，"nodeA"と座標をリンクさせる

# 表示
plt.show()


# ノック５６：輸送ルート情報を読み込んでみよう
"""ノック５１，ノック５２で見てきた物流データ（の簡略版）を用いて最適化を行うための実践的な流れを開設する"""
df_tr = pd.read_csv('trans_route.csv', index_col="工場")  # 縦要素の上に「工場」と表示する，リンクの重み
print(df_tr.head())  # あの倉庫からその工場はこれだけという情報を総当たり図で表示


# ノック５７：輸送ルート情報からネットワークを可視化してみよう
df_pos = pd.read_csv("trans_route_pos.csv")  # リンクの座標，左側に倉庫W，右側に工場Fが来る

G = nx.Graph()

# 頂点の設定
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])  # W1, W2, ..., F1, F2, ...

# 辺の設定＆エッジの重みのリスト化
num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i == j):  # 同じもの同士は考えない
            # 辺の追加
            G.add_edge(df_pos.columns[i], df_pos.columns[j])  # とりあえず倉庫同士，工場同士のリンクも張る
            # エッジの重みの追加
            if num_pre < len(G.edges):  # i == jの時以外，必ずTrue
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns) and (df_pos.columns[j] in df_tr.index):  # iが工場で，jが倉庫
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:  # 値が0でないならTrue
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]
                elif (df_pos.columns[j] in df_tr.columns) and (df_pos.columns[i] in df_tr.index):  # jが工場で，iが倉庫
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:  # 値が0でないならTrue
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]
                edge_weights.append(weight)

# 座標の設定
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0], df_pos[node][1])

# 絵画
nx.draw(G, pos, with_labels=True, font_size=16, node_size=1000,
        node_color='k', font_color='w', width=edge_weights)

# 表示
plt.show()

"""どの倉庫とどの工場の間に多くの輸送が行われているかがわかる，割とまんべんなくなっているので改善の余地あり？"""


# ノック５８：輸送コスト関数を作成しよう
"""最適化問題，目的関数，制約条件"""
"""仮説：輸送コストを下げられる効率的な輸送ルートがあるのではないか"""
df_tc = pd.read_csv('trans_cost.csv', index_col="工場")

# 輸送コスト関数
def trans_cost(df_tr, df_tc):
    cost = 0
    for i in range(len(df_tc.index)):  # i：倉庫（indexは横）
        for j in range(len(df_tr.columns)):  # j：工場（columnsは縦）
            cost += df_tr.iloc[i][j] * df_tc.iloc[i][j]
    return cost

print("総輸送コスト：" + str(trans_cost(df_tr, df_tc)))  # 現在の総輸送コスト：1433万円


# ノック５９：制約条件を作ってみよう
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 需要側（工場）の制約条件
for i in range(len(df_demand.columns)):  # 工場が求めている（最低限必要な）受け入れる量
    temp_sum = sum(df_tr[df_demand.columns[i]])  # F1, F2, ...（縦（倉庫W毎）にsumを取る）
    print(str(df_demand.columns[i]) + "への輸送量" + str(temp_sum) + "（需要量" + str(df_demand.iloc[0][i]) + "）")
    if temp_sum >= df_demand.iloc[0][i]:  # iloc：int型にしてる？
        print("需要量を満たしています．")
    else:
        print("需要量を満たしていません．輸送ルートを再計算してください．")


# 供給側（倉庫）の制約条件
for i in range(len(df_supply.columns)):  # 倉庫の（上限）出荷量
    temp_sum = sum(df_tr.loc[df_supply.columns[i]])  # W1, W2, ...（横（工場F毎）にsumを取る），.locを使う
    print(str(df_supply.columns[i]) + "からの輸送量" + str(temp_sum) + "（供給限界" + str(df_supply.iloc[0][i]) + "）")
    # k = df_supply[0][i]
    if temp_sum <= df_supply.iloc[0][i]:
        print("供給限界の範囲内です．")
    else:
        print("供給限界を超過しています．輸送ルートを再計算してください．")


# ノック６０：輸送ルートを変更して，輸送コスト関数の変化を確認しよう
df_tr_new = pd.read_csv('trans_route_new.csv', index_col="工場")
print(df_tr_new)

# 総輸送コスト再計算
print("総輸送コスト（変更後）：" + str(trans_cost(df_tr_new, df_tc)))

# 制約条件計算関数

# 需要側（工場）
def condition_demand(df_tr, df_demand):
    flag = np.zeros(len(df_demand.columns))  # F1, F2, F3, F4
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])  # F1, F2, ...（縦（倉庫W毎）にsumを取る）
        if temp_sum >= df_demand.iloc[0][i]:  # 条件を満たせば1が入る
            flag[i] = 1
    return flag

# 供給側（倉庫）
def condition_supply(df_tr, df_supply):
    flag = np.zeros(len(df_supply.columns))  # W1, W2, W3
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])  # W1, W2, ...（横（工場F毎）にsumを取る）
        if temp_sum <= df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("需要条件計算結果：" + str(condition_demand(df_tr_new, df_demand)))  # [1, 1, 1, 1]ならOK
print("供給条件計算結果：" + str(condition_supply(df_tr_new, df_supply)))  # [1, 1, 1]ならOK
