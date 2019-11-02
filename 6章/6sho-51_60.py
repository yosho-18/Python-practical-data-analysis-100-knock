# 6章_物流の最適ルートをコンサルティングする１０本ノック（物の流れ，輸送最適化）

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import networkx as nx


# ノック５１：物流に関するデータを読み込んでみよう
factories = pd.read_csv("tbl_factory.csv", index_col=0)
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
df_tr = pd.read_csv('trans_route.csv', index_col="工場")  # 縦要素の上に「工場」と表示する
print(df_tr.head())  # あの倉庫からその工場はこれだけという情報を総当たり図で表示


# ノック５７：輸送ルート情報からネットワークを可視化してみよう
df_pos = pd.read_csv("trans_route_pos.csv")

G = nx.Graph()



# ノック５８：


# ノック５９：


# ノック６０：