# 8章_数値シミュレーションで消費者行動を予測する10本ノック

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



# ノック７１：人間関係のネットワークを可視化してみよう
df_links = pd.read_csv("links.csv")

G = nx.Graph()

NUM = len(df_links.index)  # ノードの数
for i in range(1, NUM + 1):
    node_no = df_links.columns[i].strip("Node")  # 文字"Node"を除く，0行目は","なので除く
    # print(node_no)
    G.add_node(str(node_no))  # 頂点の設定

for i in range(NUM):
    for j in range(NUM):
        # print(i, j)
        if df_links.iloc[i][j] == 1:
            G.add_edge(str(i), str(j))

# draw_networkx：リンクの多いものが中心に集まるように，自動的にノードの位置を決定して可視化する（再現性なし）
nx.draw_networkx(G, node_color="k", edge_color="k", font_color="w")
plt.show()


# ノック７２：口コミによる情報伝播の様子を可視化しよう
def determine_link(percent):  # 口コミするかを確率(percent)で決定する
    rand_val = np.random.rand()
    if rand_val <= percent:
        return 1
    else:
        return 0



# ノック７３：


# ノック７４：


# ノック７５：


# ノック７６：


# ノック７７：


# ノック７８：


# ノック７９：


# ノック８０：