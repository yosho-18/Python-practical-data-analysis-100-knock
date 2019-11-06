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
            G.add_edge(str(i), str(j))  # 辺の設定

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

def simulate_percolation(num, list_active, percent_percolation):  # 口コミをシミュレートする
    for i in range(num):
        if list_active[i] == 1:
            for j in range(num):
                if df_links.iloc[i][j] == 1:  # iの人とjの人がSNSでつながっているかをチェックする
                    if determine_link(percent_percolation) == 1:  # percent_percolationの確率で口コミが広がる
                        list_active[j] = 1
    return list_active

percent_percolation = 0.1  # 口コミの起こる確率（１か月）
T_NUM = 100  # 100カ月繰り返す
NUM = len(df_links.index)  # num：人数
list_active = np.zeros(NUM)  # list_active：[1, 0, ..., 0]
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):  # 100カ月繰り返す
    list_active = simulate_percolation(NUM, list_active, percent_percolation)  # list_activeを更新
    list_timeSeries.append(list_active.copy())  # 0から99カ月のlist_activeを格納，上書きを防ぐため.copy()をする

def active_node_coloring(list_active):
    # print(list_timeSeries[t])
    list_color = []
    for i in range(len(list_timeSeries[t])):
        if list_timeSeries[t][i] == 1:
            list_color.append("r")  # 口コミが伝わっている人は赤
        else:
            list_color.append("k")  # そうでない人は
    # print(len(list_color))
    return list_color

t = 0  # 最初
nx.draw_networkx(G, font_color="w", node_color=active_node_coloring(list_timeSeries[t]))
plt.show()  # 最初は一人

t = 10  # 途中
nx.draw_networkx(G, font_color="w", node_color=active_node_coloring(list_timeSeries[t]))
plt.show()  # 徐々に増えていく（36か月くらいで全員に知れ渡る）

t = 99  # 最後
nx.draw_networkx(G, font_color="w", node_color=active_node_coloring(list_timeSeries[t]))
plt.show()  # 最後は全員に知れ渡る


# ノック７３：口コミ数の時系列変化をグラフ化してみよう
list_timeSeries_num = []
for i in range(len(list_timeSeries)):  # 100
    list_timeSeries_num.append(sum(list_timeSeries[i]))  # その期間での口コミが広まった人を格納

plt.plot(list_timeSeries_num)
plt.show()


# ノック７４：会員数の時系列変化をシミュレーションしてみよう
def simulate_population(num, list_active, percent_percolation, percent_disapparence, df_links):
    # 拡散 #
    for i in range(num):
        if list_active[i] == 1:
            for j in range(num):
                if df_links.iloc[i][j] == 1:
                    if determine_link(percent_percolation) == 1:
                        list_active[j] = 1
    # 消滅 #
    for i in range(num):
        if determine_link(percent_disapparence) == 1:  # percent_disapparencenの確率で退会する
            list_active[i] = 0
    return list_active

# percent_disapparence = 0.05，退会確率が小さいとき
percent_percolation = 0.1
percent_disapparence = 0.05
T_NUM = 100
NUM = len(df_links.index)  # NUM：人数
list_active = np.zeros(NUM)  # 入会していたら1，入会していない退会したならば0
list_active[0] = 1  # 最初は一人だけ入会している

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence, df_links)
    list_timeSeries.append(list_active.copy())

# 時系列グラフを描く
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()  # 会員が増えていく様子が見られる

# percent_disapparence = 0.2，退会確率が大きいとき
percent_percolation = 0.1
percent_disapparence = 0.2
T_NUM = 100
NUM = len(df_links.index)  # NUM：人数，20
list_active = np.zeros(NUM)  # 入会していたら1，入会していない退会したならば0
list_active[0] = 1  # 最初は一人だけ入会している

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence, df_links)
    list_timeSeries.append(list_active.copy())

# 時系列グラフを描く
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()  # 会員が減っていく様子が見られる


# ノック７５：パラメータの全体像を，「相図」を見ながら把握しよう
"""相図：将来的に見てどうなっていくかという普及の様子を俯瞰する
ここからの処理は計算時間を要する"""
"""T_NUM = 100
NUM_PhaseDiagram = 20
phaseDiagram = np.zeros((NUM_PhaseDiagram, NUM_PhaseDiagram))  # 20 * 20
for i_p in range(NUM_PhaseDiagram):  # 口コミ確率を変えていく，20
    for i_d in range(NUM_PhaseDiagram):  # 退会確率を変えていく，20
        percent_percolation = 0.05 * i_p
        percent_disapparence = 0.05 * i_d
        list_active[0] = 1
        for t in range(T_NUM):  # 100カ月
            list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence, df_links)
        phaseDiagram[i_p][i_d] = sum(list_active)  # 100カ月後にどうなっているかを表す

plt.matshow(phaseDiagram)  # 20 * 20の表を作成
plt.colorbar(shrink=0.8)
plt.xlabel('percent_disapparence')
plt.ylabel('percent_percolation')
plt.xticks(np.arange(0.0, 20.0, 0.5), np.arange(0.0, 1.0, 0.25))  # (0.0, 20.0, 0.5)を(0.0, 1.0, 0.25)に変換？
plt.yticks(np.arange(0.0, 20.0, 0.5), np.arange(0.0, 1.0, 0.25))
plt.tick_params(bottom=False, left=False, right=False, top=False)
plt.show()
"""

# ノック７６：実データを読み込んでみよう
df_mem_links = pd.read_csv("links_members.csv")  # 540人のそれぞれのSNSでのつながりを表す
df_mem_info = pd.read_csv("info_members.csv")  # 540人の24カ月の利用履歴，利用率を１，非利用率を０で表す


# ノック７７：リンク数の分布を可視化しよう
"""スモールワールド型，スケールフリー型"""
NUM = len(df_mem_links.index)  # 540人
array_linkNum = np.zeros(NUM)
for i in range(NUM):
    array_linkNum[i] = sum(df_mem_links["Node" + str(i)])  # 人iが繋がっている人の数の総和

plt.hist(array_linkNum, bins=10, range=(0, 250))  # リンク数がおおむね100程度の正規分布に近い形，ヒストグラム
plt.show()


# ノック７８：シミュレーションのために実データからパラメータを推定しよう
"""精度の良いシミュレーションができれば，そのまま将来予測に使える
パラメータを実データを用いて推測する．今回だとpercent_percolation（口コミ），percent_disapparence（退会）"""
NUM = len(df_mem_info.index)  # 540人
T_NUM = len(df_mem_info.columns) - 1  # 24カ月，0行目の顧客名のところを除く

# 消滅の確率推定 #
count_active = 0
count_active_to_inactive = 0
for t in range(1, T_NUM):
    for i in range(NUM):
        if (df_mem_info.iloc[i][t] == 1):  # その月ジムに通っているか
            count_active_to_inactive += 1
            if (df_mem_info.iloc[i][t + 1] == 0):  # 翌月は通っていない
                count_active += 1
estimated_percent_disapparence = count_active / count_active_to_inactive  # 活性（１）だったものが非活性（０）に変化した割合を数える
print(estimated_percent_disapparence)

# 拡散の確率推定 #
count_link = 0
count_link_to_active = 0
count_link_temp = 0
print(df_mem_info[str(3)] == 1)  # 各月tに対し，それぞれの人のTrue, Falseの列
print(df_mem_info[df_mem_info[str(3)] == 1])  # Trueの人の行を取り出す
for t in range(T_NUM - 1):  # 23ヶ月
    df_link_t = df_mem_info[df_mem_info[str(t)] == 1]  # 各月tに対し，それぞれの人のTrue, Falseの列
    temp_flag_count = np.zeros(NUM)  # 540人分のflag
    for i in range(len(df_link_t.index)):  # その月tに対して，通っていた人の人数
        df_link_temp = df_mem_links[df_mem_links["Node" + str(df_link_t.index[i])] == 1]  # 各人iに対し，それぞれの人のSNSでの繋がりを表す列，Trueの行を取り出す
        for j in range(len(df_link_temp.index)):  # その人iとSNSで繋がっている人の人数
            if (df_mem_info.iloc[df_link_temp.index[j]][t] == 0):  # iの友達jの期間tにおいて，ジムに通っていなかったら
                if (temp_flag_count[df_link_temp.index[j]] == 0):  # その人のtemp_flag_countが立ってなかったら
                    count_link += 1
                if (df_mem_info.iloc[df_link_temp.index[j]][t + 1] == 1):  # iの友達jの期間t + 1において，ジムに通っていたら
                    if (temp_flag_count[df_link_temp.index[j]] == 0):  # その人のtemp_flag_countが立ってなかったら
                        temp_flag_count[df_link_temp.index[j]] = 1  # その人のtemp_flag_countを立てる，一回きり
                        count_link_to_active += 1
estimated_percent_percolation = count_link_to_active / count_link  # 非活性だったものが活性に変化した割合を数える
print(estimated_percent_percolation)


# ノック７９：実データとシミュレーションを比較しよう
percent_percolation = 0.02518466132375185
percent_disapparence = 0.10147163541419416
T_NUM = 24
NUM = len(df_mem_links.index)  # 540人
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence, df_mem_links)
    list_timeSeries.append(list_active.copy())  # その期間での口コミが広まった人を配列の0,1を使って格納

list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))  # その期間でのジムに通っている人数を格納

T_NUM = len(df_mem_info.columns) - 1
list_timeSeries_num_real = []
for t in range(0, T_NUM):
    list_timeSeries_num_real.append(len(df_mem_info[df_mem_info[str(t)] == 1].index))  # 実際にその期間tのときにジムに通っている人数を格納

plt.plot(list_timeSeries_num, label='simulated')
plt.plot(list_timeSeries_num_real, label='real')
plt.xlabel('month')
plt.ylabel('population')
# シミュレーションと現実にさほど差はない


# ノック８０：シミュレーションによる将来予測を実施しよう
percent_percolation = 0.02518466132375185
percent_disapparence = 0.10147163541419416
T_NUM = 36  # 未来を予測する
NUM = len(df_mem_links.index)  # 540人
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence, df_mem_links)
    list_timeSeries.append(list_active.copy())  # その期間での口コミが広まった人を配列の0,1を使って格納

list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))  # その期間でのジムに通っている人数を格納

T_NUM = len(df_mem_info.columns) - 1
list_timeSeries_num_real = []
for t in range(0, T_NUM):
    list_timeSeries_num_real.append(len(df_mem_info[df_mem_info[str(t)] == 1].index))  # 実際にその期間tのときにジムに通っている人数を格納

plt.plot(list_timeSeries_num, label='simulated')
plt.plot(list_timeSeries_num_real, label='real')
plt.xlabel('month')
plt.ylabel('population')
# 急激な立下りが無く継続することが確認できる