# 1章_ウェブからの注文数を分析する１０本ノック（整ったデータ）

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


# ノック１：データを読み込んでみよう
customer_master = pd.read_csv("customer_master.csv")
print(customer_master.head())

item_master = pd.read_csv("item_master.csv")
print(item_master.head())
transaction_1 = pd.read_csv("transaction_1.csv")
print(transaction_1.head())
transaction_detail_1 = pd.read_csv("transaction_detail_1.csv")
print(transaction_detail_1.head())

"""motivation
最もデータ粒度の細かいtransaction_detailを中心にしてデータをunionまたはjoinしていく"""


# ノック２：データを結合（ユニオン）してみよう
transaction_2 = pd.read_csv("transaction_2.csv")
transaction = pd.concat([transaction_1, transaction_2], ignore_index=True)  # データを縦に結合
print(transaction.head())
print(len(transaction_1), len(transaction_2), len(transaction))  # 5000,1786,6786

transaction_detail_2 = pd.read_csv("transaction_detail_2.csv")
transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2], ignore_index=True)  # データを縦に結合
print(transaction_detail.head())
print(len(transaction_detail_1), len(transaction_detail_2), len(transaction_detail))  # 5000,2144,7144


# ノック３：売上データ同士を結合（ジョイン）してみよう
"""キーとするデータ列，追加するデータ列を考える"""
join_data = pd.merge(transaction_detail, transaction[["transaction_id", "payment_date", "customer_id"]],
                     on="transaction_id", how="left")  # データを横に結合，onでキーを決める
print(join_data.head())
print(len(transaction_detail), len(transaction), len(join_data))  # 7144,6786,7144

print(transaction[["transaction_id", "price"]])  # transaction_id：[]だと付かない，複数アウト
print(transaction[23:24])  # 行はスライスで指定


# ノック４：マスターデータを結合（ジョイン）してみよう
join_data = pd.merge(join_data, customer_master, on="customer_id", how="left")
join_data = pd.merge(join_data, item_master, on="item_id", how="left")
print(join_data.head())


# ノック５：必要なデータ列を作ろう
join_data["price"] = join_data["quantity"] * join_data["item_price"]  # detailについての合計金額を求める
print(join_data[["quantity", "item_price", "price"]].head())
"""件数の確認，検算が大事！！"""


# ノック６：データ演算をしよう
print(join_data["price"].sum(), join_data["price"].sum() == transaction["price"].sum())  # 971135000, True


# ノック７：各種統計量を把握しよう
"""欠損値の状況，全体の数字感を知る必要がある"""
print(join_data.isnull().sum())  # 今回は無し
print(join_data.describe())  # 各種統計量（平均，最大値など）を簡単に見られる

print(join_data["payment_date"].min(), join_data["payment_date"].max())  # 期間の確認


# ノック８：月別でデータを集計してみよう
print(join_data.dtypes)
join_data["payment_date"] = pd.to_datetime(join_data["payment_date"])  # object型をdatetime型に変更する
join_data["payment_month"] = join_data["payment_date"].dt.strftime("%Y%m")  # dtで年のみを抽出，月までを表示するpayment_monthを作る
print(join_data[["payment_date", "payment_month"]].head())
print(join_data.groupby("payment_month").sum()["price"])  # グループ化して，集計方法を指定し，price列を表示する，月別売上を表示する


# ノック９：月別，商品別でデータを集計してみよう
"""まとめたい列が複数の時はリストで指定しなければならない！！"""
print(join_data.groupby(["payment_month", "item_name"]).sum()[["price", "quantity"]])  # 月で分けてから商品で分ける

print(pd.pivot_table(join_data, index="item_name", columns="payment_month",
                     values=['price', 'quantity'], aggfunc='sum'))  # データを表にして見やすくする，columns：列
"""現場で適切に運用され施策を回していくことが大事"""


# ノック１０：商品別の売上推移を可視化してみよう
graph_data = pd.pivot_table(join_data, index="payment_month", columns="item_name",
                            values='price', aggfunc='sum')  # 横軸にpayment_month，縦軸にitem_nameが可能になる
graph_data.head()

plt.plot(list(graph_data.index), graph_data["PC-A"], label='PC-A')  # 月ごとの売り上げをグラフ化する
plt.plot(list(graph_data.index), graph_data["PC-B"], label='PC-B')
plt.plot(list(graph_data.index), graph_data["PC-C"], label='PC-C')
plt.plot(list(graph_data.index), graph_data["PC-D"], label='PC-D')
plt.plot(list(graph_data.index), graph_data["PC-E"], label='PC-E')
plt.legend()  # loc='lower right'
plt.show()