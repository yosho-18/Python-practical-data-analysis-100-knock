import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


# ノック１１：データを読み込んでみよう
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


# ノック１２：データの揺れを見てみよう


# ノック１３：データに揺れがあるまま集計しよう


# ノック１４：商品名の揺れを補正しよう



# ノック１５：金額欠損値の補完をしよう


# ノック１６：顧客名の揺れを補完しよう


# ノック１７：日付の揺れを補正しよう

# ノック１８：顧客名をキーに２つのデータを結合（ジョイン）しよう

# ノック１９：クレンジングしたデータをダンプしよう



# ノック２０：データを集計しよう
