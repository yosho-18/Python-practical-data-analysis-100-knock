# 3章_顧客の全体像を把握する１０本ノック（結果を出す，機械学習，事前分析）

import pandas as pd
import matplotlib.pyplot as plt


# ノック２１：データを読み込んで把握しよう
uselog = pd.read_csv('use_log.csv')
print(len(uselog), uselog.head())  # 197428，データ件数取得

customer = pd.read_csv('customer_master.csv')
print(len(customer), customer.head())  # 4192

class_master = pd.read_csv('class_master.csv')
print(len(class_master), class_master.head())  # 3

campaign_master = pd.read_csv('campaign_master.csv')
print(len(campaign_master), campaign_master.head())  # 3


# とりあえずcustomerを主に考えていく．
# ノック２２：顧客データを整形しよう
customer_join = pd.merge(customer, class_master, on="class", how="left")
customer_join = pd.merge(customer_join, campaign_master, on="campaign_id", how="left")
print(len(customer_join), customer_join.head())  # 4192

print(customer_join.isnull().sum())  # end_date以外はちゃんとデータが入っている


# ノック２３：顧客データの基礎集計をしよう
print(customer_join.groupby("class_name").count()["customer_id"])
print(customer_join.groupby("campaign_name").count()["customer_id"])
print(customer_join.groupby("gender").count()["customer_id"])
print(customer_join.groupby("is_deleted").count()["customer_id"])
"""いろいろな疑問：キャンペーンはいつ行われたのか，性別と会員クラスの関係，今年度の入会人数
集計確認やヒアリングが大事"""

customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])
customer_start = customer_join.loc[customer_join["start_date"] > pd.to_datetime("20180401")]  # 20180401以降に入会した人
print(len(customer_start))  # 1361


# ノック２４：最新顧客データの基礎集計をしてみよう



# ノック２５：


# ノック２６：


# ノック２７：


# ノック２８：


# ノック２９：


# ノック３０：