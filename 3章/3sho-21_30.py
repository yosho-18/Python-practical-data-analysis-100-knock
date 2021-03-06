# 3章_顧客の全体像を把握する１０本ノック（結果を出す，事前分析）

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta  # 日付の比較に使用
# 警告(worning)の非表示化
import warnings
warnings.filterwarnings('ignore')



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
customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])
customer_newer = customer_join.loc[(customer_join["end_date"] >= pd.to_datetime("20190331")) | (customer_join["end_date"].isna())]  # 退会日が20190331以降か，退会予定が無い人を抽出
print(len(customer_newer))  # 2953
print(customer_newer["end_date"].unique())  # NaT：datetime型の欠損値と2019-03-31のみ表示

print(customer_newer.groupby("class_name").count()["customer_id"])  # 辞めない顧客の会員区分
print(customer_newer.groupby("campaign_name").count()["customer_id"])  # キャンペーン区分
print(customer_newer.groupby("gender").count()["customer_id"])  # 性別毎
"""キャンペーン区分において，通常72%から81%になっている"""


"""会員と性別の区分がほぼ変わっていないので，利用履歴データを使っていく（時間的な要素の分析が可能）"""
# ノック２５：
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["年月", "customer_id"], as_index=False).count()  # as_index=Falseでインデックスにならないようにする
uselog_months.rename(columns={"log_id":"count"}, inplace=True)  # "log_id"を"count"に名称変更する
del uselog_months["usedate"]  # "usedate"列を削除
print(uselog_months.head())

uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min"])["count"]  # その顧客の一ヶ月の訪問回数の平均，中央値，最大値，最小値
uselog_customer = uselog_customer.reset_index(drop=False)  # indexの振りなおし
print(uselog_customer.head())


# ノック２６：利用履歴データから定期利用フラグを作成しよう
uselog["weekday"] = uselog["usedate"].dt.weekday  # 曜日の計算を行う，0から6が付与される
uselog_weekday = uselog.groupby(["customer_id", "年月", "weekday"], as_index=False).count()[["customer_id", "年月", "weekday", "log_id"]]  # "log_id"をカウントする
uselog_weekday.rename(columns={"log_id":"count"}, inplace=True)
print(uselog_weekday.head())

uselog_weekday = uselog_weekday.groupby("customer_id", as_index=False).max()[["customer_id", "count"]]  # 顧客毎に，各月の週ごとの最大値をとる
# uselog_weekday = uselog_weekday.max()[["customer_id", "count"]]
uselog_weekday["routine_flg"] = 0
uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"] < 4, 1)  # whereで条件に当てはま「らない」ものに1を代入する
print(uselog_weekday.head())


# ノック２７：顧客データと利用履歴データを結合しよう
print(uselog_customer.head())
print(uselog_weekday.head())
print(customer_join.head())
customer_join = pd.merge(customer_join, uselog_customer, on="customer_id", how="left")
customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on="customer_id", how="left")
print(customer_join.head())
print(customer_join.isnull().sum())


# ノック２８：会員期間を計算しよう
customer_join["calc_date"] = customer_join["end_date"]
customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))  # 欠損値に"20190430"を代入
customer_join["membership_period"] = 0
for i in range(len(customer_join)):
    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])  # 差分を計算，ilocで行指定
    customer_join["membership_period"].iloc[i] = delta.years * 12 + delta.months  # 月単位で算出
print(customer_join.head())


"""利用履歴を加味した上で顧客の分析"""
# ノック２９：顧客行動の各種統計量を把握しよう
customer_join[["mean", "median", "max", "min"]].describe()  # 顧客一人あたり月平均５回通っている
print(customer_join.groupby("routine_flg").count()["customer_id"])  # 0が779，1が3413

plt.hist(customer_join["membership_period"])  # ヒストグラム，10か月以内に離れていく顧客が多い
plt.show()

# ノック３０：退会ユーザーと継続ユーザーの違いを把握しよう

customer_end = customer_join.loc[customer_join["is_deleted"] == 1]
customer_end.describe()

customer_stay = customer_join.loc[customer_join["is_deleted"] == 0]
customer_stay.describe()

customer_join.to_csv("customer_join.csv", index=False)