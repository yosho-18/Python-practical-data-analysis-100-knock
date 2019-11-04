# 4章_顧客の行動を予測する１０本ノック（機械学習，クラスタリング）

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # K-means法を使用
from sklearn.preprocessing import StandardScaler  # 標準化
from sklearn.decomposition import PCA  # 主成分分析
from dateutil.relativedelta import relativedelta
from sklearn import linear_model  # 線形回帰モデル
import sklearn.model_selection  # 学習用データと評価用データを分ける

"""機械学習の詳細についてはAppendix②参照"""


# ノック３１：データを読み込んで確認しよう
uselog = pd.read_csv('use_log.csv')
print(uselog.isnull().sum())

customer = pd.read_csv('customer_join.csv')
print(customer.isnull().sum())


# ノック３２：クラスタリングで顧客をグループ化しよう
customer_clustering = customer[["mean", "median", "max", "min", "membership_period"]]  # クラスタリングに必要な変数に絞る
print(customer_clustering.head())

"""K-means法：変数間の距離をベースにグループ化を行う，予めグルーピングしたい数を指定する（４つ）
[mean, median, max, min]と[membership_period]ではデータが大きく異なる，数の大小を統一するため標準化が必要"""
sc = StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering)  # 標準化を実行
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(customer_clustering_sc)  # K-meansを実行
customer_clustering["cluster"] = clusters.labels_  # クラスタリングの結果を反映
print(customer_clustering["cluster"].unique())  # [1 2 3 0]
print(customer_clustering.head())


# ノック３３：クラスタリング結果を分析しよう
customer_clustering.columns = ["月内平均値", "月内中央値", "月内最大値", "月内最小値", "会員期間", "cluster"]  # 列名変更
# print(customer_clustering.head())
print(customer_clustering.groupby("cluster").count())  # countなので全ての列で同じ値，0：841，1：1248，2：771，3：1332
print(customer_clustering.groupby("cluster").mean())  # 0：会員期間（短），利用率（高），1，2：会員期間（短），利用率（低），3


# ノック３４：クラスタリング結果を可視化してみよう
"""次元削減（ここでは５次元から２次元），その中でも主成分分析を今回は行う
一般的には，２つの軸がどの変数から成り立っているかを分析すると，軸の意味付けが可能"""
X = customer_clustering_sc  # customer_clusteringを標準化したもの
pca = PCA(n_components=2)  # ２次元にする
i = pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = customer_clustering["cluster"]  # 先ほどの"cluster"をくっつける

for i in customer_clustering["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"] == i]
    plt.scatter(tmp[0], tmp[1])
plt.show()


# ノック３５：クラスタリング結果をもとに退会顧客の傾向を把握しよう
customer_clustering = pd.concat([customer_clustering, customer], axis=1)  # indexで紐づけされている，列の結合（実質merge）
pp = customer_clustering.groupby(["cluster", "is_deleted"], as_index=False).count()[["cluster", "is_deleted", "customer_id"]]
print(pp.head())  # クラスタ毎での継続顧客と退会顧客の数を表示，0：継続顧客（多）1：継続顧客（多）2：継続顧客（少）3：継続顧客（普）
# 0：初期活発？，1：継続安定？

print(customer_clustering.groupby(["cluster", "routine_flg"], as_index=False)
      .count()[["cluster", "routine_flg", "customer_id"]])  # 定期利用しているかを調べる，0：定期顧客（多）1：定期顧客（多）


# ノック３６：翌日の利用予測を行うための準備をしよう
"教師あり学習"
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["年月", "customer_id"], as_index=False).count()  # その月のその顧客の訪問回数を集計
uselog_months.rename(columns={"log_id": "count"}, inplace=True)
del uselog_months["usedate"]
print(uselog_months.head())  # ノック２５とほぼ同じ

year_months = list(uselog_months["年月"].unique())  # 201804から201903
predict_data = pd.DataFrame()
for i in range(6, len(year_months)):
    tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]]  # 目標とする月の情報を取得
    tmp.rename(columns={"count": "count_pred"}, inplace=True)
    for j in range(1, 7):
        tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i - j]]  # 過去６か月分のデータを取る
        del tmp_before["年月"]  # その年月を消す
        tmp_before.rename(columns={"count": "count_{}".format(j - 1)}, inplace=True)  # 0：１か月前，
        tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    predict_data = pd.concat([predict_data, tmp], ignore_index=True)
print(predict_data.head())  # NaNが出ていたら６か月以前に退会している

predict_data = predict_data.dropna()  # 欠損値データ除去
predict_data = predict_data.reset_index(drop=True)  # indexを初期化
print(predict_data.head())


# ノック３７：特長となる変数を付与しよう
predict_data = pd.merge(predict_data, customer[["customer_id", "start_date"]], on="customer_id", how="left")
print(predict_data.head())

predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
predict_data["period"] = None
for i in range(len(predict_data)):
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    predict_data["period"][i] = delta.years * 12 + delta.months
print(predict_data.head())  # ノック２８とほぼ同じ，会員期間を求める


# ノック３８：来月の利用予測モデルを作成しよう
"""線形回帰モデル"""
predict_data = predict_data.loc[predict_data["start_date"] >= pd.to_datetime("20180401")]  # ４月以降に入会した顧客のみ
model = linear_model.LinearRegression()  # 線形回帰モデルの初期化を行う
X = predict_data[["count_0", "count_1", "count_2", "count_3", "count_4", "count_5", "period"]]  # 説明（予測に使う）変数
y = predict_data["count_pred"]  # 目的（予測したい）変数
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)  # 無指定の場合，学習用データ75%，評価用データ25%
model.fit(X_train, y_train)

print(model.score(X_train, y_train))  # 0.62
print(model.score(X_test, y_test))  # 0.57


# ノック３９：モデルに寄与している変数を確認しよう
coef = pd.DataFrame({"feature_names": X.columns, "coefficient": model.coef_})  # 説明変数と係数が出力される，直近のものほど影響が大きい
print(coef)


# ノック４０：来月の利用回数を予測しよう
x1 = [3, 4, 4, 6, 8, 7, 8]
x2 = [2, 2, 3, 3, 4, 6, 8]
x_pred = [x1, x2]
print(model.predict(x_pred))  # [3.73461089 1.92463185]

uselog_months.to_csv("use_log_months.csv", index=False)
