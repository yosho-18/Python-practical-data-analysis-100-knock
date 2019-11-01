# 4章_顧客の行動を予測する１０本ノック（機械学習，クラスタリング）

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # K-means法を使用
from sklearn.preprocessing import StandardScaler  # 標準化
from sklearn.decomposition import PCA  # 主成分分析

"""機械学習の詳細についてはAppendix①参照"""

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

print(customer_clustering.groupby(["cluster", "routine_flg"], as_indedx=False)
      .count()[["cluster", "routine_flg", "customer_id"]])  # 定期利用しているかを調べる，0：定期顧客（多）1：定期顧客（多）


# ノック３６：翌日の利用予測を行うための準備をしよう



# ノック３７：


# ノック３８：


# ノック３９：


# ノック４０：