# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request  # Python 3 の場合

# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
from pandas.tools import plotting

# 機械学習関連のライブラリ群

from sklearn.cross_validation import train_test_split  # 訓練データとテストデータに分割
from sklearn.metrics import confusion_matrix  # 混合行列

from sklearn.decomposition import PCA  # 主成分分析
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰
from sklearn.neighbors import KNeighborsClassifier  # K近傍法
from sklearn.svm import SVC  # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier  # 決定木
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.naive_bayes import GaussianNB  # ナイーブ・ベイズ

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# from sklearn.lda import LDA # 線形判別分析
# from sklearn.qda import QDA # 二次判別分析

# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/chemo-wakate/tutorial-6th/master/beginner/data/winequality-red.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'winequality-red.csv') # Python 2 の場合
urllib.request.urlretrieve(url, 'winequality-red.txt')  # Python 3 の場合

# データの読み込み
df1 = pd.read_csv('winequality-red.txt', sep='\t', index_col=0)

print(df1.head())  # 先頭５行だけ表示

# quality が 6 未満の行を抜き出して、先頭５行を表示する
df1[df1['quality'] < 6].head()
# quality が 6 以上の行を抜き出して、先頭５行を表示する
df1[df1['quality'] >= 6].head()
fig, ax = plt.subplots(1, 1)

# quality が 6 未満の行を抜き出して、x軸を volatile acidity 、 y軸を alcohol として青色の丸を散布する
df1[df1['quality'] < 6].plot(kind='scatter', x=u'volatile acidity', y=u'alcohol',
                             ax=ax, c='blue', alpha=0.5)

# quality が 6 以上の行を抜き出して、x軸を volatile acidity 、 y軸を alcohol として赤色の丸を散布する
df1[df1['quality'] >= 6].plot(kind='scatter', x=u'volatile acidity', y=u'alcohol',
                              ax=ax, c='red', alpha=0.5, grid=True, figsize=(5, 5))
plt.show()

# quality が 6 未満のものを青色、6以上のものを赤色に彩色して volatile acidity の分布を描画
df1[df1['quality'] < 6]['volatile acidity'].hist(figsize=(3, 3), bins=20, alpha=0.5, color='blue')
df1[df1['quality'] >= 6]['volatile acidity'].hist(figsize=(3, 3), bins=20, alpha=0.5, color='red')
plt.show()
# quality が 6 未満のものを青色、6以上のものを赤色に彩色して pH の分布を描画
df1[df1['quality'] < 6]['pH'].hist(figsize=(3, 3), bins=20, alpha=0.5, color='blue')
df1[df1['quality'] >= 6]['pH'].hist(figsize=(3, 3), bins=20, alpha=0.5, color='red')
plt.show()

df1['class'] = [0 if i <= 5 else 1 for i in df1['quality'].tolist()]
print(df1.head())  # 先頭５行を表示

# それぞれに与える色を決める。
color_codes = {0: '#0000FF', 1: '#FF0000'}
colors = [color_codes[x] for x in df1['class'].tolist()]

plotting.scatter_matrix(df1.dropna(axis=1)[df1.columns[:11]], figsize=(20, 20), color=colors, alpha=0.5)
plt.show()

dfs = df1.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)  # データの正規化

pca = PCA()
pca.fit(dfs.iloc[:, :11])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :11])
# plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.5, color=colors)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component")
plt.ylabel("The second principal component")
plt.grid()
plt.show()

X = dfs.iloc[:, :11]  # 説明変数
y = df1.iloc[:, 12]  # 目的変数
print(X.head())  # 先頭５行を表示して確認

print(pd.DataFrame(y).T)  # 目的変数を確認。縦に長いと見にくいので転置して表示。

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)  # 訓練データ・テストデータへのランダムな分割
print(X_train.head())  # 先頭５行を表示して確認

print(pd.DataFrame(y_train).T)  # 縦に長いと見にくいので転置して表示

clf = LogisticRegression()  # モデルの生成
clf.fit(X_train, y_train)  # 学習
# 正解率 (train) : 学習に用いたデータをどのくらい正しく予測できるか
clf.score(X_train, y_train)
# 正解率 (test) : 学習に用いなかったデータをどのくらい正しく予測できるか
clf.score(X_test, y_test)

y_predict = clf.predict(X_test)
print(pd.DataFrame(y_predict).T)

# 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列
pd.DataFrame(confusion_matrix(y_predict, y_test), index=['predicted 0', 'predicted 1'], columns=['real 0', 'real 1'])

names = ["Logistic Regression", "Nearest Neighbors",
         "Linear SVM", "Polynomial SVM", "RBF SVM", "Sigmoid SVM",
         "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes",
         "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="poly"),
    SVC(kernel="rbf"),
    SVC(kernel="sigmoid"),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

result = []
for name, clf in zip(names, classifiers):  # 指定した複数の分類機を順番に呼び出す
    clf.fit(X_train, y_train)  # 学習
    score1 = clf.score(X_train, y_train)  # 正解率（train）の算出
    score2 = clf.score(X_test, y_test)  # 正解率（test）の算出
    result.append([score1, score2])  # 結果の格納

# test の正解率の大きい順に並べる
# df_result = pd.DataFrame(result, columns=['train', 'test'], index=names).sort('test', ascending=False)
# もし上のコードが動かない場合、以下のコーを試してみてください。
df_result = pd.DataFrame(result, columns=['train', 'test'], index=names).sort_values('test', ascending=False)
print(df_result)

# 棒グラフの描画
df_result.plot(kind='bar', alpha=0.5, grid=True)
plt.show()

result = []
for trial in range(20):  # 20 回繰り返す
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)  # 訓練データ・テストデータの生成
    for name, clf in zip(names, classifiers):  # 指定した複数の分類機を順番に呼び出す
        clf.fit(X_train, y_train)  # 学習
        score1 = clf.score(X_train, y_train)  # 正解率（train）の算出
        score2 = clf.score(X_test, y_test)  # 正解率（test）の算出
        result.append([name, score1, score2])  # 結果の格納

df_result = pd.DataFrame(result, columns=['classifier', 'train', 'test'])  # 今回はまだ並べ替えはしない
print(df_result)  # 結果の確認。同じ分類器の結果が複数回登場していることに注意。
# 分類器 (classifier) 毎にグループ化して正解率の平均を計算し、test の正解率の平均の大きい順に並べる
# df_result_mean = df_result.groupby('classifier').mean().sort('test', ascending=False)
# もし上のコードが動かない場合、以下のコーを試してみてください。
df_result_mean = df_result.groupby('classifier').mean().sort_values('test', ascending=False)
print(df_result_mean)  # 結果の確認
# エラーバーの表示に用いる目的で、標準偏差を計算する
errors = df_result.groupby('classifier').std()
print(errors)  # 結果の確認

# 平均値と標準偏差を用いて棒グラフを描画
df_result_mean.plot(kind='bar', alpha=0.5, grid=True, yerr=errors)
plt.show()