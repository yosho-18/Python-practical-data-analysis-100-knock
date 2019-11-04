# 5章_顧客の退会を予測する１０本ノック（決定木）

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.tree import DecisionTreeClassifier  # 決定木を使用するためのライブラリ
import sklearn.model_selection  # 学習用データと評価用データを分ける


# ノック４１：データを読み込んで利用データを整形しよう
customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')

# 当月と１カ月以内の利用回数から予測（ノック３６参照）
year_months = list(uselog_months["年月"].unique())
uselog = pd.DataFrame()
for i in range(1, len(year_months)):
    tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]]
    tmp.rename(columns={"count": "count_0"}, inplace=True)
    tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i - 1]]  # １か月前のデータを取得
    del tmp_before["年月"]
    tmp_before.rename(columns={"count": "count_1"}, inplace=True)
    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")  # join
    uselog = pd.concat([uselog, tmp], ignore_index=True)  # union
print(uselog.head())


# ノック４２：退会前月の退会顧客のデータを作成しよう
"""2018.8：退会申請，退会前月，2018.9：退会申請済み，退会月，2018.10：退会
退会申請を出す確率を予測する"""
exit_customer = customer.loc[customer["is_deleted"] == 1]
exit_customer["exit_date"] = None
exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])
for i in range(len(exit_customer)):
    exit_customer["exit_date"].iloc[i] = exit_customer["end_date"].iloc[i] - relativedelta(months=1)  # "end-date"の１か月前を計算
exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")  # 「年月」はexit_date
uselog["年月"] = uselog["年月"].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "年月"], how="left")  # ["customer_id", "年月"]両方一致で結合，
print(len(uselog))  # 33851
print(exit_uselog.head())  # 退会した顧客の退会前月のデータのみが残る

exit_uselog = exit_uselog.dropna(subset=["name"])  # subset：指定した列に欠損値があれば，その行を削除（end_dateで全て消されるのを防止）
print(len(exit_uselog))  # 1104
print(len(exit_uselog["customer_id"].unique()))  # 1104
print(exit_uselog.head())



# ノック４３：継続顧客のデータを作成しよう
conti_customer = customer.loc[customer["is_deleted"] == 0]
conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")  # 同じ人でも期間が違うと，複数の継続顧客データになる
print(len(conti_uselog))  # 33851
conti_uselog = conti_uselog.dropna(subset=["name"])
print(len(conti_uselog))  # 27422

"""1104:27422では不均衡なデータになるので，サンプル数を調整する
顧客あたり１件のデータにする"""
conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)  # シャッフルを行う，元のindexを消す（振りなおす）
conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")  # "customer_id"が重複しているデータは最初のみを取る
print(len(conti_uselog))  # 2842
print(conti_uselog.head())

predict_data = pd.concat([conti_uselog, exit_uselog], ignore_index=True)  # 継続顧客データと退会顧客データをunionする
print(len(predict_data))  # 3946
print(predict_data.head())


# ノック４４：予測する月の在籍期間を作成しよう
"""時間的な要素が入ったデータ，在籍期間などのデータを加えるのは良いアプローチ"""
predict_data["period"] = 0
predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
for i in range(len(predict_data)):
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    k = delta.years * 12 + delta.months
    predict_data["period"][i] = int(delta.years * 12 + delta.months)
print(predict_data.head())  # ノック３７とほぼ同じ


# ノック４５：欠損値を除去しよう
print(predict_data.isna().sum())  # end_date,exit_dateは退会顧客しか持っていない，count_1：これがNaNなら削除
predict_data = predict_data.dropna(subset=["count_1"])


# ノック４６：文字列型の変数を処理できるように整形しよう
"""カテゴリアル変数をダミー変数化する
分類：離散，回帰：連続，詳細はAppendix②参照"""
target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]  # 説明変数（"is_deleted"以外）
predict_data = predict_data[target_col]
print(predict_data.head())

predict_data = pd.get_dummies(predict_data)  # カテゴリアル変数をダミー変数化する
print(predict_data.head())
"""男性，女性の列は２つもいらない（女性の列が0なら男性，1なら女性と分かる）ので１つを消去していく"""
del predict_data["campaign_name_通常"]
del predict_data["class_name_ナイト"]
del predict_data["gender_M"]
print(predict_data.head())
"""「データ分析は，データ加工が８割」"""


# ノック４７：決定木を用いて退会予測をモデルを作成してみよう
exit = predict_data.loc[predict_data["is_deleted"] == 1]
conti = predict_data.loc[predict_data["is_deleted"] == 0].sample(len(exit))  # 退会と継続のデータ件数を揃える，1104:1104になる

X = pd.concat([exit, conti], ignore_index=True)  # 同数のexitとcontiをくっつける
y = X["is_deleted"]  # 目的変数
del X["is_deleted"]  # 説明変数にする
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)  # 学習データと評価データの分割
model = DecisionTreeClassifier(random_state=0)  # 決定木モデルを定義
model.fit(X_train, y_train)  # fitに学習データを指定，モデルの構築
y_test_pred = model.predict(X_test)  # 構築したモデルを用いて評価データの予測を行う
print(y_test_pred)  # 0：継続，1：退会

result_test = pd.DataFrame({"y_test": y_test, "y_pred": y_test_pred})  # 正解との比較
print(result_test.head())


# ノック４８：予測モデルの評価を行い，モデルのチューニングをしてみよう
correct = len(result_test.loc[result_test["y_test"] == result_test["y_pred"]])  # 予測と正解が一致していたらOK
data_count = len(result_test)
score_test = correct / data_count
print(score_test)  # 87%

"""学習用データで予測した精度と評価用データで予測した精度の差が小さいのが理想"""
print(model.score(X_test, y_test))  # 87%
print(model.score(X_train, y_train))  # 98%（過学習傾向にある）
"""過学習の対策：データを増やす，変数を見直す，モデルのパラメータを変更する"""

"""決定木：最も綺麗に0と1を分割できる説明変数及びその条件を探す作業を木構造状に派生させていく手法
　　　　　 →分割していく木構造の深さを浅くすることで，モデルを簡易化できる"""
X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
model = DecisionTreeClassifier(random_state=0, max_depth=5)  # 決定木モデルを定義，最大深さを５に指定する
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 92%
print(model.score(X_train, y_train))  # 92%


# ノック４９：モデルに寄与している変数を確認しよう
importance = pd.DataFrame({"feature_names": X.columns,
                           "coefficient": model.feature_importances_})  # model.feature_importances_で重要変数を取得できる
print(importance)  # １か月前の利用回数，在籍期間，定期利用かどうかが大きく寄与している
"""決定木は「木構造の可視化」を行うこともできる（graphvizなど）
　　→直感的にモデルの理解が可能となり，説明もしやすい"""

# ノック５０：顧客の退会を予測しよう
count_1, routing_flg, period, campaign_name, class_name, gender \
    = 3, 1, 10, "入会費無料", "オールタイム", "M"  # ノック４０と同じような感じでやっていく

# カテゴリアル変数使用のため，前処理がいる
if campaign_name == "入会費半額":
    campaign_name_list = [1, 0]
elif campaign_name == "入会費無料":
    campaign_name_list = [0, 1]
elif campaign_name == "通常":
    campaign_name_list = [0, 0]

if class_name == "オールタイム":
    class_name_list = [1, 0]
elif class_name == "デイタイム":
    class_name_list = [0, 1]
elif class_name == "ナイト":
    class_name_list = [0, 0]

if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]

input_data = [count_1, routing_flg, period]
input_data.extend(campaign_name_list)  # listをくっつける
input_data.extend(class_name_list)
input_data.extend(gender_list)

print(model.predict([input_data]))  # 1
print(model.predict_proba([input_data]))  # 確率で表すこともできる，1の確率が98%
"""予測モデルを立てることで，迅速かつ自動的，そしてデータドリブンな判断ができる"""
