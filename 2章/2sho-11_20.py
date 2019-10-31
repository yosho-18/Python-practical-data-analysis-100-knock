# 2章_小売店のデータでデータ加工を行う10本ノック（汚いデータ）

import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline


# ノック１１：データを読み込んでみよう
uriage_data = pd.read_csv("uriage.csv")
print(uriage_data.head())
kokyaku_data = pd.read_excel("kokyaku_daicho.xlsx")  # pd.read_excelでexcelファイルを読み込む
print(kokyaku_data.head())
"""データの揺れ：データ等で顕在する入力ミスや表記方法の違い等が混在し，不整合を起こしている状態
（欠損値，表記の整合性なし），データの持つ属性や意味を理解することが大事！！"""

# ノック１２：データの揺れを見てみよう
print(uriage_data["item_name"].head())  # 半角全角，大文字小文字が統一されていない
print(uriage_data["item_price"].head())  # データに欠損がある

# ノック１３：データに揺れがあるまま集計しよう
uriage_data["purchase_date"] = pd.to_datetime(uriage_data["purchase_date"])
uriage_data["purchase_month"] = uriage_data["purchase_date"].dt.strftime("%Y%m")
res = uriage_data.pivot_table(index="purchase_month", columns="item_name", aggfunc="size", fill_value=0)  # 件数集計，0埋め
# res = pivot_table(uriage_data, index="purchase_month", columns="item_name", aggfunc="size", fill_value=0) と（多分）同じ
print(res)  # [7 rows x 99 columns]：99行つまり，99個の商品があることになっている，本来は26個
res = uriage_data.pivot_table(index="purchase_month", columns="item_name", values="item_price", aggfunc="sum",
                              fill_value=0)  # valuesをもとにsumを計算
print(res)

# ノック１４：商品名の揺れを補正しよう
"""（今回の場合）スペースの有無，半角全角を補正する"""
print(len(pd.unique(uriage_data.item_name)))  # 99，重複除外
uriage_data["item_name"] = uriage_data["item_name"].str.upper()  # 商品を大文字にする
uriage_data["item_name"] = uriage_data["item_name"].str.replace("　", "")  # 全角を無くす
uriage_data["item_name"] = uriage_data["item_name"].str.replace(" ", "")  # 半角を無くす
print(uriage_data.sort_values(by=["item_name"], ascending=True))  # "item_name"の昇順にソートして表示，一つなら[]無くてもいい

print(pd.unique(uriage_data["item_name"]))  # 商品A～商品Z
print(len(pd.unique(uriage_data["item_name"])))  # 26

"""欠損値補完をしないと，最小値にNaNが出力される
for trg in list(uriage_data["item_name"].sort_values().unique()):
    print(trg + "の最大額" + str(uriage_data.loc[uriage_data["item_name"] == trg]["item_price"].max())
          + "，最小額" + str(uriage_data.loc[uriage_data["item_name"] == trg]["item_price"].min(skipna=False)))"""

# ノック１５：金額欠損値の補完をしよう
print(uriage_data.isnull().any(axis=0))  # item_priceがTrue（item_priceに欠損値が存在する），商品価格変動無しなので，他の同じ商品から補完する

flg_is_null = uriage_data["item_price"].isnull()  # どの行に欠損値が存在するかを保持
# .locで条件を付与し，それに合致するデータを抽出
for trg in list(uriage_data.loc[flg_is_null, "item_name"].unique()):  # 条件：金額が欠損している，２番目item_nameの列を取得
    price = uriage_data.loc[(~flg_is_null) & (uriage_data["item_name"] == trg), "item_price"].max()  # ~：否定演算子，欠損していないデータから値段を引っ張ってくる
    uriage_data["item_price"].loc[(flg_is_null) & (uriage_data["item_name"] == trg)] = price  # 欠損しているところを埋める
print(uriage_data.head())
print(uriage_data.isnull().any(axis=0))  # 確認

for trg in list(uriage_data["item_name"].sort_values().unique()):
    print(trg + "の最大額" + str(uriage_data.loc[uriage_data["item_name"] == trg]["item_price"].max())
          + "，最小額" + str(uriage_data.loc[uriage_data["item_name"] == trg]["item_price"].min(skipna=False)))  # maxとminは同じになる
# skipnaはNaNを無視するか否か，Trueなら最小値はNaNと出力される


# ノック１６：顧客名の揺れを補完しよう
print(kokyaku_data["顧客名"].head())
print(uriage_data["customer_name"].head())
kokyaku_data["顧客名"] = kokyaku_data["顧客名"].str.replace("　", "")
kokyaku_data["顧客名"] = kokyaku_data["顧客名"].str.replace(" ", "")
print(kokyaku_data["顧客名"].head())
"""実際には，名前の誤変換，同姓同名などにも注意"""


# ノック１７：日付の揺れを補正しよう
"""Excelデータ：書式が違うデータが混在する，正しく認識しない場合が出てくる"""
flg_is_serial = kokyaku_data["登録日"].astype("str").str.isdigit()  # 登録日が数値かどうかを判定
print(flg_is_serial.sum())  # 22
fromSerial = pd.to_timedelta(kokyaku_data.loc[flg_is_serial, "登録日"].astype("float"), unit="D")\
             + pd.to_datetime("1900/01/01")  # pd.to_timedeltaで数値から日付に変更
print(fromSerial)
fromString = pd.to_datetime(kokyaku_data.loc[~flg_is_serial, "登録日"])  # 2018/01/04を2018-01-04にする等
print(fromString)

i = pd.concat([fromSerial, fromString])
kokyaku_data["登録日"] = pd.concat([fromSerial, fromString])  # 作り替えたデータを結合（ユニオン）
print(kokyaku_data)  # 元のままくっつくのか？？

kokyaku_data["登録年月"] = kokyaku_data["登録日"].dt.strftime("%Y%m")  # 登録年月の形を作る
rslt = kokyaku_data.groupby("登録年月").count()["顧客名"]  # 月ごとにデータ件数をカウントする
print(rslt, len(kokyaku_data))  # 200

flg_is_serial = kokyaku_data["登録日"].astype("str").str.isdigit()
print(flg_is_serial.sum())  # 0


# ノック１８：顧客名をキーに２つのデータを結合（ジョイン）しよう


# ノック１９：クレンジングしたデータをダンプしよう


# ノック２０：データを集計しよう
