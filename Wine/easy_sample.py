import pandas as pd
# 簡単な例
toy_data = pd.DataFrame([[1,4,7,10,13,16],[2,5,8,11,14,27],[3,6,9,12,15,17],[21,24,27,20,23,26]],
                   index = ['i1','i2','i3', 'i4'],
                   columns = list("abcdef"))
toy_data # 中身の確認
# F列の値が 20 未満の列だけを抜き出す
toy_data[toy_data['f'] < 20]
# F列の値が 20 以上の列だけを抜き出す
toy_data[toy_data['f'] >= 20]
# F列の値が 20 以上の列だけを抜き出して、そのB列を得る
pd.DataFrame(toy_data[toy_data['f'] >= 20]['b'])
# classという名の列を作り、F列の値が 20 未満なら 0 を、そうでなければ 1 を入れる
toy_data['class'] = [0 if i < 20 else 1 for i in toy_data['f'].tolist()]

toy_data # 中身を確認