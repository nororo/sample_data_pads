# 機械学習を実践しよう（応用編）の教材
  
生成AIを使用したプログラミング学習の強みは、みなさんのやりたいことをベースに解説を作ってくれることです。

## 生成AIを利用したPythonによるデータ分析

#### ①プロンプトで指示
分析目的、処理の概要を伝える必要がある
データの内容や読み込み方法など生成AIに情報を渡す必要がある。
=> …すると、いきなりドラフトが出てくる（ベンダーや部下から納品された状態）のがスタート地点

#### ②生成AIの出力コードを理解
エラーがない状態まで試行錯誤を自動化、テストをクリアしている状態
しかし、ソフトウェアやメジャーなOSSライブラリと違って信頼できない

#### ③生成AIの出力コードを意思決定に活かす
カスタマイズして分析する
追加のリライト指示で修正=>②へ
分析結果を解釈する


## ①プロンプトで指示
#### 1. 最低限
- 意図、目的
- 使うデータに関する情報

#### 2. 実行結果や実行ログを生成AIに入力し、生成・改善のループを行いたい
- 既存のコード
- エラー内容
- 実行結果、実行ログ(icecream)（実行結果や実行ログはデータそのものを出力する場合があるため、生成AIに入力して大丈夫なデータで行うこと）

#### 3. テストコードの実行結果を生成AIに入力し、生成・改善のループを行いたい
- テスト実行結果

※プロンプト（生成AIへの入力）は、テンプレートを用意して、編集は最小限にします。=> AIコーディングツールであれば自動で既存のコードや実行結果を取得でき、2.や3.を自動で実行してくれます。  
※テンプレートは、プロンプトの用途が決まればオプション設定のように使えます。（テンプレートは編集しやすく、変更管理がしやすいようにします。）  
※Google colaboratolyでは、”AI で生成”機能を有効にすると有効にしたGoogleアカウントではノートブックのデータをモデルの学習に追加されてしまうため、秘密情報や個人情報は書かないでください。  

#### テストの簡単な書き方

簡単に一行で書く場合
```python
assert dataset_0501['pred_prob'].isna().sum() == 0,"予測値が欠損である個数が0ではありません"
```

pytestを使う場合
```python
import pytest

```

型が適切か確認する
```python
import pydantic 
```

 
## ②生成AIの出力コードの理解
とりあえず動くが信頼できないコードを、使えるようにします。
エラーがあっても処理がとまらない実装で回避している場合 =>アウトプットを確認しましょう（0行になっていないか、クラスタリングのクラスターが2以上か） 

#### 1. 生成AIに説明してもらう
次のような指示をプロンプトに追加します。
```
入門者がわかるような易しいコメントをつけてください
各変数がどのようなものなのかコメントしてください
各処理について、入力と出力とどのような操作をしているか次のフォーマットで記載してください。
#### 入力: 入力データ（形式）、出力: 出力データ（形式）、処理内容
```

#### 2. 検証環境を作成して変数の内容を確認
よく使うデータハンドリングを紹介します。

##### 2.1 変数の型を調べる
```python
type(sample_var)
```
##### 2.2 pandasでdataの内容を調べる
```python
data.head(3) # 先頭3行を表示
data.tail(3) # 一番下3行を表示
data.shape # サイズ（n行k列）を表示
data.dtypes # 各列の変数の型
data["column_name"].nunique() # ユニークな数
data["column_name"].duplicates().sum() # 重複している数
data["column_name"].value_counts() # 頻度（カテゴリ）
data["column_name"].hist(nbins=100) # 頻度（連続値）
data["column_name"].isna().sum() # 欠損値の個数
```

##### 2.3 フィルター(外れ値以外をみたい、特定の条件のデータだけをみたい)
```python
data_f = data.query("column_name == 1")
df.query("column_name in ['categ1', 'categ2', ...]")
# 変数で絞りたいとき
val = 1
data_f = data.query("column_name == @val") 
# カラム名が変わる場合
colname_variable="column_1"
data_f = data.query("{1} == 'A'".format(colname_variable))
```

##### 2.4 mergeがうまくいっているかの確認: indexの重複関係を調べる
```python
assert set(df1.index) == set(df2.index),"index集合が異なります"
assert set(df1.index) - set(df2.index)>0,"df2に含まれないindexがdf1にあります"
```

##### 2.5 dictを調べる
```python
list(sample_dict.keys())
list(sample_dict.values())
pd.Series(sample_dict)
```

##### 2.6 インスタンスを調べる
```python
# メソッド全部表示
dir(model)
# 先頭が"_"以外のメソッド表示
[name for name in dir(model) if name[0] != "_"]
# 先頭が"_"以外のメソッドの名前と値
[item for item in inspect.getmembers(model)  if item[0][0] != "_"]
# pandas.DataFrameに変換してもみれます
pd.DataFrame(inspect.getmembers(model),columns=["name","value"]).query("name.str[0] != '_' ")

```
##### 変数の内容を出力
notebookでは最後の出力を自動で表示してくれるが、変数の内容を途中で表示したい場合
print
```
print(var)
```

icecreamライブラリを使う場合（変数名:値の形式で出力するためわかりやすい）
```
from icecream import ic
ic(var)
```


## ③生成AIの出力コードを意思決定に活かす
=>講義スライドへ

## 補足:ライブラリを理解する
公式ドキュメントを参照してAIで理解する
- バージョンに注意
```python
import lightgbm as lgb
lgb.__version__
```
### pythonの基礎知識
##### よくある関数の書き方
```python

def function_name(var1: list[int], var2: int = 0)->int:
  var = var2
  return var

```

##### 継承
```python

def function_name(var1: list[int], var2: int = 0)->int:
  var = var2
  return var

```




