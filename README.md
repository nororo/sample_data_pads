# 機械学習を実践しよう（応用編）の教材
  
生成AIを使用したプログラミング学習の強みは、みなさんのやりたいことをベースに解説を作ってくれることです。

## 生成AIを利用したPythonによるデータ分析

#### プロンプトで指示
分析目的、処理の概要を伝える必要がある
データの内容や読み込み方法など生成AIに情報を渡す必要がある。
=> …すると、いきなりドラフトが出てくる（ベンダーや部下から納品された状態）のがスタート地点

#### 生成AIの出力コードを理解
エラーがない状態まで試行錯誤を自動化
テストをクリアしている状態
でも、ソフトウェアやメジャーなOSSライブラリと違って信頼できない

（実行結果や実行ログはデータそのものを出力する場合があるため、生成AIに入力して大丈夫なデータで行うこと）

#### 生成AIの出力コードを意思決定に活かす
カスタマイズして分析する
追加のリライト指示で修正=>②へ
分析結果を解釈する


### 1. プロンプトで指示
- 意図、目的
- 使うデータに関する情報

（実行結果や実行ログを生成AIに入力し、再度生成、改善のループ）
- 既存のコード
- エラー内容
- 実行結果、実行ログ(icecream)
（テストコードの実行結果を生成AIに入力し、再度生成、改善のループ）
- テスト実行結果
- 

- プロンプト（生成AIへの入力）は、テンプレートを用意して、編集は最小限にします。=> AIコーディングツールであれば自動で既存のコードや実行結果を取得できます。
- テンプレートは、プロンプトの用途が決まればオプション設定のようなものです。（テンプレートは編集しやすく、変更管理がしやすいようにします。）
  
#### ※Google colaboratolyでは、”AI で生成”機能を有効にすると有効にしたGoogleアカウントではノートブックのデータをモデルの学習に追加されてしまうため、秘密情報や個人情報は書かないでください。
  
### 2 生成AIの出力コードの理解
#### よく使うデータハンドリング

#### まず変数の型
```python
type(sample_var)
```
#### pandasでdataの内容を調べる
```python
data.head(3) # 先頭3行を表示
data.tail(3) # 一番下3行を表示
data.shape # サイズ（n行k列）を表示
data.dtypes # 各列の変数の型
data["column_name"].nunique() # ユニークな数
data["column_name"].value_counts() # 頻度（カテゴリ）
data["column_name"].hist(nbins=100) # 頻度（連続値）
data["column_name"].isna().sum() # 欠損値の個数

### フィルター(外れ値以外をみたい、特定の条件のデータだけをみたい)
data_f = data.query("column_name == 1")
df.query("column_name in ['categ1', 'categ2', ...]")
# 変数で絞りたいとき
val = 1
data_f = data.query("column_name == @val") 

# カラム名が変わる場合
colname_variable="column_1"
df.query("{1} == 'A'".format(colname_variable))


```

#### mergeの確認
indexの重複関係


#### dictを調べる
```python
list(sample_dict.keys())
list(sample_dict.values())
pd.Series(sample_dict)
```

#### インスタンス
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
  
### （削除予定）プロンプトテンプレートについて
  
pythonの変数でプロンプトテキストを作っています。
  
1. role: 役割を記載したテキスト
2. instruction: 指示内容を記載したテキスト（ここを編集）
3. constraints: 注意事項を記載したテキスト（リスト変数。行頭に#をつけることで、その行のテキストを無効化できます。）
4. dataset_explanation: データに関する説明のテキスト（colaboratory以外の生成AIを利用する場合はこれも含める）
  
これらを改行（改行2つ"\n\n"）でつなげることでプロンプトを完成させます。これをprint()関数で出力し、コピー&ペーストで使用します。
  
```python
roleのテキスト
instructionのテキスト
[constraints]リストの要素のテキスト（"\n".join(constraints)はリストの要素を改行でつなげています。）
(dataset_explanation)データ説明のテキスト
```
  
  
  
### 3 プロンプト作成pythonコード
  
#### プロンプト共通のrole変数、constraints変数、dataset_explanation変数
```python
# テンプレート部分
role = "あなたは入門者向けpythonプログラミング学習の補助アシスタントです。"

constraints = [
  "#### 次に注意してください。",
  #"- サンプルデータを作成してください。"
  "- 元のデータの変数(data)を上書きしないでください。",
  "- コードブロックを実行しながら挙動を確認できるように、できるだけ関数化しないでください。",
  "- 入門者がわかるような易しいコメントをつけてください。",
  #"- pythonのバージョンは3.10です。",
  #"- 既に記載されているものは追加で記載する必要はありません。"
  "- 各変数がどのようなものなのかコメントしてください。",
  "- これを教材として、pythonの各データ形式やインスタンスやクラスを解説してください。",
  "- コメントは日本語で書いてください。"
  ]

dataset_explanation = """データの取得部分は次のコードを使用してください
```python
filename = "/content/sample_data_pads/dataset/store_dataset.csv"
dtypes = {
    'prod_id_unique':str, # index(商品番号) (棚に並んでいる商品ひとつひとつを区別 ex A_1_20240401, A_2_20240401, ...)

    'sold_today':bool,# 目的変数

    'date':str, #仕入日付 YYYY-MM-DD
    'product_name':str, # 商品名 {A,B,C,D<E}
    'expiry_date':str, # 消費期限 YYYY-MM-DD （仕入日付との差分に意味があります）
    'product_type':str, # 商品タイプ {チョコ,ピザ,食パン,クロワッサン}
    'price':str, # 価格 100〜250
    'weekday':str, # 曜日 {月, 火, 水, 木, 金, 土, 日}
    'weather':str, # 天候 {晴れ, 雨, 曇り}
    'same_prod_type_stock':str # 同じ商品の開始在庫数
}

data = pd.read_csv(filename, index_col=None, dtype=dtypes, encoding='utf-8')
```""" # ここまでdataset_explanationのテキストです。
```
  
#### 4.1 機械学習を実践しよう(1) ロジスティック回帰モデルによる販売or在庫の予測 
```python

# 指示
instruction = """変数dataのsold_todayカラムを予測するlogistic回帰モデルのpythonコードを提供してください。"""
# プロンプトを構成
prompt = role + "\n\n" + instruction + "\n\n" + "\n".join(constraints)
#prompt = prompt+"\n\n"+dataset_explanation # colaboratory以外の生成AIを利用する場合は行頭の#をはずし、有効にする
print(prompt)
```

#### 4.1 機械学習を実践しよう(2) LightGBM分類モデルによる販売or在庫の予測
  
（変数constraintsとdataset_explanationは共通です。）
```python
instruction = """dataのsold_todayカラムを予測するLightGBM分類モデルのpythonコードを提供してください。"""

prompt = role + "\n\n" + instruction + "\n\n" + "\n".join(constraints)
#prompt = prompt+"\n\n"+dataset_explanation # colaboratory以外の生成AIを利用する場合は行頭の#をはずし、有効にする
print(prompt)
```
  
#### 4.1 機械学習を実践しよう(3) LightGBM分類モデルによる販売or在庫の予測 予測対象データへの予測
```python
instruction = """LightGBM分類モデルで5月1日の商品データの予測を行なってください。
5月1日のデータは"/content/sample_data_pads/dataset/store_dataset_0501.csv"です。
5月1日のデータには'sold_today'カラムがないことに注意してください。
"""
prompt = role + "\n\n" + instruction + "\n\n" + "\n".join(constraints)
#prompt = prompt+"\n\n"+dataset_explanation # colaboratory以外の生成AIを利用する場合は行頭の#をはずし、有効にする
print(prompt)
```
  
  
#### 4.2 機械学習アウトプットの使い方と解釈(1) SHAPによる機械学習モデルの説明
```python
instruction = """LightGBM分類モデルの出力の説明をSHAPで行うpythonコードを提供してください。サンプルを例にSHAP値のウォーターフォール図を作成してください"""

prompt = role + "\n\n" + instruction + "\n\n" + "\n".join(constraints)
#prompt = prompt+"\n\n"+dataset_explanation # colaboratory以外の生成AIを利用する場合は行頭の#をはずし、有効にする
print(prompt)
```
  
  
#### 4.2 機械学習アウトプットの使い方と解釈(2) 分位点回帰による区間予測
```python
instruction = """変数dataのsale_countカラムを分位点回帰により予測するpythonコードを提供してください。"""
prompt = role + "\n\n" + instruction + "\n\n" + "\n".join(constraints)
print(prompt)
```
  
  
#### 4.2 機械学習アウトプットの使い方と解釈(3) ngboostによる分布予測
```python
instruction = """変数dataのsale_countカラムをngboostにより分布予測するpythonコードを提供してください。"""
prompt = role + "\n\n" + instruction + "\n\n" + "\n".join(constraints)+"\n\n"#+cot
print(prompt)
```
