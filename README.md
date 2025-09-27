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

型が適切か確認する
```python
import pydantic 
```

```python
from pandera.typing import DataFrame, Series
from pydantic import BaseModel, Field

class Resource(BaseModel):
    label_lab: StrOrNone
    lang: StrOrNone
    role: StrOrNone
```
 
## ②生成AIの出力コードの理解
とりあえず動くが信頼できないコードを、使えるようにします。
エラーがあっても処理がとまらない実装で回避している場合 =>アウトプットを確認しましょう（0行になっていないか、クラスタリングのクラスターが2以上か） 

#### 1. 生成AIに説明してもらう
次のような指示をプロンプトに追加します。
```python
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
print(lgb.__version__)
```
lightGBMの公式ドキュメント
https://lightgbm.readthedocs.io/en/v4.6.0/

### pythonの基礎知識
##### よくある関数の書き方
```python

def function_name(var1: list[int], var2: int = 0)->int:
  var = var2
  return var

```
##### よくあるclassの書き方
```python
class class_name(class_base)->None:
  def __init__(self):
    parameter = 0

  def function(
        self,
        input_1:,
        input_2: bool = False,
      )
  @decoration
  def save(self, filename)->None:
    self.data.to_csv(filename)




```

##### 継承
```python

def function_name(var1: list[int], var2: int = 0)->int:
  var = var2
  return var

```

## 補足:生成がうまくいかない時用

#### 1. 前処理
```python
# カテゴリカル変数をone-hotエンコーディング
data_encoded = pd.get_dummies(data, columns=['product_name', 'product_type', 'weekday', 'weather'],dtype=int)

# 数値型の列を変換
data_encoded['price'] = data_encoded['price'].astype(float)
data_encoded['same_prod_type_stock'] = data_encoded['same_prod_type_stock'].astype(float)
# 消費期限までの日数の特徴量を追加
data_encoded['expiry_term']=(pd.to_datetime(data_encoded['expiry_date'])-pd.to_datetime(data_encoded['date'])).dt.days

# 特徴量と目的変数を分離
X = data_encoded.drop(['sold_today', 'date', 'expiry_date'], axis=1)
y = data_encoded['sold_today']

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 2. モデルの訓練
```
# LightGBMモデルを作成し、学習

model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# モデルの精度を確認
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"訓練データの精度: {train_accuracy:.4f}")
print(f"テストデータの精度: {test_accuracy:.4f}")
```


#### 3.予測対象データの読み込み

```python
# 予測対象データセット
filename = "/content/sample_data_pads/dataset/store_dataset_0501.csv"
dtypes = {
    'sold_today':bool,# 目的変数

    'date':str, #仕入日付
    'prod_id_unique':str, # 商品番号 (棚に並んでいる商品ひとつひとつを区別 ex A_1_20240401, A_2_20240401, ...)
    'product_name':str, # 商品名 {A,B,C,D<E}
    'expiry_date':str, # 消費期限 YYYY/MM/DD
    'product_type':str, # 商品タイプ {チョコ,ピザ,食パン,クロワッサン}
    'price':str, # 価格 100〜250
    'weekday':str, # 曜日 {月, 火, 水, 木, 金, 土, 日}
    'weather':str, # 天候 {晴れ, 雨, 曇り}
    'same_prod_type_stock':str # 同じ商品の開始在庫数
}

dataset_0501 = pd.read_csv(filename, index_col=None, dtype=dtypes, encoding='utf-8')
data_0501 = dataset_0501.set_index('prod_id_unique')
```


#### 4.予測対象データの前処理

```
data_encoded_0501 = pd.get_dummies(data_0501, columns=['product_name', 'product_type', 'weekday', 'weather'], dtype=int)

# 数値型の列を変換
data_encoded_0501['price'] = data_encoded_0501['price'].astype(float)
data_encoded_0501['same_prod_type_stock'] = data_encoded_0501['same_prod_type_stock'].astype(float)
# 消費期限までの日数の特徴量を追加
data_encoded_0501['expiry_term']=(pd.to_datetime(data_encoded_0501['expiry_date'])-pd.to_datetime(data_encoded_0501['date'])).dt.days


X_0501 = data_encoded_0501.drop(['date', 'expiry_date'], axis=1)
X_0501 = pd.DataFrame(X_0501,columns=X_train.columns).fillna(0)

```

#### 5.予測

```
dataset_0501['pred_prob']=model.predict_proba(X_0501)[:,1]
dataset_0501.head(5)
```


#### A.1 説明可能AIの例
ライブラリのインストール
```sh
!pip install shap --quiet
```

```
# ランダムフォレストモデルを作成し、学習

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# モデルの精度を確認
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"訓練データの精度: {train_accuracy:.4f}")
print(f"テストデータの精度: {test_accuracy:.4f}")

# SHAP値を計算
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
```

```
# サンプルを1つ選択してウォーターフォール図を作成
sample_index = 0 # 例として、テストデータの最初のサンプルを使用
shap.plots.waterfall(shap_values[sample_index][:,1])
```

#### A.2 区間予測
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
```

#### 訓練データの前処理
```python
# 訓練データセットのロード

filename = "/content/sample_data_pads/dataset/store_dataset_sale.csv"
dtypes = {
    'date':str, #仕入日付
    'product_name':str, # 商品名 {A,B,C,D<E}
    'sale_count':str, # 販売個数
    'product_type':str, # 商品タイプ {チョコ,ピザ,食パン,クロワッサン}
    'price':str, # 価格 100〜250
    'weekday':str, # 曜日 {月, 火, 水, 木, 金, 土, 日}
    'weather':str, # 天候 {晴れ, 雨, 曇り}
}
dataset_sale = pd.read_csv(filename, index_col=None, dtype=dtypes, encoding='utf-8')

# 前処理

# カテゴリカル変数をone-hotエンコーディング
dataset_sale_encoded = pd.get_dummies(dataset_sale, columns=['product_name', 'product_type', 'weekday', 'weather'],dtype=int)

# 数値型の列を変換
dataset_sale_encoded['price'] = dataset_sale_encoded['price'].astype(float)
dataset_sale_encoded['sale_count'] = dataset_sale_encoded['sale_count'].astype(float)

# 特徴量と目的変数を分離
X = dataset_sale_encoded.drop(['sale_count', 'date'], axis=1)
y = dataset_sale_encoded['sale_count']

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

#### 区間予測モデルの訓練
```python
# 機械学習モデルの訓練

# 勾配ブースティングのハイパーパラメータの設定
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=9,
    min_samples_split=9,
)

# 複数のモデルをつくるため、それをまとめて格納する辞書を定義
all_models = {}

# 90% 予測区間と中央値
for alpha in [0.05, 0.20, 0.50, 0.80, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
# バリデーションデータに予測値をつけます

```


#### バリデーションデータについて区間予測
```python
# テストデータにおける予測の確認
y_test_df = dataset_sale.query("index in @y_test.index").copy()
y_test_df['ans'] = y_test
y_test_df["pred_y_lower"] = all_models["q 0.20"].predict(X_test)
y_test_df["pred_y_upper"] = all_models["q 0.80"].predict(X_test)
```

#### 予測対象データのロード
```python
# 予測対象データのロード

filename = "/content/sample_data_pads/dataset/store_dataset_sale_0501.csv"
dataset_sale_0501 = pd.read_csv(filename, index_col=None, dtype=dtypes, encoding='utf-8')

dataset_sale_encoded_0501 = pd.get_dummies(dataset_sale_0501, columns=['product_name', 'product_type', 'weekday', 'weather'], dtype=int)

# 数値型の列を変換
dataset_sale_encoded_0501['price'] = dataset_sale_encoded_0501['price'].astype(float)
dataset_sale_encoded_0501['sale_count'] = dataset_sale_encoded_0501['sale_count'].astype(float)

# 特徴量と目的変数を分離
X_0501 = dataset_sale_encoded_0501.drop(['sale_count', 'date'], axis=1)
X_0501 = pd.DataFrame(X_0501,columns=X_train.columns).fillna(0)
y_0501 = dataset_sale_encoded_0501['sale_count']

```
#### 区間を予測
```python
# 予測対象データについて区間予測

y_test_df = dataset_sale_0501
y_test_df['ans'] = y_0501
y_test_df["pred_y_lower"] = all_models["q 0.20"].predict(X_0501)
y_test_df["pred_y_upper"] = all_models["q 0.80"].predict(X_0501)

```


#### A.3 分布予測
ライブラリのインストール
```sh
!pip install ngboost --quiet
```

#### 使用するライブラリ
```python
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

```

#### 訓練データの読み込み
```python
# 訓練データセットのロード（区間予測と同じ）

filename = "/content/sample_data_pads/dataset/store_dataset_sale.csv"
dtypes = {
    'date':str, #仕入日付
    'product_name':str, # 商品名 {A,B,C,D<E}
    'sale_count':str, # 販売個数
    'product_type':str, # 商品タイプ {チョコ,ピザ,食パン,クロワッサン}
    'price':str, # 価格 100〜250
    'weekday':str, # 曜日 {月, 火, 水, 木, 金, 土, 日}
    'weather':str, # 天候 {晴れ, 雨, 曇り}
}
dataset_sale = pd.read_csv(filename, index_col=None, dtype=dtypes, encoding='utf-8')


# 前処理（区間予測と同じ）

# カテゴリカル変数をone-hotエンコーディング
dataset_sale_encoded = pd.get_dummies(dataset_sale, columns=['product_name', 'product_type', 'weekday', 'weather'],dtype=int)

# 数値型の列を変換
dataset_sale_encoded['price'] = dataset_sale_encoded['price'].astype(float)
dataset_sale_encoded['sale_count'] = dataset_sale_encoded['sale_count'].astype(float)

# 特徴量と目的変数を分離
X = dataset_sale_encoded.drop(['sale_count', 'date'], axis=1)
y = dataset_sale_encoded['sale_count']

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

#### 訓練データの前処理
```python

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# NGBoostモデルの作成と学習
ngb = NGBRegressor(Dist=Normal, verbose=False)
ngb.fit(X_train, y_train)

# テストデータに対する予測
y_pred_dist = ngb.pred_dist(X_test)

# 予測分布から平均値と標準偏差を取得
y_pred_mean = y_pred_dist.params['loc']
y_pred_std = y_pred_dist.params['scale']

# 平均二乗誤差の計算
mse = mean_squared_error(y_test, y_pred_mean)
print(f"Mean Squared Error: {mse:.4f}")
```


#### バリデーションデータの予測分布の可視化
```python
# 予測分布の可視化（例として2番目のテストデータの予測分布）
plt.figure(figsize=(8, 6))
i=1
plt.hist(y_pred_dist.sample(1000)[i], bins=10, density=True, alpha=0.5, label='Predicted Distribution')
plt.axvline(y_test.iloc[i], color='red', linestyle='dashed', linewidth=2, label='True Value')
plt.xlabel('Sale Count')
plt.ylabel('Probability Density')
plt.title('Predicted Distribution vs. True Value')
plt.legend()
plt.show()
```


#### 予測データの前処理（同様）
```python
# 予測対象データのロード（区間予測と同じ）

filename = "/content/sample_data_pads/dataset/store_dataset_sale_0501.csv"
dataset_sale_0501 = pd.read_csv(filename, index_col=None, dtype=dtypes, encoding='utf-8')

dataset_sale_encoded_0501 = pd.get_dummies(dataset_sale_0501, columns=['product_name', 'product_type', 'weekday', 'weather'], dtype=int)

# 数値型の列を変換
dataset_sale_encoded_0501['price'] = dataset_sale_encoded_0501['price'].astype(float)
dataset_sale_encoded_0501['sale_count'] = dataset_sale_encoded_0501['sale_count'].astype(float)

# 特徴量と目的変数を分離
X_0501 = dataset_sale_encoded_0501.drop(['sale_count', 'date'], axis=1)
X_0501 = pd.DataFrame(X_0501,columns=X_train.columns).fillna(0)
y_0501 = dataset_sale_encoded_0501['sale_count']
```

#### 分布を予測
```python
# 予測対象データについての分布予測
y_pred_dist_0501 = ngb.pred_dist(X_0501)
y_pred_dist_0501.sample(1)[0]
```

#### モンテカルロシミュレーション
```python
initial_stock=30
initial_stock_hist=[]
for itr_day in range(0,30):
  if initial_stock<=10:
    initial_stock=initial_stock+10
  initial_stock=max(initial_stock-y_pred_dist_0501.sample(1)[0][i],0)
  initial_stock_hist.append(initial_stock)

plt.plot(initial_stock_hist)
```
