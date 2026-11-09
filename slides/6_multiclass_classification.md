# 第6週 線形多クラス分類

---

## 目次

1. 多クラス分類とは
1. 手書き文字認識
1. 線形多クラス分類
1. 多クラスロジスティック回帰
1. ソフトマックス関数の性質
1. 最尤推定と確率的勾配降下法
1. 評価
1. Pythonによる実装
1. まとめ

---

## 1. 多クラス分類とは

--

### 1.1 多クラス分類の定義

- 二値分類を拡張し、事例を **3個以上** のクラスに分類するタスク
- 多値分類とも呼ぶ
- 二値分類との対応
  - 二値分類：2クラスのどちらか（例：スパムか否か）
  - 多クラス分類：$K$ クラスのどれか（例：数字の 0〜9）

--

### 1.2 多クラス分類の応用例

- **物体認識**
  - 画像の各ピクセルを「人間」「車」「スノーボード」などに分類する
  - 物体の種類と位置がわかる
- **機械翻訳**
  - 翻訳先言語の全単語をクラスとみなす
  - 翻訳元の文とこれまでの訳語から、次の単語を分類問題として予測していく
- **ユーザ属性の推定**
  - プロフィールや投稿、閲覧履歴から年齢層・性別などを推定する
  - 広告の最適化やパーソナライゼーションに使われる

---

## 2. 手書き文字認識

--

### 2.1 MNIST データセット

- 手書きの数字 70,000 事例（訓練用 60,000、評価用 10,000）
- 各事例は $28 \times 28$ ピクセルのグレースケール画像と、正解の数字の組
- 入力 $\boldsymbol{x}$ から、数字 $\hat{y}$ を 10 クラス $\mathcal{Y} = \\{0, 1, \dots, 9\\}$ の中から選ぶ

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="MNISTの手書き数字の例" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:250px !important; margin:0.25em auto !important; object-fit:contain;">

<small>画像：Josef Steppan, CC BY-SA 4.0, via Wikimedia Commons</small>

--

### 2.2 画像は行列で表される

- $28 \times 28$ の画像は $28 \times 28$ の行列
- 各要素はピクセルの明るさ（輝度）で、$0$（暗い）〜 $1$（明るい）
  - 元データの $0$〜$255$ を $255$ で割っている
- 要素へのアクセスは **縦 → 横** の順：`x[b][a]`

<img src="images/multiclass_classification/digit_matrix.png" alt="手書き風の数字3を28×28の行列として表示した図。赤枠は下に6番目、右に13番目のピクセル" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:240px !important; margin:0.25em auto !important; object-fit:contain;">

<small>※ 説明用に作成した画像（MNIST の実データではない）。赤枠は `x[6][13]` = 0.557</small>

--

### 2.3 画像をベクトルに変換する（平坦化）

- 上の行から左→右に画素値を読み、1列に並べる
- 下に $b$、右に $a$ のピクセルは、ベクトルの $(28b + a)$ 番目の要素になる
  - 例：$(a, b) = (13, 6)$ は $28 \times 6 + 13 = 181$ 番目
- 末尾に常に $1$ となる要素を追加し、$d = 28 \times 28 + 1 = 785$ 次元にする

<img src="images/multiclass_classification/flatten.png" alt="画像を行ごとに読んで1次元のベクトルに並べた図。末尾に1が追加されている" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:215px !important; margin:0.25em auto !important; object-fit:contain;">

Note: テキストでは 28×28 = 754 と書かれている箇所があるが、正しくは 784。末尾の 1 はバイアス項（第2週の計画行列の「1の列」と同じ役割）。

--

### 2.3 画像をベクトルに変換する（平坦化）

複数の画像をまとめて変換するプログラム

```python
def images_to_vectors(X):
    X = np.reshape(X, (len(X), -1))     # 平坦化: (N, 28, 28) -> (N, 784)
    return np.c_[X, np.ones(len(X))]    # 1 を追加: (N, 784) -> (N, 785)

X_train = images_to_vectors(data['train_x'])   # (60000, 785)
X_test = images_to_vectors(data['test_x'])     # (10000, 785)
```

2次元の位置関係を捨てる乱暴な表現だが、これでも驚くほどうまくいく。

---

## 3. 線形多クラス分類

--

### 3.1 入力と出力

- 入力：$d$ 次元の特徴ベクトル $\boldsymbol{x} \in \mathbb{R}^d$
- クラス集合：$K$ 個のクラス $\mathcal{Y} = \\{\mathcal{C}\_1, \mathcal{C}\_2, \dots, \mathcal{C}\_K\\}$
- 出力：最も適切なクラス $\hat{y} \in \mathcal{Y}$

MNIST の場合

- $d = 785$
- $K = 10$、$\mathcal{Y} = \\{0, 1, \dots, 9\\}$

--

### 3.2 線形多クラス分類のラベル推定式

各クラス $y$ に重みベクトル $\boldsymbol{w}\_y \in \mathbb{R}^d$ を用意し、事例との内積が最大のクラスに分類する。

$$
\hat{y} = \mathop{\mathrm{argmax}}\limits\_{y \in \mathcal{Y}} \boldsymbol{w}\_y^\top \boldsymbol{x}
$$

<img src="images/multiclass_classification/linear_multi_flow.png" alt="入力画像から特徴ベクトルを作り、10個の重みベクトルとの内積を計算して最大のクラスを選ぶ流れ" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:235px !important; margin:0.25em auto !important; object-fit:contain;">

<small>※ スコアの値は説明用</small>

--

### 3.3 補足：2次元で見ると

2次元の人工データ（3クラス）で学習した線形多クラス分類器の分類領域

<img src="images/multiclass_classification/decision_regions.png" alt="3クラスの2次元データと、線形多クラス分類器による分類領域。境界は直線" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:260px !important; margin:0.25em auto !important; object-fit:contain;">

- クラス $j$ と $h$ の境界は $\boldsymbol{w}\_j^\top \boldsymbol{x} = \boldsymbol{w}\_h^\top \boldsymbol{x}$、つまり $(\boldsymbol{w}\_j - \boldsymbol{w}\_h)^\top \boldsymbol{x} = 0$
- 境界が直線（一般には超平面）になるので「線形」分類と呼ぶ

Note: このスライドはテキストにはない補足。学習は 6 章の確率的勾配降下法（確認問題の解答例と同じ実装）で行った。

---

## 4. 多クラスロジスティック回帰

--

### 4.1 多クラスロジスティック回帰

- 線形多クラス分類を実現するモデルの一つ
- 事例 $\boldsymbol{x}$ がクラス $\mathcal{C}\_j$ に分類される条件付き確率を、次の式で求める

$$
P(\hat{y} = \mathcal{C}\_j \mid \boldsymbol{x}) = \frac{\exp (\boldsymbol{w}\_j^\top \boldsymbol{x})}{\sum\_{k=1}^{K} \exp (\boldsymbol{w}\_k^\top \boldsymbol{x})}
$$

- 分子：クラス $j$ のスコアを $\exp$ で正の値にしたもの
- 分母：全クラス分の和（合計を $1$ にするための正規化）

--

### 4.2 ソフトマックス関数

クラス $j$ のスコア（内積）を $a\_j = \boldsymbol{w}\_j^\top \boldsymbol{x}$ とし、ベクトルにまとめる。

$$
\boldsymbol{a} = \begin{pmatrix} a\_1 \\\\ a\_2 \\\\ \vdots \\\\ a\_K \end{pmatrix} = \begin{pmatrix} \boldsymbol{w}\_1^\top \boldsymbol{x} \\\\ \boldsymbol{w}\_2^\top \boldsymbol{x} \\\\ \vdots \\\\ \boldsymbol{w}\_K^\top \boldsymbol{x} \end{pmatrix}
$$

すると、条件付き確率は**ソフトマックス関数** $\sigma: \mathbb{R}^K \to \mathbb{R}^K$ で書ける。

$$
P(\hat{y} = \mathcal{C}\_j \mid \boldsymbol{x}) = \sigma(\boldsymbol{a})\_j = \frac{\exp a\_j}{\sum\_{k=1}^{K} \exp a\_k}
$$

$\sigma(\boldsymbol{a})\_j$ は、出力ベクトルの $j$ 番目の要素を表す。

--

### 4.3 ソフトマックス関数の例

```python
def softmax(a):
    ea = np.exp(a)
    return ea / ea.sum()

a = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
softmax(a)
# array([0.198, 0.146, 0.241, 0.120, 0.295])
```

<img src="images/multiclass_classification/softmax_bar.png" alt="ソフトマックス関数の入力スコアと出力確率の棒グラフ" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:245px !important; margin:0.25em auto !important; object-fit:contain;">

- $K$ 個のスコアを確率分布に変換する（二値分類のシグモイド関数に対応）
- スコアの大小関係は保たれる

--

### 4.4 行列による表記

重みベクトル $\boldsymbol{w}\_1^\top, \dots, \boldsymbol{w}\_K^\top$ を縦に並べた行列 $\boldsymbol{W} \in \mathbb{R}^{K \times d}$ を考える。

$$
\boldsymbol{W} = \begin{pmatrix} \boldsymbol{w}\_1^\top \\\\ \boldsymbol{w}\_2^\top \\\\ \vdots \\\\ \boldsymbol{w}\_K^\top \end{pmatrix}
\quad \Rightarrow \quad
\boldsymbol{a} = \boldsymbol{W} \boldsymbol{x}
$$

$p\_j = \sigma(\boldsymbol{a})\_j$ と略記し、確率を並べたベクトルを $\boldsymbol{p} \in \mathbb{R}^K$ とすると

$$
\boldsymbol{p} = \sigma(\boldsymbol{W} \boldsymbol{x})
$$

- $\boldsymbol{W}$ は、事例の空間 $\mathbb{R}^d$ をクラスの空間 $\mathbb{R}^K$ に写す
- MNIST では $\boldsymbol{W} \in \mathbb{R}^{10 \times 785}$

Note: 第2週の重回帰の ŷ = Xw と形が似ている。こちらは「1事例・Kクラス」をまとめた形。

---

## 5. ソフトマックス関数の性質

--

### 5.1 確率分布になる

定義から、次の2つが成り立つ。

$$
\forall k: \sigma(\boldsymbol{a})\_k > 0
$$

$$
\sum\_{k=1}^{K} \sigma(\boldsymbol{a})\_k = 1
$$

$\exp$ は常に正なので各要素は正、分母で割っているので合計は $1$ になる。

--

### 5.2 $K = 2$ ではシグモイド関数になる

$$
\sigma(\boldsymbol{a})\_1 = \frac{e^{a\_1}}{e^{a\_1} + e^{a\_2}} = \frac{1}{1 + \frac{e^{a\_2}}{e^{a\_1}}} = \frac{1}{1 + e^{-(a\_1 - a\_2)}}
$$

$(a\_1 - a\_2)$ を入力としたシグモイド関数と等価である。

<img src="images/multiclass_classification/softmax_sigmoid.png" alt="K=2のソフトマックス関数の出力が、a1-a2を入力とするシグモイド関数に重なる図" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:245px !important; margin:0.25em auto !important; object-fit:contain;">

$a\_2$ の値によらず、$a\_1 - a\_2$ だけで確率が決まる。

--

### 5.3 実装上の注意：オーバーフロー

$\boldsymbol{a} = (1000, 0)$ なら $\sigma(\boldsymbol{a}) \approx (1, 0)$ のはずだが…

```python
a = np.array([1000, 0])
softmax(a)
# RuntimeWarning: overflow encountered in exp
# array([nan,  0.])
```

$e^{1000}$ の計算でオーバーフローが起きている。

--

### 5.3 実装上の注意：定数を引いても結果は同じ

任意の実数 $b$ について

$$
\begin{aligned}
\sigma(\boldsymbol{a})\_j &= \frac{\exp a\_j}{\sum\_{k=1}^{K} \exp a\_k} = \frac{\exp (a\_j - b + b)}{\sum\_{k=1}^{K} \exp (a\_k - b + b)} \\\\
&= \frac{\exp (a\_j - b) \cdot \exp b}{\sum\_{k=1}^{K} \exp (a\_k - b) \cdot \exp b} = \frac{\exp (a\_j - b)}{\sum\_{k=1}^{K} \exp (a\_k - b)}
\end{aligned}
$$

$b$ として $\boldsymbol{a}$ の最大値を引けば、$\exp$ の値は $1$ 以下に収まる。

```python
def softmax(a):
    ea = np.exp(a - np.max(a))
    return ea / ea.sum()

softmax(np.array([1000, 0]))   # array([1., 0.])
```

--

### 5.4 ソフトマックス関数の微分

$\sigma(\boldsymbol{a})\_j$ は $\boldsymbol{a}$ の**全要素**に依存するので、全ての組合せの偏微分が必要になる。

$$
\frac{\partial \sigma(\boldsymbol{a})\_j}{\partial a\_h} \quad (j, h \in \\{1, \dots, K\\})
$$

商の微分法則を使う。

$$
f(x) = \frac{g(x)}{h(x)} \quad \Rightarrow \quad f'(x) = \frac{g'(x) h(x) - g(x) h'(x)}{\\{h(x)\\}^2}
$$

以下、分母を $Z = \sum\_{k=1}^{K} \exp a\_k$ と書く（$\frac{\partial Z}{\partial a\_h} = \exp a\_h$）。

--

### 5.4 ソフトマックス関数の微分（$h = j$ のとき）

$$
\begin{aligned}
\frac{\partial \sigma(\boldsymbol{a})\_j}{\partial a\_j}
&= \frac{\partial}{\partial a\_j} \left( \frac{\exp a\_j}{Z} \right) \\\\
&= \frac{(\exp a\_j) \cdot Z - (\exp a\_j) \cdot (\exp a\_j)}{Z^2} \\\\
&= \frac{\exp a\_j}{Z} - \left( \frac{\exp a\_j}{Z} \right)^2 \\\\
&= \sigma(\boldsymbol{a})\_j - \\{\sigma(\boldsymbol{a})\_j\\}^2 \\\\
&= \sigma(\boldsymbol{a})\_j \\{1 - \sigma(\boldsymbol{a})\_j\\}
\end{aligned}
$$

--

### 5.4 ソフトマックス関数の微分（$h \neq j$ のとき）

分子 $\exp a\_j$ は $a\_h$ を含まないので、その微分は $0$ になる。

$$
\begin{aligned}
\frac{\partial \sigma(\boldsymbol{a})\_j}{\partial a\_h}
&= \frac{\partial}{\partial a\_h} \left( \frac{\exp a\_j}{Z} \right) \\\\
&= \frac{0 \cdot Z - (\exp a\_j) \cdot (\exp a\_h)}{Z^2} \\\\
&= -\frac{\exp a\_j}{Z} \cdot \frac{\exp a\_h}{Z} \\\\
&= -\sigma(\boldsymbol{a})\_j \sigma(\boldsymbol{a})\_h
\end{aligned}
$$

--

### 5.4 ソフトマックス関数の微分（まとめ）

$j = h$ のとき $1$、それ以外は $0$ を返す記号 $\delta\_{jh}$（クロネッカーのデルタ）を使う。

$$
\delta\_{jh} = \begin{cases} 1 & (j = h) \\\\ 0 & (j \neq h) \end{cases}
$$

2つの場合をまとめると

$$
\frac{\partial \sigma(\boldsymbol{a})\_j}{\partial a\_h} = \sigma(\boldsymbol{a})\_j \left( \delta\_{jh} - \sigma(\boldsymbol{a})\_h \right)
$$

シグモイド関数の微分 $\sigma'(a) = \sigma(a) \\{1 - \sigma(a)\\}$ とよく似た形になる。

---

## 6. 最尤推定と確率的勾配降下法

--

### 6.1 正解の表現：one-hot ベクトル

正解がクラス $\mathcal{C}\_j$ であることを、$K$ 次元ベクトル $\boldsymbol{y}$ で表す。

$$
y\_k = \begin{cases} 1 & (k = j) \\\\ 0 & (k \neq j) \end{cases}
$$

- 1つの要素だけが $1$ なので **1-of-K 表現** または **one-hot ベクトル** と呼ぶ
- 例：数字 $0$ を $\mathcal{C}\_1$、$1$ を $\mathcal{C}\_2$、… とすると、数字 $3$ は $\mathcal{C}\_4$

$$
\boldsymbol{y} = \begin{pmatrix} 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \end{pmatrix}^\top
$$

データは $\mathcal{D} = \left\\{ (\boldsymbol{x}\_i, \boldsymbol{y}\_i) \right\\}\_{i=1}^{N}$ と書ける。

--

### 6.2 事例ごとの尤度

正解が $\mathcal{C}\_j$ の事例に対する尤度は、モデルがその事例を $\mathcal{C}\_j$ に分類する確率 $p\_j$ である。

one-hot ベクトルを使うと、$j$ を明示せずに書ける。

$$
\hat{l}\_{\boldsymbol{x}, \boldsymbol{y}}(\boldsymbol{W}) = p\_j = \prod\_{k=1}^{K} \begin{cases} p\_k & (y\_k = 1) \\\\ 1 & (y\_k = 0) \end{cases} = \prod\_{k=1}^{K} p\_k^{y\_k}
$$

- $y\_k = 0$ のクラスは $p\_k^0 = 1$ となり、積に影響しない
- 事例の尤度＝画像が正解の数字に分類される確率

--

### 6.3 学習データ全体の尤度と目的関数

事例が独立同分布（i.i.d.）であると仮定し、全事例の尤度の積をとる。

$$
\hat{L}\_{\mathcal{D}}(\boldsymbol{W}) = \prod\_{i=1}^{N} \hat{l}\_{\boldsymbol{x}\_i, \boldsymbol{y}\_i}(\boldsymbol{W})
$$

二値分類と同様に、**負の対数尤度**を最小化する問題に書き換える。

$$
\hat{\mathcal{L}}\_{\mathcal{D}}^{\mathrm{MLE}}(\boldsymbol{W}) = -\log \hat{L}\_{\mathcal{D}}(\boldsymbol{W}) = -\sum\_{i=1}^{N} \log \hat{l}\_{\boldsymbol{x}\_i, \boldsymbol{y}\_i}(\boldsymbol{W})
$$

--

### 6.4 $L\_2$ 正則化を入れる場合

$$
\hat{\mathcal{L}}\_{\mathcal{D}}^{\mathrm{MAP}}(\boldsymbol{W}) = -\sum\_{i=1}^{N} \log \hat{l}\_{\boldsymbol{x}\_i, \boldsymbol{y}\_i}(\boldsymbol{W}) + \alpha \lVert \boldsymbol{W} \rVert\_F^2 \quad (\alpha > 0)
$$

**フロベニウスノルム** $\lVert \boldsymbol{W} \rVert\_F$：行列の全要素の二乗和の平方根

$$
\boldsymbol{X} = \begin{pmatrix} 2 & 3 \\\\ 4 & 1 \end{pmatrix}
\quad \Rightarrow \quad
\lVert \boldsymbol{X} \rVert\_F = \sqrt{2^2 + 3^2 + 4^2 + 1^2} = \sqrt{30}
$$

- 要素を1列に並べたベクトルの $L\_2$ ノルムと同じ
- 行列の $L\_2$ ノルムには別の定義（スペクトルノルム）があるので、添字 $F$ をつけて区別する

--

### 6.5 確率的勾配降下法

- 目的関数は偏微分できるが、偏微分 $= \boldsymbol{0}$ の解析解は求められない
- 目的関数は事例ごとの損失の和なので、確率的勾配降下法を使う

事例 $(\boldsymbol{x}, \boldsymbol{y})$ をランダムに選びながら、次の更新を繰り返す。

$$
\boldsymbol{W}^{(t+1)} = \boldsymbol{W}^{(t)} + \eta\_t \nabla \log \hat{l}\_{\boldsymbol{x}, \boldsymbol{y}}(\boldsymbol{W}^{(t)})
$$

行列のままだとわかりにくいので、各クラスの重みベクトル $\boldsymbol{w}\_j$ ごとに書く。

$$
\boldsymbol{w}\_j^{(t+1)} = \boldsymbol{w}\_j^{(t)} + \eta\_t \left. \frac{\partial \log \hat{l}\_{\boldsymbol{x}, \boldsymbol{y}}(\boldsymbol{W})}{\partial \boldsymbol{w}\_j} \right|\_{\boldsymbol{W} = \boldsymbol{W}^{(t)}}
$$

--

### 6.6 対数尤度の勾配

事例の対数尤度を整理する。

$$
\log \hat{l}\_{\boldsymbol{x}, \boldsymbol{y}}(\boldsymbol{W}) = \log \prod\_{k=1}^{K} p\_k^{y\_k} = \sum\_{k=1}^{K} y\_k \log p\_k
$$

$\boldsymbol{w}\_j$ で偏微分すると

$$
\frac{\partial \log \hat{l}\_{\boldsymbol{x}, \boldsymbol{y}}(\boldsymbol{W})}{\partial \boldsymbol{w}\_j} = \sum\_{k=1}^{K} \frac{\partial}{\partial \boldsymbol{w}\_j} (y\_k \log p\_k) = \sum\_{k=1}^{K} \frac{y\_k}{p\_k} \frac{\partial p\_k}{\partial \boldsymbol{w}\_j}
$$

$p\_k = \sigma(\boldsymbol{a})\_k$、$a\_j = \boldsymbol{w}\_j^\top \boldsymbol{x}$ という合成関数であることに着目する。

--

### 6.6 対数尤度の勾配

連鎖律と、5.4 のソフトマックス関数の微分を使う。

$$
\begin{aligned}
\frac{\partial \log \hat{l}\_{\boldsymbol{x}, \boldsymbol{y}}(\boldsymbol{W})}{\partial \boldsymbol{w}\_j}
&= \sum\_{k=1}^{K} \frac{y\_k}{p\_k} \frac{\partial p\_k}{\partial a\_j} \frac{\partial a\_j}{\partial \boldsymbol{w}\_j}
= \sum\_{k=1}^{K} \frac{y\_k}{p\_k} \\{p\_k (\delta\_{kj} - p\_j)\\} \boldsymbol{x} \\\\
&= \boldsymbol{x} \sum\_{k=1}^{K} y\_k (\delta\_{kj} - p\_j)
= \boldsymbol{x} \left( \sum\_{k=1}^{K} y\_k \delta\_{kj} - p\_j \sum\_{k=1}^{K} y\_k \right) \\\\
&= \boldsymbol{x} (y\_j - p\_j)
\end{aligned}
$$

- $\sum\_{k} y\_k \delta\_{kj} = y\_j$（$k = j$ の項だけ残る）
- $\sum\_{k} y\_k = 1$（one-hot ベクトル）

--

### 6.7 パラメータの更新式

勾配を更新式に代入すると、とても簡単な形になる。

$$
\boldsymbol{w}\_j^{(t+1)} = \boldsymbol{w}\_j^{(t)} + \eta\_t \left( y\_j - p\_j^{(t)} \right) \boldsymbol{x}
$$

- （正解 $-$ 予測確率）でスケールした特徴ベクトル $\boldsymbol{x}$ を、重みベクトルに足し込む
- 全てのクラス $j = 1, \dots, K$ について同時に更新する
- 予測が正解に近いほど $y\_j - p\_j^{(t)}$ が小さく、更新も小さい

--

### 6.8 更新式の解釈

事例 $\boldsymbol{x}$ の正解が $\mathcal{C}\_j$ で、$h \neq j$ のクラス $\mathcal{C}\_h$ には分類すべきでないとする。

$$
\begin{aligned}
\boldsymbol{w}\_j^{(t+1)} &= \boldsymbol{w}\_j^{(t)} + \eta\_t \left( 1 - p\_j^{(t)} \right) \boldsymbol{x} \quad (y\_j = 1) \\\\
\boldsymbol{w}\_h^{(t+1)} &= \boldsymbol{w}\_h^{(t)} - \eta\_t p\_h^{(t)} \boldsymbol{x} \quad (y\_h = 0)
\end{aligned}
$$

更新後のスコアは（$\boldsymbol{x}^\top \boldsymbol{x} \ge 0$ に注意）

$$
\begin{aligned}
a\_j^{(t+1)} &= \boldsymbol{x}^\top \boldsymbol{w}\_j^{(t+1)} = a\_j^{(t)} + \eta\_t \left( 1 - p\_j^{(t)} \right) \boldsymbol{x}^\top \boldsymbol{x} \geq a\_j^{(t)} \\\\
a\_h^{(t+1)} &= \boldsymbol{x}^\top \boldsymbol{w}\_h^{(t+1)} = a\_h^{(t)} - \eta\_t p\_h^{(t)} \boldsymbol{x}^\top \boldsymbol{x} \leq a\_h^{(t)}
\end{aligned}
$$

正解クラスのスコアは上がり、それ以外のクラスのスコアは下がる。

Note: テキストの a\_h の式では p\_j と書かれているが、正しくは p\_h。

--

### 6.8 更新式の解釈（数値例）

2次元・3クラスの人工データで、正解が $\mathcal{C}\_3$ の事例に1回だけ更新した例（$\eta = 0.2$）

<img src="images/multiclass_classification/sgd_update.png" alt="1回の更新で、正解クラスのスコアと確率が上がり、他のクラスが下がる棒グラフ" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:240px !important; margin:0.25em auto !important; object-fit:contain;">

- 更新前は $\mathcal{C}\_2$ に分類されていた（$p\_2 = 0.43$ が最大）
- 1回の更新で $\mathcal{C}\_3$ のスコアが上がり、$p\_3$ が最大になった

---

## 7. 評価

--

### 7.1 混同行列

予測したクラスと正解のクラスの組合せごとに事例数を数えた表を**混同行列**（confusion matrix）と呼ぶ。

<img src="images/multiclass_classification/confusion_matrix.png" alt="MNISTの評価データに対する混同行列。対角成分が正解した事例数" class="regression-figure week6-figure" style="display:block; width:auto !important; height:auto !important; max-width:82% !important; max-height:385px !important; margin:0.25em auto !important; object-fit:contain;">

<small>テキストの scikit-learn による実行結果（行：正解、列：予測）</small>

Note: テキスト本文の表は行が予測・列が正解で、数値も別の実行結果になっている。ここでは scikit-learn の出力（行が正解）に統一した。

--

### 7.2 正解率

- 対角成分は、予測が正解した事例数
- 正解率は、対角成分の和を全事例数で割ったもの

$$
\text{正解率} = \frac{962 + 1112 + 896 + 928 + 916 + 769 + 895 + 955 + 805 + 892}{10000} = 0.913
$$

- 9 を 4 と取り違える（43件）、4 を 9 と取り違える（40件）など、形の似た数字の誤りが多い

--

### 7.3 クラスごとの適合率・再現率・F1スコア

二値分類と同じ指標を、クラス $\mathcal{C}\_k$ ごとに計算する。

- **適合率**：$\mathcal{C}\_k$ と予測した事例のうち、正解も $\mathcal{C}\_k$ の割合（対角成分 ÷ **列**の合計）
- **再現率**：正解が $\mathcal{C}\_k$ の事例のうち、$\mathcal{C}\_k$ と予測できた割合（対角成分 ÷ **行**の合計）
- **F1スコア**：適合率と再現率の調和平均

| 数字 | 適合率 | 再現率 | F1スコア |
| :--: | :----: | :----: | :------: |
| 0 | $962 / 1021 = 0.942$ | $962 / 980 = 0.982$ | $0.962$ |
| 8 | $805 / 909 = 0.886$ | $805 / 974 = 0.826$ | $0.855$ |
| 9 | $892 / 1011 = 0.882$ | $892 / 1009 = 0.884$ | $0.883$ |

どのクラスに強いか・弱いかがわかる（8 は F1 が最も低い）。

--

### 7.4 マクロ平均とマイクロ平均

モデル同士を比較するには、クラスごとの結果を1つにまとめたい。

- **マクロ平均**：クラスごとの指標を計算してから平均する
  - 全クラスを平等に扱う（事例の少ないクラスの性能も同じ重みで効く）
- **マイクロ平均**：分子・分母の事例数を全クラスで足してから割る
  - 事例数の多いクラスの影響が大きい

| | 適合率 | 再現率 | F1スコア |
| :--: | :----: | :----: | :------: |
| マクロ平均 | $0.912$ | $0.912$ | $0.912$ |
| マイクロ平均 | $0.913$ | $0.913$ | $0.913$ |

全クラスを評価する場合、マイクロ平均は正解率と一致する。

--

### 7.5 評価から外すクラスがある場合

- 物体認識の「その他」のように、評価から外すクラスを含むことが多い
- 例：数字 1〜4 だけを評価し、それ以外を「その他」として除外する

$$
\begin{aligned}
\text{Micro P} &= \frac{1112 + 896 + 928 + 916}{1162 + 946 + 1050 + 1020} = \frac{3852}{4178} = 0.922 \\\\
\text{Micro R} &= \frac{1112 + 896 + 928 + 916}{1135 + 1032 + 1010 + 982} = \frac{3852}{4159} = 0.926
\end{aligned}
$$

| | 適合率 | 再現率 | F1スコア |
| :--: | :----: | :----: | :------: |
| マクロ平均 | $0.921$ | $0.925$ | $0.923$ |
| マイクロ平均 | $0.922$ | $0.926$ | $0.924$ |

この場合、マイクロ平均の適合率と再現率は一致しない。

---

## 8. Pythonによる実装

--

### 8.1 scikit-learn での学習

`SGDClassifier` で損失をロジスティック回帰にする。

```python
import numpy as np
from sklearn.linear_model import SGDClassifier

def image_to_vector(X):
    return np.reshape(X, (len(X), -1))   # (N, 28, 28) -> (N, 784)

data = np.load('mnist.npz')
Xtrain = image_to_vector(data['train_x'])   # バイアス項は不要
Ytrain = data['train_y']                    # one-hot ではなくクラス番号
Xtest = image_to_vector(data['test_x'])
Ytest = data['test_y']

model = SGDClassifier(loss='log_loss')
model.fit(Xtrain, Ytrain)
```

- デフォルトは線形SVMなので、`loss` の指定が必要
- バイアス項はモデル内部で自動的に作られる

Note: テキストでは loss='log' だが、scikit-learn 1.1 で 'log_loss' に改名され、1.3 で 'log' は削除された。

--

### 8.2 予測と評価

```python
model.predict(Xtest[0:1])          # array([7]) ←1事例でも (1, d) の行列で渡す
model.predict_proba(Xtest[0:1])    # 各クラスの確率 P(C_k | x)
# 7 の確率が 0.955、3 の確率が 0.043 …

model.score(Xtest, Ytest)          # 正解率 0.913

from sklearn.metrics import confusion_matrix, f1_score
Ytest_pred = model.predict(Xtest)
confusion_matrix(Ytest, Ytest_pred)            # 行：正解、列：予測
f1_score(Ytest, Ytest_pred, average=None)      # クラスごと
f1_score(Ytest, Ytest_pred, average='macro')   # マクロ平均 0.912
f1_score(Ytest, Ytest_pred, average='micro')   # マイクロ平均 0.913
```

- 重み行列 $\boldsymbol{W}$（$K \times d$）は `model.coef_` で取り出せる
- `model.coef_[0].reshape(28, 28)` で、数字 0 の重みを画像として可視化できる

--

### 8.3 確認問題

1. 確率的勾配降下法で多クラスロジスティック回帰モデルを学習するアルゴリズムを実装せよ（MNIST を使う）
1. 評価データ上で、学習したモデルの正解率を測定せよ

解答例は次のスライド

--

### 8.3 確認問題（解答例）

```python
def softmax(a):
    ea = np.exp(a - np.max(a))
    return ea / ea.sum()

def train_sgd(X, Y, K, eta=0.1, epochs=5, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((K, X.shape[1]))              # (K, d)
    for t in range(epochs):
        for i in rng.permutation(len(X)):     # 事例をランダムな順に
            p = softmax(W @ X[i])              # p = σ(Wx)
            y = np.eye(K)[Y[i]]                # one-hot ベクトル
            W += eta * np.outer(y - p, X[i])   # w_j += η (y_j - p_j) x
    return W

W = train_sgd(X_train, data['train_y'], K=10)  # 2.3 の 785 次元表現
Y_pred = np.argmax(X_test @ W.T, axis=1)       # 内積が最大のクラス
print((Y_pred == data['test_y']).mean())       # 正解率
```

`np.outer(y - p, x)` で、全クラスの更新（6.7）を1行でまとめて計算している。

Note: 3.3 と 6.8 の人工データの図はこの関数で学習した。MNIST での正解率は手元で確認すること（学習率やエポック数で変わる）。

---

## 9. まとめ

--

### 9.1 線形多クラス分類のまとめ

- 線形多クラス分類は、クラスごとの重みベクトルとの内積が最大のクラスを選ぶ
- 多クラスロジスティック回帰は、ソフトマックス関数 $\boldsymbol{p} = \sigma(\boldsymbol{W}\boldsymbol{x})$ で各クラスの確率を求める
- 正解を one-hot ベクトルで表すと、負の対数尤度の勾配は $(y\_j - p\_j)\boldsymbol{x}$ という簡単な形になる
- 確率的勾配降下法で、正解クラスのスコアを上げ、他のクラスのスコアを下げるように学習する
- 多クラスの評価では、混同行列と、マクロ平均・マイクロ平均を使う

---

## 参考文献

岡崎直観『機械学習帳』
6\. 線形多クラス分類
https://chokkan.github.io/mlnote/classification/02multi.html
