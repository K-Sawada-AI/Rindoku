# 第2週 重回帰

---

## 目次

1. 重回帰とは
1. ベクトル・行列による表記
1. 二乗残差和を最小化する
1. 目的関数の凸性
1. Pythonによる実装
1. 重回帰の性質
1. まとめ

---

## 1. 重回帰とは

--

### 1.1 前回の振り返り

- 前回は4点のデータに直線 $\hat{y} = ax + b$ を当てはめた
- 二乗残差を最小にしても、残差は残る（$R^2 = 0.754$）

<img src="images/multiple_regression/sra_residuals.png" alt="前回の4点のデータと回帰直線、それぞれの残差" class="regression-figure">

--

### 1.2 より表現力の高いモデル

直線の代わりに、$d$ 次の多項式を当てはめることを考える。

$$
\hat{y} = w\_0 + w\_1 x + w\_2 x^2 + \cdots + w\_d x^d
$$

- $w\_0, w\_1, \ldots, w\_d$ がモデルのパラメータ
- $x$ について見ると非線形だが、パラメータ $w\_k$ について見ると**線形**
- $1, x, x^2, \ldots, x^d$ を別々の説明変数とみなせば、重みつき和で予測していることになる

--

### 1.3 線形回帰

$d$ 個の説明変数 $x\_1, x\_2, \ldots, x\_d$ の重みつき和で目的変数を予測することを**線形回帰**と呼ぶ。

$$
\hat{y} = w\_1 x\_1 + w\_2 x\_2 + \cdots + w\_d x\_d
$$

- 説明変数が1つ：単回帰
- 説明変数が2つ以上：**重回帰**

$x\_1 = 1, x\_2 = x, x\_3 = x^2, \ldots$ とおけば、多項式のフィッティングも重回帰になる。

--

### 1.4 多項式による当てはめの例

前回の4点に、1次・2次・3次関数を当てはめた結果

<img src="images/multiple_regression/polynomial_fits.png" alt="4点のデータに1次、2次、3次関数を当てはめた結果" class="regression-figure">

- 3次関数（パラメータ4個）では、4点すべてを通る
- 本章の方法で、重み $(-13, 23.67, -10, 1.33)$ を行列計算で求められる

Note: 2次の係数は0になり、このデータでは2次にしても改善しない。3次でR²=1になるが、これが良いモデルかは第3章で扱う。

--

### 1.5 説明変数の設計

説明変数を何個用意し、どう定義するかは分析者が決めてよい。

- 例：東京のアイスクリームへの支出額を予測する
  - そのまま使う：最高気温、月、アイスクリームの平均価格
  - 変換して使う：最高気温の平方根、最高気温 × 月
- 変換の仕方を人が設計すれば、非線形な関係も線形回帰の枠組みで扱える

---

## 2. ベクトル・行列による表記

--

### 2.1 ベクトルによる表記

- ベクトルは小文字の太字で表し、特に断りがなければ**列ベクトル**とする
- $d$ 個の説明変数と重みを、それぞれベクトルにまとめる

$$
\boldsymbol{x} = \begin{pmatrix} x\_1 \\\\ x\_2 \\\\ \vdots \\\\ x\_d \end{pmatrix},
\quad
\boldsymbol{w} = \begin{pmatrix} w\_1 \\\\ w\_2 \\\\ \vdots \\\\ w\_d \end{pmatrix}
$$

横に書くときは転置を使い、$\boldsymbol{v} = \begin{pmatrix} 1 & 0 & -1 \end{pmatrix}^\top$ のように書く。

--

### 2.2 予測値は内積で書ける

$d$ 次元ベクトル $\boldsymbol{u}, \boldsymbol{v}$ の内積は、転置を使って次のように書ける。

$$
\boldsymbol{u}^\top \boldsymbol{v} = u\_1 v\_1 + u\_2 v\_2 + \cdots + u\_d v\_d
$$

したがって、重回帰の予測値は $\boldsymbol{x}$ と $\boldsymbol{w}$ の内積になる。

$$
\hat{y} = \boldsymbol{x}^\top \boldsymbol{w} = w\_1 x\_1 + w\_2 x\_2 + \cdots + w\_d x\_d
$$

--

### 2.3 事例とデータ

- 単回帰と同じく、説明変数と目的変数の組を**事例**と呼ぶ
- 違いは、説明変数がスカラー $x$ からベクトル $\boldsymbol{x}$ になったこと

$N$ 個の事例からなるデータ

$$
\mathcal{D} = \left\\{ (\boldsymbol{x}\_1, y\_1), (\boldsymbol{x}\_2, y\_2), \ldots, (\boldsymbol{x}\_N, y\_N) \right\\} = \left\\{ (\boldsymbol{x}\_i, y\_i) \right\\}\_{i=1}^{N}
$$

--

### 2.4 計画行列

各事例の $\boldsymbol{x}\_i^\top$ を行として縦に $N$ 個並べた行列 $\boldsymbol{X} \in \mathbb{R}^{N \times d}$ を**計画行列**（design matrix）と呼ぶ。

$$
\boldsymbol{X} = \begin{pmatrix} \boldsymbol{x}\_1^\top \\\\ \boldsymbol{x}\_2^\top \\\\ \vdots \\\\ \boldsymbol{x}\_N^\top \end{pmatrix}
= \begin{pmatrix}
X\_{1,1} & X\_{1,2} & \cdots & X\_{1,d} \\\\
X\_{2,1} & X\_{2,2} & \cdots & X\_{2,d} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
X\_{N,1} & X\_{N,2} & \cdots & X\_{N,d}
\end{pmatrix},
\quad
\boldsymbol{y} = \begin{pmatrix} y\_1 \\\\ y\_2 \\\\ \vdots \\\\ y\_N \end{pmatrix}
$$

- 行が事例、列が説明変数に対応する
- $X\_{i,j}$ は $i$ 番目の事例の $j$ 番目の説明変数

--

### 2.5 行列・ベクトル積

$N \times d$ 行列と $d$ 次元ベクトルの積は、$N$ 次元ベクトルになる。

例：

$$
\begin{pmatrix} 1 & 3 & 5 \\\\ 6 & 4 & 2 \end{pmatrix}
\begin{pmatrix} 2 \\\\ -2 \\\\ 3 \end{pmatrix}
= \begin{pmatrix} 1 \times 2 + 3 \times (-2) + 5 \times 3 \\\\ 6 \times 2 + 4 \times (-2) + 2 \times 3 \end{pmatrix}
= \begin{pmatrix} 11 \\\\ 10 \end{pmatrix}
$$

各行とベクトルの内積を縦に並べたものになっている。

--

### 2.6 予測値ベクトル

$N$ 個の予測値を縦に並べたベクトル $\hat{\boldsymbol{y}}$ は、計画行列とパラメータベクトルの積で書ける。

$$
\hat{\boldsymbol{y}} = \begin{pmatrix} \hat{y}\_1 \\\\ \hat{y}\_2 \\\\ \vdots \\\\ \hat{y}\_N \end{pmatrix}
= \begin{pmatrix} \boldsymbol{x}\_1^\top \boldsymbol{w} \\\\ \boldsymbol{x}\_2^\top \boldsymbol{w} \\\\ \vdots \\\\ \boldsymbol{x}\_N^\top \boldsymbol{w} \end{pmatrix}
= \boldsymbol{X} \boldsymbol{w}
$$

$N$ 個の予測を、1回の行列計算でまとめて求められる。

--

### 2.7 単回帰を表現する

$d = 2$ として、次のように定義する。

$$
\boldsymbol{x} = \begin{pmatrix} 1 \\\\ x \end{pmatrix},
\quad
\boldsymbol{w} = \begin{pmatrix} b \\\\ a \end{pmatrix}
\quad \Rightarrow \quad
\hat{y} = \boldsymbol{x}^\top \boldsymbol{w} = b + xa = ax + b
$$

- 定数 $1$ の説明変数の重みが切片になる
- 単回帰は重回帰の特別な場合なので、この章の結果は単回帰にも当てはまる

--

### 2.7 単回帰を表現する

前回の4点 $(1, 2), (2, 5), (3, 4), (4, 7)$ の場合

$$
\boldsymbol{X} = \begin{pmatrix} 1 & 1 \\\\ 1 & 2 \\\\ 1 & 3 \\\\ 1 & 4 \end{pmatrix},
\quad
\boldsymbol{y} = \begin{pmatrix} 2 \\\\ 5 \\\\ 4 \\\\ 7 \end{pmatrix},
\quad
\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w} = \begin{pmatrix} a + b \\\\ 2a + b \\\\ 3a + b \\\\ 4a + b \end{pmatrix}
$$

---

## 3. 二乗残差和を最小化する

--

### 3.1 残差と目的関数

単回帰と同じく、真の値と予測値の差を残差とする。

$$
\epsilon\_i = y\_i - \hat{y}\_i = y\_i - \boldsymbol{x}\_i^\top \boldsymbol{w}
$$

平均二乗残差は

$$
\hat{L} = \frac{1}{N} \sum\_{i=1}^{N} \left( y\_i - \boldsymbol{x}\_i^\top \boldsymbol{w} \right)^2
$$

- $\frac{1}{N}$ は定数倍なので、最小化する $\boldsymbol{w}$ には影響しない
- 以降は**二乗残差和**を目的関数とする

--

### 3.2 ベクトルで書いた目的関数

残差を並べたベクトルは

$$
\boldsymbol{\epsilon} = \begin{pmatrix} \epsilon\_1 \\\\ \vdots \\\\ \epsilon\_N \end{pmatrix} = \boldsymbol{y} - \hat{\boldsymbol{y}} = \boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}
$$

$L\_2$ ノルム $\lVert \boldsymbol{v} \rVert = \sqrt{\boldsymbol{v}^\top \boldsymbol{v}} = \sqrt{v\_1^2 + \cdots + v\_d^2}$ を使うと、目的関数は

$$
\hat{L}\_{\mathcal{D}}(\boldsymbol{w}) = \lVert \boldsymbol{\epsilon} \rVert^2 = \lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{w} \rVert^2 = \sum\_{i=1}^{N} \left( y\_i - \boldsymbol{x}\_i^\top \boldsymbol{w} \right)^2
$$

--

### 3.3 偏微分する

あるパラメータ $w\_j$ で偏微分する（$j = 1, \ldots, d$）。

$$
\begin{aligned}
\frac{\partial \hat{L}\_{\mathcal{D}}(\boldsymbol{w})}{\partial w\_j}
&= \sum\_{i=1}^{N} 2 \cdot \underbrace{\left( y\_i - \boldsymbol{x}\_i^\top \boldsymbol{w} \right)}\_{= \epsilon\_i} \cdot \left( -X\_{i,j} \right) \\\\
&= -2 \sum\_{i=1}^{N} X\_{i,j} \epsilon\_i \\\\
&= -2 \begin{pmatrix} X\_{1,j} & X\_{2,j} & \cdots & X\_{N,j} \end{pmatrix} \begin{pmatrix} \epsilon\_1 \\\\ \epsilon\_2 \\\\ \vdots \\\\ \epsilon\_N \end{pmatrix} \\\\
&= -2 \left( \boldsymbol{X}^\top \boldsymbol{\epsilon} \right)\_j
\end{aligned}
$$

$\boldsymbol{X}$ の第 $j$ 列は、$\boldsymbol{X}^\top$ の第 $j$ 行にあたることを使った。

--

### 3.4 勾配

各 $w\_j$ での偏微分を縦に並べたものを**勾配**と呼び、$\nabla$ で表す。

$$
\nabla \hat{L}\_{\mathcal{D}}(\boldsymbol{w})
= \begin{pmatrix} \frac{\partial}{\partial w\_1} \\\\ \vdots \\\\ \frac{\partial}{\partial w\_d} \end{pmatrix} \hat{L}\_{\mathcal{D}}(\boldsymbol{w})
= -2 \boldsymbol{X}^\top \boldsymbol{\epsilon}
$$

$\boldsymbol{\epsilon} = \boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}$ を代入すると

$$
\nabla \hat{L}\_{\mathcal{D}}(\boldsymbol{w}) = 2 \boldsymbol{X}^\top \left( \boldsymbol{X}\boldsymbol{w} - \boldsymbol{y} \right)
$$

勾配は、関数の値が最も大きく増える方向を表す。

--

### 3.5 正規方程式

勾配を $\boldsymbol{0}$ とおいて $\boldsymbol{w}$ について解く。

$$
\begin{aligned}
2 \boldsymbol{X}^\top \left( \boldsymbol{X}\boldsymbol{w} - \boldsymbol{y} \right) &= \boldsymbol{0} \\\\
\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} - \boldsymbol{X}^\top \boldsymbol{y} &= \boldsymbol{0} \\\\
\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} &= \boldsymbol{X}^\top \boldsymbol{y}
\end{aligned}
$$

最後の式を**正規方程式**（normal equation）と呼ぶ。

--

### 3.5 正規方程式

正規方程式を解くと、重回帰のパラメータが得られる。

$$
\boldsymbol{w} = \left( \boldsymbol{X}^\top \boldsymbol{X} \right)^{-1} \boldsymbol{X}^\top \boldsymbol{y}
$$

- データ $\boldsymbol{X}, \boldsymbol{y}$ が与えられれば、行列演算だけで解析的に求まる
- $\boldsymbol{X}^\top \boldsymbol{X}$ が逆行列をもたない場合は、ムーア・ペンローズの擬似逆行列で代用できる

Note: 勾配が0というのは最小値の必要条件にすぎない。最小値であることは4章で凸性を使って示す。

--

### 3.6 実例：前回の結果を再現する

前回の4点で計算すると

$$
\boldsymbol{X}^\top \boldsymbol{X} = \begin{pmatrix} 4 & 10 \\\\ 10 & 30 \end{pmatrix},
\quad
\boldsymbol{X}^\top \boldsymbol{y} = \begin{pmatrix} 18 \\\\ 52 \end{pmatrix}
$$

$$
\boldsymbol{w} = \frac{1}{20} \begin{pmatrix} 30 & -10 \\\\ -10 & 4 \end{pmatrix} \begin{pmatrix} 18 \\\\ 52 \end{pmatrix}
= \begin{pmatrix} 1.0 \\\\ 1.4 \end{pmatrix}
$$

$b = 1.0, a = 1.4$ となり、前回求めた $\hat{y} = 1.4x + 1.0$ と一致する。

--

### 3.7 別の導出：展開する

目的関数をベクトルのまま展開する。

$$
\begin{aligned}
\hat{L}\_{\mathcal{D}}(\boldsymbol{w}) &= \left( \boldsymbol{y} - \boldsymbol{X}\boldsymbol{w} \right)^\top \left( \boldsymbol{y} - \boldsymbol{X}\boldsymbol{w} \right) \\\\
&= \left( \boldsymbol{y}^\top - \boldsymbol{w}^\top \boldsymbol{X}^\top \right) \left( \boldsymbol{y} - \boldsymbol{X}\boldsymbol{w} \right) \\\\
&= \boldsymbol{y}^\top \boldsymbol{y} - \boldsymbol{y}^\top \boldsymbol{X}\boldsymbol{w} - \boldsymbol{w}^\top \boldsymbol{X}^\top \boldsymbol{y} + \boldsymbol{w}^\top \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} \\\\
&= \boldsymbol{y}^\top \boldsymbol{y} - 2 \left( \boldsymbol{X}^\top \boldsymbol{y} \right)^\top \boldsymbol{w} + \boldsymbol{w}^\top \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w}
\end{aligned}
$$

- $(\boldsymbol{A}\boldsymbol{B})^\top = \boldsymbol{B}^\top \boldsymbol{A}^\top$
- スカラーなので $\boldsymbol{u}^\top \boldsymbol{v} = \boldsymbol{v}^\top \boldsymbol{u}$ が成り立ち、2つの項をまとめられる

--

### 3.7 別の導出：ベクトルの微分公式

ベクトル $\boldsymbol{a}, \boldsymbol{x}$ と行列 $\boldsymbol{A}$ について

内積の微分

$$
\frac{\partial}{\partial \boldsymbol{x}} \boldsymbol{a}^\top \boldsymbol{x} = \frac{\partial}{\partial \boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{a} = \boldsymbol{a}
$$

二次形式の微分

$$
\frac{\partial}{\partial \boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x} = \left( \boldsymbol{A} + \boldsymbol{A}^\top \right) \boldsymbol{x}
$$

スカラーをベクトルで微分した結果は列ベクトルとする（分母レイアウト）。

--

### 3.7 別の導出：微分する

$\boldsymbol{X}^\top \boldsymbol{X}$ は対称行列なので、$\left( \boldsymbol{X}^\top \boldsymbol{X} \right) + \left( \boldsymbol{X}^\top \boldsymbol{X} \right)^\top = 2 \boldsymbol{X}^\top \boldsymbol{X}$ となる。

$$
\begin{aligned}
\nabla \hat{L}\_{\mathcal{D}}(\boldsymbol{w})
&= \frac{\partial}{\partial \boldsymbol{w}} \left( \boldsymbol{y}^\top \boldsymbol{y} - 2 \left( \boldsymbol{X}^\top \boldsymbol{y} \right)^\top \boldsymbol{w} + \boldsymbol{w}^\top \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} \right) \\\\
&= 0 - 2 \boldsymbol{X}^\top \boldsymbol{y} + 2 \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} \\\\
&= 2 \boldsymbol{X}^\top \left( \boldsymbol{X}\boldsymbol{w} - \boldsymbol{y} \right)
\end{aligned}
$$

3.4 で成分ごとに求めた勾配と一致する。

---

## 4. 目的関数の凸性

--

### 4.1 なぜ凸性が必要か

- 勾配が $\boldsymbol{0}$ になる点は、極小値・極大値・鞍点のどれでもありうる
- 目的関数が**凸関数**なら、勾配が $\boldsymbol{0}$ の点で最小値をとる
- 方針：目的関数の二階微分（ヘッセ行列）が半正定値であることを示す

まず、凸関数の一般的な性質を確認する。

--

### 4.2 凸関数の定義

関数 $f: \mathbb{R}^d \to \mathbb{R}$ が、任意の $\boldsymbol{u}, \boldsymbol{v}$ と $0 \le \lambda \le 1$ について

$$
f\left( \lambda \boldsymbol{u} + (1 - \lambda) \boldsymbol{v} \right) \le \lambda f(\boldsymbol{u}) + (1 - \lambda) f(\boldsymbol{v})
$$

を満たすとき、$f$ を凸関数と呼ぶ。

<img src="images/multiple_regression/convex_chord.png" alt="凸関数の曲線と、2点を結ぶ弦" class="regression-figure" width="50px" height="500px">

曲線上の2点を結ぶ弦は、曲線より上にある。

--

### 4.3 一階微分による条件

$f$ が微分可能で、任意の $\boldsymbol{x}, \boldsymbol{z}$ について

$$
f(\boldsymbol{z}) \ge f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^\top \left( \boldsymbol{z} - \boldsymbol{x} \right)
$$

が成り立つならば、$f$ は凸関数である。

<img src="images/multiple_regression/convex_tangent.png" alt="凸関数の曲線と、点xでの接線" class="regression-figure">

接線は、曲線より下にある。

--

### 4.3 一階微分による条件（証明）

$\boldsymbol{x} = \lambda \boldsymbol{u} + (1 - \lambda) \boldsymbol{v}$ とおき、$\boldsymbol{z} = \boldsymbol{u}, \boldsymbol{v}$ に適用する。

$$
\begin{aligned}
f(\boldsymbol{u}) &\ge f(\boldsymbol{x}) + (1 - \lambda) \nabla f(\boldsymbol{x})^\top \left( \boldsymbol{u} - \boldsymbol{v} \right) \\\\
f(\boldsymbol{v}) &\ge f(\boldsymbol{x}) + \lambda \nabla f(\boldsymbol{x})^\top \left( \boldsymbol{v} - \boldsymbol{u} \right)
\end{aligned}
$$

1本目に $\lambda$、2本目に $1 - \lambda$ を掛けて足すと、勾配の項が打ち消し合う。

$$
\lambda f(\boldsymbol{u}) + (1 - \lambda) f(\boldsymbol{v}) \ge f(\boldsymbol{x}) = f\left( \lambda \boldsymbol{u} + (1 - \lambda) \boldsymbol{v} \right)
$$

これは凸関数の定義そのものである。

--

### 4.4 二階微分による条件

二階偏微分を並べた行列を**ヘッセ行列** $\nabla^2 f$ と呼ぶ。

$$
\nabla^2 f(\boldsymbol{x}) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x\_1^2} & \frac{\partial^2 f}{\partial x\_1 \partial x\_2} & \cdots & \frac{\partial^2 f}{\partial x\_1 \partial x\_d} \\\\
\frac{\partial^2 f}{\partial x\_2 \partial x\_1} & \frac{\partial^2 f}{\partial x\_2^2} & \cdots & \frac{\partial^2 f}{\partial x\_2 \partial x\_d} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial^2 f}{\partial x\_d \partial x\_1} & \frac{\partial^2 f}{\partial x\_d \partial x\_2} & \cdots & \frac{\partial^2 f}{\partial x\_d^2}
\end{pmatrix}
$$

ヘッセ行列が**半正定値**、すなわち任意の $\boldsymbol{u}$ について

$$
\boldsymbol{u}^\top \nabla^2 f(\boldsymbol{x}) \boldsymbol{u} \ge 0
$$

ならば、$f$ は凸関数である。

--

### 4.4 二階微分による条件（理由）

テイラーの定理より、ある $0 < \lambda < 1$ が存在して

$$
f(\boldsymbol{z}) = f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^\top \left( \boldsymbol{z} - \boldsymbol{x} \right) + \frac{1}{2} \underbrace{\left( \boldsymbol{z} - \boldsymbol{x} \right)^\top \nabla^2 f\left( \boldsymbol{x} + \lambda (\boldsymbol{z} - \boldsymbol{x}) \right) \left( \boldsymbol{z} - \boldsymbol{x} \right)}\_{\ge 0}
$$

- ヘッセ行列が半正定値なので、第3項は非負（$\boldsymbol{u} = \boldsymbol{z} - \boldsymbol{x}$ とおく）
- よって一階微分による条件が成り立ち、$f$ は凸関数
- 1変数なら $f''(x) \ge 0$、つまり傾きが減らないことに対応する

--

### 4.5 重回帰の目的関数は凸関数

一階微分は

$$
\nabla \hat{L}\_{\mathcal{D}}(\boldsymbol{w}) = 2 \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} - 2 \boldsymbol{X}^\top \boldsymbol{y}
$$

さらに微分すると、ヘッセ行列は

$$
\nabla^2 \hat{L}\_{\mathcal{D}}(\boldsymbol{w}) = 2 \boldsymbol{X}^\top \boldsymbol{X}
$$

任意のベクトル $\boldsymbol{u}$ について

$$
\boldsymbol{u}^\top \left( 2 \boldsymbol{X}^\top \boldsymbol{X} \right) \boldsymbol{u} = 2 \left( \boldsymbol{X}\boldsymbol{u} \right)^\top \left( \boldsymbol{X}\boldsymbol{u} \right) = 2 \lVert \boldsymbol{X}\boldsymbol{u} \rVert^2 \ge 0
$$

ヘッセ行列は半正定値なので目的関数は凸関数であり、正規方程式の解で最小値をとる。

---

## 5. Pythonによる実装

--

### 5.1 正規方程式をそのまま実装する

```python
import numpy as np

x = np.array([1, 2, 3, 4])  # 説明変数
y = np.array([2, 5, 4, 7])  # 目的変数

def fit(x, y, d):
    X = np.vander(x, d + 1, increasing=True)  # 列: 1, x, ..., x^d
    w = np.linalg.solve(X.T @ X, X.T @ y)     # 正規方程式を解く
    eps = y - X @ w                           # 残差
    r2 = 1 - eps.var() / y.var()              # 決定係数
    return w, r2

for d in (1, 2, 3):
    w, r2 = fit(x, y, d)
    print(d, np.round(w, 2), round(r2, 3))
```

逆行列を明示的に作るより、`np.linalg.solve` で連立方程式として解くほうが数値的に安定する。

--

### 5.1 正規方程式をそのまま実装する

| 次数 | $\boldsymbol{w}$ | $R^2$ |
| ---- | :------: | :---: |
| 1    | $(1.0, 1.4)$ | $0.754$ |
| 2    | $(1.0, 1.4, 0.0)$ | $0.754$ |
| 3    | $(-13.0, 23.67, -10.0, 1.33)$ | $1.000$ |

- 1次の結果は前回の $\hat{y} = 1.4x + 1.0$ と一致する
- このデータでは2次の係数が $0$ になり、改善しない

--

### 5.2 実データ：アイスクリームの支出額

東京の月別データ（最高気温の月平均と、アイスクリーム・シャーベットへの支出額）に多項式を当てはめる。

`numpy.polynomial` を使う場合

```python
from numpy.polynomial import Polynomial

W = Polynomial.fit(X, Y, 3)  # 3次関数で当てはめ
W.convert().coef             # 0次, 1次, 2次, 3次の係数
# array([ 2.646e+02,  2.929e+01, -1.447e+00,  5.012e-02])

y_hat = W(x)                 # 予測値
```

--

### 5.3 scikit-learn を使う場合

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

reg = make_pipeline(PolynomialFeatures(3), LinearRegression())
reg.fit(X.reshape(-1, 1), Y)   # 説明変数は (事例数, 変数の数) の行列で渡す

reg[1].coef_          # array([ 0.  , 29.29, -1.447, 0.0501])
reg[1].intercept_     # 264.63
reg.score(X.reshape(-1, 1), Y)  # 決定係数 0.878
```

- `PolynomialFeatures` が $1, x, x^2, x^3$ の列を作る
- `LinearRegression` がその重みを求める（1.3 の考え方そのもの）

--

### 5.4 次数と決定係数

<img src="images/multiple_regression/icecream_fits.png" alt="アイスクリームの支出額データに1次から4次の関数を当てはめた結果" class="regression-figure">

- 1次から2次で $R^2$ が大きく改善する
- 3次以降はほとんど変わらない

---

## 6. 重回帰の性質

--

### 6.1 性質の出発点

最適な $\boldsymbol{w}$ では勾配が $\boldsymbol{0}$ なので

$$
\nabla \hat{L}\_{\mathcal{D}}(\boldsymbol{w}) = -2 \boldsymbol{X}^\top \boldsymbol{\epsilon} = \boldsymbol{0}
\quad \Rightarrow \quad
\boldsymbol{X}^\top \boldsymbol{\epsilon} = \boldsymbol{0}
$$

成分で書くと、任意の $j = 1, \ldots, d$ について

$$
\sum\_{i=1}^{N} X\_{i,j} \epsilon\_i = 0
$$

以降の性質は、すべてこの式から導ける。

--

### 6.2 バイアス項の仮定

切片（バイアス項）をもたせるため、$\boldsymbol{X}$ の先頭列はすべて $1$ とする。

$$
\boldsymbol{X} = \begin{pmatrix}
1 & X\_{1,2} & \cdots & X\_{1,d} \\\\
1 & X\_{2,2} & \cdots & X\_{2,d} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
1 & X\_{N,2} & \cdots & X\_{N,d}
\end{pmatrix}
$$

- $w\_1$ がバイアス項に対応する
- $1$ の列は先頭でなくてもよく、どこかにあれば以降の議論は同じ

--

### 6.3 重回帰の性質

前回の単回帰と同じ性質が、重回帰でも成り立つ。

1. 残差の和および平均は 0
1. 目的変数の推定値の平均値は、観測値の平均値と一致する
1. 観測データの重心は回帰平面上にある
1. 各説明変数と残差には相関がない
1. 目的変数の推定値と残差には相関がない

--

#### 6.3.1 残差の和および平均は 0

出発点の式で $j = 1$（すべて $1$ の列）とおくと

$$
\sum\_{i=1}^{N} X\_{i,1} \epsilon\_i = \sum\_{i=1}^{N} \epsilon\_i = 0
$$

単回帰では $b = \overline{y} - a\overline{x}$ から示したが、重回帰ではバイアス項の列から直接出てくる。

--

#### 6.3.2 目的変数の推定値の平均値は、観測値の平均値と一致する

$\boldsymbol{\epsilon} = \boldsymbol{y} - \hat{\boldsymbol{y}}$ の各成分の和をとると

$$
\sum\_{i=1}^{N} \epsilon\_i = \sum\_{i=1}^{N} y\_i - \sum\_{i=1}^{N} \hat{y}\_i
$$

左辺は $0$ なので

$$
\begin{aligned}
\frac{1}{N} \sum\_{i=1}^{N} y\_i &= \frac{1}{N} \sum\_{i=1}^{N} \hat{y}\_i \\\\
\overline{y} &= \overline{\hat{y}}
\end{aligned}
$$

--

#### 6.3.3 観測データの重心は回帰平面上にある

説明変数の平均ベクトル $\overline{\boldsymbol{x}} = \frac{1}{N} \sum\_{i=1}^{N} \boldsymbol{x}\_i$ に対して推定値を求めると

$$
\begin{aligned}
\overline{\boldsymbol{x}}^\top \boldsymbol{w} &= \frac{1}{N} \sum\_{i=1}^{N} \boldsymbol{x}\_i^\top \boldsymbol{w} \\\\
&= \frac{1}{N} \sum\_{i=1}^{N} \hat{y}\_i \\\\
&= \overline{\hat{y}} = \overline{y}
\end{aligned}
$$

単回帰の「回帰直線は重心を通る」の一般化になっている。

--

#### 6.3.4 各説明変数と残差には相関がない

$j$ 番目の説明変数を確率変数 $\mathcal{X}\_j$ とみなす。$\overline{\epsilon} = 0$ を使うと

$$
\begin{aligned}
\mathrm{Cov}[\mathcal{X}\_j, \mathcal{E}] &= \frac{1}{N} \sum\_{i=1}^{N} \left( X\_{i,j} - \overline{X\_{:,j}} \right) \left( \epsilon\_i - \overline{\epsilon} \right) \\\\
&= \frac{1}{N} \sum\_{i=1}^{N} X\_{i,j} \epsilon\_i - \frac{\overline{X\_{:,j}}}{N} \sum\_{i=1}^{N} \epsilon\_i \\\\
&= \frac{1}{N} \underbrace{\sum\_{i=1}^{N} X\_{i,j} \epsilon\_i}\_{0} - \frac{\overline{X\_{:,j}}}{N} \underbrace{\sum\_{i=1}^{N} \epsilon\_i}\_{0} \\\\
&= 0
\end{aligned}
$$

単回帰では $\partial \hat{L} / \partial a = 0$ を使ったが、重回帰では全ての $j$ について一度に言える。

--

#### 6.3.5 目的変数の推定値と残差には相関がない

$$
\begin{aligned}
\mathrm{Cov}[\hat{\mathcal{Y}}, \mathcal{E}] &= \frac{1}{N} \sum\_{i=1}^{N} \left( \hat{y}\_i - \overline{\hat{y}} \right) \left( \epsilon\_i - \overline{\epsilon} \right) \\\\
&= \frac{1}{N} \sum\_{i=1}^{N} \hat{y}\_i \epsilon\_i - \frac{\overline{\hat{y}}}{N} \underbrace{\sum\_{i=1}^{N} \epsilon\_i}\_{0} \\\\
&= \frac{1}{N} \hat{\boldsymbol{y}}^\top \boldsymbol{\epsilon} = \frac{1}{N} \left( \boldsymbol{X}\boldsymbol{w} \right)^\top \boldsymbol{\epsilon} \\\\
&= \frac{1}{N} \boldsymbol{w}^\top \underbrace{\boldsymbol{X}^\top \boldsymbol{\epsilon}}\_{\boldsymbol{0}} = 0
\end{aligned}
$$

--

#### 6.3.6 補足：幾何的な見方

$\boldsymbol{X}^\top \boldsymbol{\epsilon} = \boldsymbol{0}$ は、残差 $\boldsymbol{\epsilon}$ が $\boldsymbol{X}$ のどの列とも直交することを意味する。

<img src="images/multiple_regression/projection.png" alt="ベクトルyを、Xの列が張る空間へ直交射影した図" class="regression-figure">

予測値 $\hat{\boldsymbol{y}}$ は、$\boldsymbol{y}$ を $\boldsymbol{X}$ の列が張る空間へ直交射影したものになっている。

Note: このスライドはテキストにはない補足。

--

### 6.4 決定係数

目的変数の観測値の分散は、推定値の分散と残差の分散の和に分解できる。

$$
\begin{aligned}
\mathrm{Var}[\mathcal{Y}] &= \frac{1}{N} \sum\_{i=1}^{N} \left( y\_i - \overline{y} \right)^2 \\\\
&= \frac{1}{N} \sum\_{i=1}^{N} \left\\{ \left( \hat{y}\_i - \overline{y} \right) + \epsilon\_i \right\\}^2 \\\\
&= \mathrm{Var}[\hat{\mathcal{Y}}] + 2 \cdot \frac{1}{N} \underbrace{\sum\_{i=1}^{N} \hat{y}\_i \epsilon\_i}\_{0} - 2 \overline{y} \cdot \frac{1}{N} \underbrace{\sum\_{i=1}^{N} \epsilon\_i}\_{0} + \mathrm{Var}[\mathcal{E}] \\\\
&= \mathrm{Var}[\hat{\mathcal{Y}}] + \mathrm{Var}[\mathcal{E}]
\end{aligned}
$$

第2項は 6.3.5、第3項は 6.3.1 の結果から $0$ になる。

--

### 6.4 決定係数

- 全変動 $\mathrm{Var}[\mathcal{Y}]$ = 回帰変動 $\mathrm{Var}[\hat{\mathcal{Y}}]$ + 残差変動 $\mathrm{Var}[\mathcal{E}]$
- 全変動のうち、回帰で表現できた割合が決定係数

$$
R^2 = \frac{\mathrm{Var}[\hat{\mathcal{Y}}]}{\mathrm{Var}[\mathcal{Y}]} = 1 - \frac{\mathrm{Var}[\mathcal{E}]}{\mathrm{Var}[\mathcal{Y}]}
$$

単回帰と同じ形で定義でき、$[0, 1]$ の値をとる。

---

## 7. まとめ

--

### 7.1 重回帰のまとめ

- 重回帰は、複数の説明変数の重みつき和 $\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w}$ で予測するモデルである
- 多項式や変数の積も、説明変数を作り直せば線形回帰として扱える
- 正規方程式 $\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} = \boldsymbol{X}^\top \boldsymbol{y}$ を解けば、パラメータが求まる
- 目的関数は凸関数なので、その解で最小値をとる
- $\boldsymbol{X}^\top \boldsymbol{\epsilon} = \boldsymbol{0}$ から、単回帰と同じ残差の性質が導ける

---

## 参考文献

岡崎直観『機械学習帳』
2\. 重回帰
https://chokkan.github.io/mlnote/regression/02mra.html
