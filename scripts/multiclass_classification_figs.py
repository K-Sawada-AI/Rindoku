import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFilter

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13

OUT = "images/multiclass_classification"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------
# 文字と線の重なりを検出する（week2 と同じ仕組み）
# ---------------------------------------------------------------
def audit(fig, name):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, FancyArrowPatch
    from matplotlib.collections import LineCollection
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_text() and getattr(t, "_audit", True)]
        segs = []
        for a in ax.get_children():
            if isinstance(a, Line2D) and a.get_visible() and a not in ax.get_xgridlines() + ax.get_ygridlines():
                segs.append(a.get_transform().transform(a.get_xydata()))
            elif isinstance(a, Polygon):
                segs.append(a.get_transform().transform(a.get_path().interpolated(200).vertices))
            elif isinstance(a, FancyArrowPatch):
                segs.append(a.get_transform().transform(a.get_path().vertices))
            elif isinstance(a, LineCollection):
                for s in a.get_segments():
                    segs.append(a.get_transform().transform(s))
        for t in texts:
            bb = matplotlib.text.Text.get_window_extent(t, r).expanded(1.08, 1.15)  # 注釈の矢印は除く
            hit = False
            for p in segs:
                for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
                    n = int(max(abs(x1 - x0), abs(y1 - y0)) / 2) + 2
                    xs = np.linspace(x0, x1, n)
                    ys = np.linspace(y0, y1, n)
                    if np.any((xs > bb.x0) & (xs < bb.x1) & (ys > bb.y0) & (ys < bb.y1)):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                print(f"[重なり] {name}: '{t.get_text()}'")


def save(fig, name):
    fig.tight_layout()
    audit(fig, name)
    fig.savefig(f"{OUT}/{name}", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------------------------------------------------------------
# 説明用の手書き風「3」（MNIST の実データではない）
# ---------------------------------------------------------------
def make_digit3():
    S = 280
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    w = 26
    d.arc([70, 40, 200, 150], start=200, end=90, fill=255, width=w)
    d.arc([60, 120, 210, 250], start=270, end=160, fill=255, width=w)
    img = img.rotate(-6, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(6))
    small = img.resize((28, 28), Image.LANCZOS)
    a = np.asarray(small, dtype=float)
    a = a / a.max()
    a[a < 0.05] = 0
    return a


digit = make_digit3()

# 1. 画像は行列
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(digit, cmap="gray")
ax.set_xticks(range(0, 28, 4))
ax.set_yticks(range(0, 28, 4))
ax.set_xlabel("横方向のインデックス $a$")
ax.set_ylabel("縦方向のインデックス $b$")
b0, a0 = 6, 13
ax.add_patch(plt.Rectangle((a0 - 0.5, b0 - 0.5), 1, 1, fill=False, ec="tab:red", lw=2))
fig.colorbar(im, ax=ax, label="輝度（0: 暗い, 1: 明るい）")
save(fig, "digit_matrix.png")
PIX_VALUE = digit[b0, a0]

# 2. 平坦化
x_flat = np.r_[digit.ravel(), 1.0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2.4]})
ax1.imshow(digit, cmap="gray")
for r in [2, 9, 16, 23]:
    ax1.annotate("", xy=(26.5, r), xytext=(0.5, r),
                 arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.5))
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("上の行から左→右に読む")
ax2.plot(range(len(x_flat)), x_flat, color="tab:blue", lw=1)
ax2.scatter([784], [1.0], color="tab:red", zorder=3, s=30)
ax2.annotate("末尾に常に 1 を追加", xy=(784, 1.0), xytext=(470, 1.14),
             color="tab:red", arrowprops=dict(arrowstyle="->", color="tab:red"))
ax2.set_xlim(-10, 800)
ax2.set_ylim(-0.05, 1.3)
ax2.set_xlabel("ベクトルの要素番号")
ax2.set_ylabel("輝度")
ax2.grid(alpha=0.4)
save(fig, "flatten.png")

# 3. 線形多クラス分類の流れ（スコアは説明用の値）
scores = np.array([-1.2, -3.0, 1.1, 4.2, -2.1, 1.8, -0.6, 0.3, 2.0, 0.9])
fig = plt.figure(figsize=(12, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.2, 1])
ax0 = fig.add_subplot(gs[0])
ax0.imshow(digit, cmap="gray")
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title("入力画像 → $\\boldsymbol{x}$")
ax1 = fig.add_subplot(gs[1])
cols = ["tab:red" if k == 3 else "tab:gray" for k in range(10)]
ax1.barh(range(10), scores, color=cols)
ax1.set_yticks(range(10), [f"$\\boldsymbol{{w}}_{k}^\\top\\boldsymbol{{x}}$" for k in range(10)])
ax1.invert_yaxis()
ax1.axvline(0, color="black", lw=0.8)
ax1.set_xlabel("内積（スコア）")
ax1.set_xlim(-3.5, 5.5)
ax1.set_title("全クラスの重みベクトルとの内積")
ax1.grid(axis="x", alpha=0.4)
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.text(0.5, 0.62, "argmax", ha="center", fontsize=18)
ax2.text(0.5, 0.40, "$\\hat{y} = 3$", ha="center", fontsize=26, color="tab:red")
save(fig, "linear_multi_flow.png")

# ---------------------------------------------------------------
# 2次元の3クラス人工データと、SGD による学習（確認問題の解答例と同じ実装）
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.2]])
Xt = np.vstack([rng.normal(c, 0.9, size=(60, 2)) for c in centers])
Yt = np.repeat(np.arange(3), 60)
Xb = np.c_[Xt, np.ones(len(Xt))]


def softmax(a):
    ea = np.exp(a - np.max(a))
    return ea / ea.sum()


def train_sgd(X, Y, K, eta=0.1, epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((K, X.shape[1]))
    for t in range(epochs):
        for i in rng.permutation(len(X)):
            p = softmax(W @ X[i])
            y = np.eye(K)[Y[i]]
            W += eta * np.outer(y - p, X[i])
    return W


W = train_sgd(Xb, Yt, 3)
acc = np.mean(np.argmax(Xb @ W.T, axis=1) == Yt)
print("toy accuracy", acc)

colors = ["tab:blue", "tab:orange", "tab:green"]
gx, gy = np.meshgrid(np.linspace(-5, 5, 1000), np.linspace(-4, 5, 1000))
G = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
pred = np.argmax(G @ W.T, axis=1).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=["#cfe0f1", "#fde0c5", "#cdeccd"])
ax.contour(gx, gy, pred, levels=[0.5, 1.5], colors="black", linewidths=1)
for k in range(3):
    ax.scatter(*Xt[Yt == k].T, color=colors[k], s=18, label=f"クラス $\\mathcal{{C}}_{k + 1}$")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 5)
save(fig, "decision_regions.png")

# 4. ソフトマックス関数の入出力
a = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.bar(range(1, 6), a, color="tab:gray")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xlabel("$k$")
ax1.set_ylabel("$a_k$")
ax1.set_title("入力（スコア）")
ax1.set_ylim(-0.6, 0.7)
ax2.bar(range(1, 6), softmax(a), color="tab:blue")
for k, v in enumerate(softmax(a), start=1):
    ax2.text(k, v + 0.008, f"{v:.3f}", ha="center", fontsize=11)
ax2.set_xlabel("$k$")
ax2.set_ylabel("$\\sigma(\\boldsymbol{a})_k$")
ax2.set_title("出力（確率、合計 1）")
ax2.set_ylim(0, 0.35)
for ax in (ax1, ax2):
    ax.grid(axis="y", alpha=0.4)
save(fig, "softmax_bar.png")

# 5. K = 2 ではシグモイド関数になる
fig, ax = plt.subplots(figsize=(6.6, 4.2))
z = np.linspace(-6, 6, 200)
ax.plot(z, 1 / (1 + np.exp(-z)), color="black", lw=6, alpha=0.25, label="シグモイド関数 $1/(1+e^{-z})$")
for a2, c in zip([-2, 0, 2], ["tab:blue", "tab:orange", "tab:green"]):
    a1 = z + a2
    s = np.exp(a1) / (np.exp(a1) + np.exp(a2))
    ax.plot(z, s, color=c, lw=1.5, ls="--", label=f"$\\sigma(\\boldsymbol{{a}})_1$（$a_2 = {a2}$）")
ax.set_xlabel("$z = a_1 - a_2$")
ax.set_ylabel("確率")
ax.set_xlim(-6, 6)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.4)
ax.legend(loc="upper left", fontsize=11)
save(fig, "softmax_sigmoid.png")

# 6. SGD の1回の更新でスコアと確率がどう動くか
rs = np.random.default_rng(3)
W0 = rs.normal(0, 0.4, size=(3, 3))
cands = [x for x in Xb[Yt == 2] if np.argmax(W0 @ x) != 2]
x1 = cands[0]                # 正解は C_3 だが、更新前は別のクラスに分類される事例
eta = 0.2
p0 = softmax(W0 @ x1)
y1 = np.eye(3)[2]
W1 = W0 + eta * np.outer(y1 - p0, x1)
a0_, a1_ = W0 @ x1, W1 @ x1
p1 = softmax(a1_)
print("update a", a1_, "p", p1, "xx", x1 @ x1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
idx = np.arange(3)
wbar = 0.38
labels = ["$\\mathcal{C}_1$", "$\\mathcal{C}_2$", "$\\mathcal{C}_3$（正解）"]
ax1.bar(idx - wbar / 2, a0_, wbar, color="tab:gray", label="更新前")
ax1.bar(idx + wbar / 2, a1_, wbar, color="tab:red", label="更新後")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(idx, labels)
ax1.set_ylabel("スコア $a_k$")
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
ax2.bar(idx - wbar / 2, p0, wbar, color="tab:gray", label="更新前")
ax2.bar(idx + wbar / 2, p1, wbar, color="tab:red", label="更新後")
for k in range(3):
    ax2.text(k - wbar / 2, p0[k] + 0.02, f"{p0[k]:.2f}", ha="center", fontsize=11)
    ax2.text(k + wbar / 2, p1[k] + 0.02, f"{p1[k]:.2f}", ha="center", fontsize=11)
ax2.set_xticks(idx, labels)
ax2.set_ylabel("確率 $p_k$")
ax2.set_ylim(0, 0.7)
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
for ax in (ax1, ax2):
    ax.grid(axis="y", alpha=0.4)
save(fig, "sgd_update.png")

# 7. 混同行列（テキストの scikit-learn の実行結果）
C = np.array([
    [962, 0, 1, 2, 0, 3, 4, 5, 1, 2],
    [0, 1112, 2, 2, 1, 1, 4, 1, 12, 0],
    [10, 12, 896, 25, 12, 5, 12, 17, 36, 7],
    [5, 1, 12, 928, 3, 18, 3, 14, 15, 11],
    [1, 2, 3, 3, 916, 1, 3, 5, 5, 43],
    [9, 2, 0, 38, 11, 769, 15, 10, 26, 12],
    [11, 3, 6, 2, 10, 26, 895, 2, 3, 0],
    [3, 8, 16, 6, 9, 2, 0, 955, 2, 27],
    [14, 14, 8, 29, 18, 41, 10, 18, 805, 17],
    [6, 8, 2, 15, 40, 8, 0, 34, 4, 892]])
fig, ax = plt.subplots(figsize=(8.4, 7.2))
M = np.log1p(C)
ax.imshow(M, cmap="Blues")
for i in range(10):
    for j in range(10):
        v = C[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=10.5,
                color="white" if M[i, j] > 5 else "black",
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("予測したクラス")
ax.set_ylabel("正解のクラス")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
for j, v in enumerate(C.sum(axis=0)):
    ax.text(j, 10.0, str(v), ha="center", va="center", fontsize=10.5, color="tab:orange", clip_on=False)
ax.text(-0.75, 10.0, "列の合計", ha="right", va="center", fontsize=11, color="tab:orange", clip_on=False)
for i, v in enumerate(C.sum(axis=1)):
    ax.text(10.0, i, str(v), ha="center", va="center", fontsize=10.5, color="tab:green", clip_on=False)
ax.text(10.0, -0.85, "行の合計", ha="center", va="bottom", fontsize=11, color="tab:green", clip_on=False)
save(fig, "confusion_matrix.png")

print("pixel", round(float(PIX_VALUE), 3))
print(sorted(os.listdir(OUT)))
