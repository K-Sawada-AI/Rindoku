import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13

OUT = "images/multiple_regression"
os.makedirs(OUT, exist_ok=True)

# 前回（week1）と同じ4点
x = np.array([1, 2, 3, 4], float)
y = np.array([2, 5, 4, 7], float)


def fit(x, y, d):
    X = np.vander(x, d + 1, increasing=True)
    w = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ w
    return w, 1 - e.var() / y.var()


def poly(w, t):
    return sum(w[k] * t ** k for k in range(len(w)))


def setup(ax, xlim=(0, 5), ylim=(0, 9)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(alpha=0.4)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# 1. 前回の回帰直線と残差
w, r2 = fit(x, y, 1)
fig, ax = plt.subplots(figsize=(7, 4.6))
t = np.array([0, 5])
ax.plot(t, poly(w, t), "tab:red", label=f"$\\hat{{y}} = {w[1]:.1f}x + {w[0]:.1f}$  ($R^2 = {r2:.3f}$)")
for xi, yi in zip(x, y):
    ax.vlines(xi, min(yi, poly(w, xi)), max(yi, poly(w, xi)), "tab:blue", linestyles="dashed")
ax.scatter(x, y, zorder=3, color="tab:blue", s=50)
setup(ax)
ax.legend(loc="upper left")
save(fig, "sra_residuals.png")

# 2. 1次・2次・3次の比較
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
t = np.linspace(0, 5, 300)
for ax, d in zip(axes, [1, 2, 3]):
    w, r2 = fit(x, y, d)
    yy = poly(w, t)
    ax.plot(t, yy, "tab:red")
    ax.scatter(x, y, zorder=3, color="tab:blue", s=45)
    setup(ax, ylim=(-1, 10))
    ax.set_title(f"{d}次関数　$R^2 = {r2:.3f}$")
save(fig, "polynomial_fits.png")

# 3. 凸関数の定義（弦）
f = lambda s: 0.25 * (s - 4) ** 2 + 1
u, v, lam = 2, 9, 0.25
lx = lam * u + (1 - lam) * v
ly = lam * f(u) + (1 - lam) * f(v)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
s = np.linspace(0, 10, 300)
ax.plot(s, f(s), "black")
ax.plot([u, v], [f(u), f(v)], "tab:red", lw=2)
ax.scatter([u, v], [f(u), f(v)], color="black", zorder=3)
ax.vlines(lx, 0, ly, color="tab:purple", linestyles="dotted")
ax.scatter([lx], [f(lx)], color="tab:purple", zorder=3)
ax.scatter([lx], [ly], color="tab:red", zorder=3)
ax.annotate("", xy=(lx, ly), xytext=(lx, f(lx)),
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2))
ax.text(5.0, 5.0, "弦", color="tab:red", fontsize=15)
ax.text(5.5, 1.6, "曲線", fontsize=15)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks([u, lx, v], ["$u$", r"$\lambda u + (1-\lambda)v$", "$v$"])
ax.set_yticks([f(lx), ly], [r"$f(\lambda u + (1-\lambda)v)$", r"$\lambda f(u) + (1-\lambda)f(v)$"])
ax.set_aspect("equal")
save(fig, "convex_chord.png")

# 4. 一階の条件（接線）
f = lambda s: 0.15 * (s - 2) ** 2 + 1
g = lambda s: 0.3 * (s - 2)
x0, z = 4, 8
lin = f(x0) + g(x0) * (z - x0)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
s = np.linspace(0, 10, 300)
ax.plot(s, f(s), "black")
ax.plot([0, 10], [f(x0) - g(x0) * x0, f(x0) + g(x0) * (10 - x0)], "tab:red", lw=2)
ax.scatter([x0, z], [f(x0), f(z)], color="black", zorder=3)
ax.scatter([z], [lin], color="tab:red", zorder=3)
ax.annotate("", xy=(z, f(z)), xytext=(z, lin),
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2))
ax.text(6.6, 6.6, "曲線", fontsize=15)
ax.text(6.9, 3.2, "接線", color="tab:red", fontsize=15)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks([x0, z], ["$x$", "$z$"])
ax.set_yticks([f(x0), lin, f(z)], ["$f(x)$", r"$f(x) + \nabla f(x)^\top(z - x)$", "$f(z)$"])
ax.set_aspect("equal")
save(fig, "convex_tangent.png")

# 5. アイスクリームのデータ：1〜4次
IX = np.array([
    9.1, 11.2, 12.3, 18.9, 22.2, 26., 30.9, 31.2, 28.8, 23., 18.3,
    11.1, 8.3, 9.1, 12.5, 18.5, 23.6, 24.8, 30.1, 33.1, 29.8, 23.,
    16.3, 11.2, 9.6, 10.3, 16.4, 19.2, 24.1, 26.5, 31.4, 33.2, 28.8,
    23., 17.4, 12.1, 10.6, 9.8, 14.5, 19.6, 24.7, 26.9, 30.5, 31.2,
    26.9, 23., 17.4, 11., 10.4, 10.4, 15.5, 19.3, 26.4, 26.4, 30.1,
    30.5, 26.4, 22.7, 17.8, 13.4, 10.6, 12.2, 14.9, 20.3, 25.2, 26.3,
    29.7, 31.6, 27.7, 22.6, 15.5, 13.8, 10.8, 12.1, 13.4, 19.9, 25.1,
    26.4, 31.8, 30.4, 26.8, 20.1, 16.6, 11.1, 9.4, 10.1, 16.9, 22.1,
    24.6, 26.6, 32.7, 32.5, 26.6, 23., 17.7, 12.1, 10.3, 11.6, 15.4,
    19., 25.3, 25.8, 27.5, 32.8, 29.4, 23.3, 17.7, 12.6, 11.1, 13.3,
    16., 18.2, 24., 27.5, 27.7, 34.1, 28.1, 21.4, 18.6, 12.3])
IY = np.array([
    463., 360., 380., 584., 763., 886., 1168., 1325., 847.,
    542., 441., 499., 363., 327., 414., 545., 726., 847.,
    1122., 1355., 916., 571., 377., 465., 377., 362., 518.,
    683., 838., 1012., 1267., 1464., 1000., 629., 448., 466.,
    404., 343., 493., 575., 921., 1019., 1149., 1303., 805.,
    739., 587., 561., 486., 470., 564., 609., 899., 946.,
    1295., 1325., 760., 667., 564., 633., 478., 450., 567.,
    611., 947., 962., 1309., 1307., 930., 668., 496., 650.,
    506., 423., 531., 672., 871., 986., 1368., 1319., 924.,
    716., 651., 708., 609., 535., 717., 890., 1054., 1077.,
    1425., 1378., 900., 725., 554., 542., 561., 459., 604.,
    745., 1105., 973., 1263., 1533., 1044., 821., 621., 601.,
    549., 572., 711., 819., 1141., 1350., 1285., 1643., 1133.,
    784., 682., 587.])
fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True)
t = np.linspace(0, 35, 300)
for ax, d in zip(axes.ravel(), [1, 2, 3, 4]):
    w, r2 = fit(IX, IY, d)
    ax.scatter(IX, IY, marker=".", color="tab:blue")
    ax.plot(t, poly(w, t), "tab:red")
    ax.set_title(f"{d}次関数　$R^2 = {r2:.3f}$")
    ax.set_xlim(0, 35)
    ax.set_ylim(-250, 2000)
    ax.grid(alpha=0.4)
for ax in axes[1]:
    ax.set_xlabel("最高気温の月平均（℃）")
for ax in axes[:, 0]:
    ax.set_ylabel("支出額（円）")
save(fig, "icecream_fits.png")

# 6. 直交射影のイメージ
fig, ax = plt.subplots(figsize=(6.6, 5))
plane = np.array([[0.3, 1.0], [6.0, 0.2], [8.6, 2.6], [2.9, 3.4], [0.3, 1.0]])
ax.fill(plane[:, 0], plane[:, 1], color="tab:blue", alpha=0.12)
ax.plot(plane[:, 0], plane[:, 1], color="tab:blue", lw=1)
o = np.array([2.2, 1.5])
yh = np.array([5.8, 2.1])
yv = np.array([5.8, 6.4])
ax.annotate("", xy=yv, xytext=o, arrowprops=dict(arrowstyle="->", lw=2.5, color="black"))
ax.annotate("", xy=yh, xytext=o, arrowprops=dict(arrowstyle="->", lw=2.5, color="tab:blue"))
ax.plot([yh[0], yv[0]], [yh[1], yv[1]], "--", color="tab:red", lw=2.5)
ax.plot([yh[0] - 0.3, yh[0] - 0.3, yh[0]], [yh[1] + 0.05, yh[1] + 0.35, yh[1] + 0.3], color="gray", lw=1)
ax.text(3.4, 4.4, r"$\boldsymbol{y}$", fontsize=18)
ax.text(3.6, 1.3, r"$\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w}$", fontsize=16, color="tab:blue")
ax.text(6.0, 4.2, r"$\boldsymbol{\epsilon} = \boldsymbol{y} - \hat{\boldsymbol{y}}$", fontsize=16, color="tab:red")
ax.text(6.4, 0.6, "$\\boldsymbol{X}$ の列が張る空間", fontsize=13, color="tab:blue")
ax.set_xlim(0, 9)
ax.set_ylim(0, 7)
ax.set_aspect("equal")
ax.axis("off")
save(fig, "projection.png")

print(sorted(os.listdir(OUT)))
