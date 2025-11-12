import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pysr
import pickle
import re
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)

basis = []
for i in range(0, 12):
    basis.append(lambda x, th, p=i: th**p)
    basis.append(lambda x, th, p=i: th**p * x)
    basis.append(lambda x, th, p=i: th**p * x**2)


def fit_model(n=7):
    x, th, y = df["x_fp"], df["th_fp"], df["th_tg"]
    A = construct_matrix(x, th, n)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    tol = 1e-10 * S[0]
    S_inv = np.where(S > tol, 1 / S, 0)
    return (Vt.T * S_inv) @ (U.T @ y)


def model(x_sim, th_sim, c):
    def construct_matrix(x, th, n):
        x = np.ravel(x)
        th = np.ravel(th)
        A = np.column_stack([basis[i](x, th) for i in range(n)])
        return A

    A_sim = construct_matrix(x_sim, th_sim, 7)
    y_flat = A_sim @ c

    return y_flat.reshape(x_sim.shape)


def draw_weights(c):
    labels = [rf"$a_{{{i}}}$" for i in range(len(c))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, c, color="steelblue", edgecolor="black")

    ax.set_ylabel(r"$c_i$")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig("images/weights.pdf")
    plt.show()


def basic_graphs(c=None):
    # scatter points
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel(r"$x_{fp}$")
    ax.set_ylabel(r"$\theta_{fp}$")
    ax.set_zlabel(r"$\theta_{tg}$")

    x = df["x_fp"].to_numpy()
    th = df["th_fp"].to_numpy()
    y = df["th_tg"].to_numpy()

    ax.scatter(x, th, y, s=20, c=y, cmap="viridis", label="data")

    # create grid for surface
    if c is None:
        plt.tight_layout()
        plt.show()
        return
    xg = np.linspace(x.min(), x.max(), 50)
    thg = np.linspace(th.min(), th.max(), 50)
    X, TH = np.meshgrid(xg, thg)
    Y = model(X, TH, c)

    # draw surface
    ax.plot_surface(X, TH, Y, cmap="plasma", alpha=0.5, linewidth=0)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.show()


def draw_res(res):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(res, bins=20, edgecolor="black", alpha=0.7)

    ax.set_ylabel("N")
    ax.set_xlabel(r"$\theta_{tg} - \theta_{tg model}$")
    plt.tight_layout()
    fig.savefig("images/distribution.pdf")
    plt.show()


def construct_matrix(x, th, n):
    A = np.column_stack([basis[i](x, th) for i in range(n)])
    return A


def learn():
    x = df["x_fp"]
    th = df["th_fp"]
    y = df["th_tg"]
    err = 0
    chis = []
    for i in range(3, len(basis) - 3 * 4, 3):
        A = construct_matrix(x, th, i)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        S_inv = 1 / S
        c = (Vt.T @ np.diag(S_inv) @ U.T) @ y

        K = i // 3
        err = (
            K * (th ** (K - 1)) * 1e-3
            + +np.sqrt(th ** (2 * K) * 1 + (K * th ** (K - 1) * x) ** 2 * 1e-6)
            + +np.sqrt(
                (2 * x * th**K) ** 2 * 1 + (K * th ** (K - 1) * x**2) ** 2 * 1e-6
            )
        )

        y_fit = A @ c
        sigma = err
        res = y - y_fit
        chi2 = np.sum(((y - y_fit) / sigma) ** 2)
        ndf = len(y) - len(c)
        chi2_reduced = chi2 / ndf
        chis.append(chi2_reduced)
        print(i, chi2_reduced)

        if i == 6:
            draw_res(res)
            draw_weights(c)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(chis) + 1), chis, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\chi^2_{red}$")
    ax.grid()
    fig.savefig("images/chiton.pdf")
    plt.show()


def symbolic_regression(load=False):
    X = df[["x_fp", "th_fp"]].to_numpy()
    y = df["th_tg"].to_numpy()
    if load:
        with open("pysr_model.pkl", "rb") as f:
            model = pickle.load(f)
        print(model)
        for i in range(20):
            eq = model.equations_.iloc[i].equation
            print(eq)
            y_fit = model.predict(X)

            x0, x1 = sp.symbols("x0 x1")
            f = eq
            df_dx0 = sp.diff(f, x0)
            df_dx1 = sp.diff(f, x1)

            f_num = sp.lambdify((x0, x1), f, "numpy")
            g0_num = sp.lambdify((x0, x1), df_dx0, "numpy")
            g1_num = sp.lambdify((x0, x1), df_dx1, "numpy")

            def sigma_y(x0_val, x1_val, sx0, sx1, cov01=0.0):
                g0 = g0_num(x0_val, x1_val)
                g1 = g1_num(x0_val, x1_val)
                return (g0**2 * sx0**2 + g1**2 * sx1**2 + 2 * g0 * g1 * cov01) ** 0.5

            sigma = sigma_y(df["x_fp"], df["th_fp"], 1, 1e-3)
            chi2 = np.sum(((y - y_fit) / sigma) ** 2)
            ndf = len(y) - 2
            chi2_reduced = chi2 / ndf
            print(i, "chi2_red =", chi2_reduced)
        return
    X = df[["x_fp", "th_fp"]].to_numpy()
    y = df["th_tg"].to_numpy()

    model = pysr.PySRRegressor(
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp"],
        populations=10,
        progress=True,
        batching=True,
        batch_size=1024
    )

    model.fit(X, y)
    print(model)
    with open("pysr_model.pkl", "wb") as f:
        pickle.dump(model, f)


df = pd.read_csv(
    "/data/modelska/106/thtg-xfp-thfp.dat",
    sep=r"\s+",
    names=["th_tg", "x_fp", "th_fp"],
    engine="python",
)
df["th_fp"] = df["th_fp"] * np.pi / 180
df["th_tg"] = df["th_tg"] * np.pi / 180

# c = fit_model()
# basic_graphs(c)
# learn()
symbolic_regression(load=True)
