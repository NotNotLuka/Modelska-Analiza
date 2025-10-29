import matplotlib.pyplot as plt
import time
import numpy as np
import scipy
import pandas as pd


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def pretty_graphs(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["cases.confirmed"])
    ax.set_xlabel("$t[d]$")
    ax.set_ylabel("cases.confirmed")
    ax.grid()
    fig.savefig("images/cases.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["deceased.todate"])
    ax.set_xlabel("$t[d]$")
    ax.set_ylabel("deceased.todate")
    ax.grid()
    fig.savefig("images/deceased.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["vaccination.administered.todate"])
    ax.set_xlabel("$t[d]$")
    ax.set_ylabel("vaccination.administered")
    ax.grid()
    fig.savefig("images/vaccination.pdf")
    plt.show()


def classic(x, y, consts):
    # print("x", x)
    t = int(x // 30)
    alpha, beta = consts[2 * t], consts[2 * t + 1]
    D, B, I = y
    return [
        -alpha * D * B,
        alpha * D * B - beta * B,
        beta * B,
    ]


def jac(x, y, consts):
    alpha, beta, gamma, delta = consts
    D, B, I, M = y
    return np.array(
        [
            [-alpha * B - gamma, -alpha * D, 0.0],
            [alpha * B, alpha * D - beta, 0.0],
            [0.0, beta, 0.0, 0.0],
        ],
        dtype=float,
    )


def simulate(t_span, consts):
    t_eval = np.arange(t_span[0], t_span[1])
    print(t_span[1], len(consts))

    fun = lambda x, y: classic(x, y, consts)
    sol = scipy.integrate.solve_ivp(
        fun,
        (t_span[0], t_span[1] - 1),
        [(2e6 - 286) / 2e6, 286 / 2e6, 0],
        method="RK45",
        t_eval=t_eval,
        # jac=lambda x, y: jac(x, y, consts),
    )

    return sol


def get_data():
    df = pd.read_csv("stats.csv")
    df = df[
        [
            "day",
            "cases.confirmed",
            "cases.confirmed.todate",
            "cases.active",
            "deceased.todate",
            "vaccination.administered",
            "vaccination.administered.todate",
        ]
    ]
    df = df[22:1250]
    df = df.fillna(0)
    df.loc[842:, "vaccination.administered.todate"] = df.loc[
        842, "vaccination.administered.todate"
    ]
    return df
    df["week"] = df["day"] // 7
    df_weekly = df.groupby("week").agg(
        {
            "cases.active": "mean",
            "deceased.todate": "last",
            "vaccination.administered": "sum",
        }
    )

    return df_weekly


def residuals(consts, df):
    t = time.time()
    sol = simulate((0, len(df)), consts)
    print("simul", time.time() - t)
    D, B, I = sol.y
    sick_fellas = (B - df["cases.active"]) ** 2
    # sick_fellas = (2e6 * (np.diff(B) - df["cases.confirmed"][:-1])) ** 2
    # dead_fellas = (M - df["deceased.todate"]) ** 2

    print(time.time() - t)

    return np.concatenate([sick_fellas])  # , dead_fellas, vaccinated_fellas])


def fit_to_data():
    df = get_data()[:300]
    df["cases.confirmed"] = df["cases.confirmed"] / (2e6)
    df["cases.active"] = df["cases.active"] / (2e6)

    N = 30
    x0 = np.ones(len(df) // N * 2)
    for i in range(0, len(x0), 2):
        x0[i] = 5e-3
        x0[i + 1] = 3e-3
    res = None
    for i in range(2):
        if i == 1: x0 = res.x
        lb = np.zeros([len(df) // N * 2])
        ub = np.ones([len(df) // N * 2]) * 100
        res = scipy.optimize.least_squares(residuals, x0, args=(df,), bounds=(lb, ub))

    sol = simulate((0, len(df)), res.x)
    print(res.x)
    plt.plot(sol.y[1], label="Rešitev")
    plt.plot(df["cases.active"].to_numpy(), label="Podatki")
    plt.xlabel("t[d]")
    plt.ylabel("$N/N_0$")
    plt.tight_layout()
    plt.legend()
    plt.grid(alpha=0.8)
    # plt.savefig("images/actual.pdf")
    plt.show()

fit_to_data()
