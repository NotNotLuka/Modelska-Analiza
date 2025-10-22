import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 24,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def draw_solution(x_fun, v_fun, a_fun, filename):
    plt.rcParams.update({"font.size": 22})

    v0_range = np.linspace(-1, 5, 10)
    t = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(
        1, 4, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1, 1, 0.2]}
    )

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=v0_range.min(), vmax=v0_range.max())

    for v0 in v0_range:
        x = x_fun(t, v0)
        v = v_fun(t, v0)
        a = a_fun(t, v0)

        axes[0].plot(t, x, c=cmap(norm(v0)))
        axes[1].plot(t, v, c=cmap(norm(v0)))
        axes[2].plot(t, a, c=cmap(norm(v0)))

    axes[0].set_title(r"$\mathcal{X(T)}$")
    axes[1].set_title(r"$\mathcal{V(T)}$")
    axes[2].set_title(r"$\mathcal{A(T)}$")
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    axes[0].axhline(y=1, color="black", linestyle="--", linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=axes[3])
    cbar.set_label(r"$\mathcal{V}_0$")

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    fig.savefig(f"images/{filename}")


def unknown_final_v():
    x_fun = lambda t, v0: v0 * t + (1 - v0) / 2 * (3 * t**2 - t**3)
    v_fun = lambda t, v0: v0 + 3 / 2 * (1 - v0) * (2 * t - t**2)
    a_fun = lambda t, v0: 3 * (1 - v0) * (1 - t)

    draw_solution(x_fun, v_fun, a_fun, "sol0.pdf")


def other_functionals():
    p = 3
    x_fun = lambda t, v0: v0 * t + (1 - v0) * (4 * p - 1) / (2 * p) * (
        t + (2 * p - 1) / (4 * p - 1) * (np.power(1 - t, (4 * p - 1) / (2 * p - 1)) - 1)
    )
    v_fun = lambda t, v0: v0 + (1 - v0) * (4 * p - 1) / (2 * p) * (
        1 - np.power(1 - t, 2 * p / (2 * p - 1))
    )
    a_fun = (
        lambda t, v0: (1 - v0)
        * (4 * p - 1)
        / (2 * p - 1)
        * np.power(1 - t, 1 / (2 * p - 1))
    )

    draw_solution(x_fun, v_fun, a_fun, "sol1.pdf")


def limit_v(filename):
    plt.rcParams.update({"font.size": 22})

    v_range = np.linspace(-1, 5, 10)
    t = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(
        1, 2, figsize=(6, 6), gridspec_kw={"width_ratios": [1, 0.1]}
    )

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=v_range.min(), vmax=v_range.max())

    k = -10
    for v0 in v_range:
        tanhk = (np.exp(2 * k) - 1) / (np.exp(2 * k) + 1)
        A = k * (v0 - 1) / ((1 + np.exp(2 * k)) * (k - tanhk))
        B = np.exp(2 * k) * A
        D = (k - v0 * tanhk) / (k - tanhk)

        v = A * np.exp(k * t) + B * np.exp(-k * t) + D

        axes[0].plot(t, v, c=cmap(norm(v0)))

    axes[0].set_title(r"$\mathcal{V(T)}$")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=axes[1])
    cbar.set_label(r"$\mathcal{V}_0$")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    fig.savefig(f"images/{filename}")


def get_parameters(N, v0):
    def derivative_eq(n, N):
        line = []
        for _ in range(n):
            line += [0, 0, 0]
        line += [2, 1, 0, 0, -1, 0]
        for _ in range(n + 2, N):
            line += [0, 0, 0]

        return line

    def continuity_eq(n, N):
        line = []
        for _ in range(n):
            line += [0, 0, 0]
        line += [1, 1, 1, 0, 0, -1]
        for _ in range(n + 2, N):
            line += [0, 0, 0]

        return line

    def distance_eq(n, N):
        line = []
        for _ in range(n):
            line += [0, 0, 0]
        line += [2, 3, 6]
        for _ in range(n + 1, N):
            line += [0, 0, 0]

        return line

    matrix = []
    b = [v0, 0]
    matrix.append([0, 0, 1] + [0 for i in range(3, 3 * N)])
    matrix.append([0 for i in range(3, 3 * N)] + [2, 1, 0])

    for n in range(N):
        matrix.append(distance_eq(n, N))
        if n == N - 1:
            b.append(6)
            continue
        matrix.append(derivative_eq(n, N))
        matrix.append(continuity_eq(n, N))
        b += [6, 0, 0]

    matrix = np.array(matrix)
    b = np.array(b)
    solved = np.linalg.solve(matrix, b)

    return solved


def multiple(filename):
    plt.rcParams.update({"font.size": 22})
    N = 5

    v0_range = np.linspace(-1, 5, 10)
    t = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 0.05]}
    )

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=v0_range.min(), vmax=v0_range.max())

    for v0_start in v0_range:
        solved = get_parameters(N, v0_start)
        v_global = []
        t_global = []
        for i in range(N):
            sol = solved[i * 3 : i * 3 + 3]
            v_fun = lambda t: sol[0] * np.power(t, 2) + sol[1] * t + sol[2]
            v = v_fun(t)
            t_global.extend(i + t)
            v_global.extend(v)

        axes[0].plot(t_global, v_global, c=cmap(norm(v0_start)))

    axes[0].set_title(r"$\mathcal{V(T)}$")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for n in range(1, N + 1):
        axes[0].axvline(x=n, color="black", linestyle="--", linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=axes[1])
    cbar.set_label(r"$\mathcal{V}_0$")

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    fig.savefig(f"images/{filename}")


def draw_multiple_naive(filename):
    N = 5
    plt.rcParams.update({"font.size": 22})

    v0_range = np.linspace(-1, 5, 10)
    t = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 0.05]}
    )

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=v0_range.min(), vmax=v0_range.max())

    for v0_start in v0_range:
        t_global = []
        v_global = []
        v0 = v0_start
        for n in range(N):
            v_fun = lambda t, v0: v0 + 3 / 2 * (1 - v0) * (2 * t - t**2)
            v = v_fun(t, v0)
            v0 = v[-1]
            t_global.extend(n + t)
            v_global.extend(v)

        axes[0].plot(t_global, v_global, c=cmap(norm(v0_start)))

    axes[0].set_title(r"$\mathcal{V(T)}$")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for n in range(1, N + 1):
        axes[0].axvline(x=n, color="black", linestyle="--", linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=axes[1])
    cbar.set_label(r"$\mathcal{V}_0$")

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    fig.savefig(f"images/{filename}")


def extra():
    N = 10
    a = [[0, 0, 1], [2, 1, 0], [2, 3, 6]]
    t = np.linspace(0, 1, 200)
    for n in range(1, N + 1):
        b = [0, 0, 6 * n]
        a = np.array(a)
        b = np.array(b)
        sol = np.linalg.solve(a, b)
        v_fun = lambda t: sol[0] * np.power(t, 2) + sol[1] * t + sol[2]
        v = v_fun(t)
        print(n, v[-1])


def get_parameters_extra(N, tns_cumsum, v0):
    def derivative_eq(n, t, N):
        line = []
        for _ in range(n):
            line += [0, 0, 0]
        line += [2 * t, 1, 0, -2 * t, -1, 0]
        for _ in range(n + 2, N):
            line += [0, 0, 0]

        return line

    def continuity_eq(n, t, N):
        line = []
        for _ in range(n):
            line += [0, 0, 0]
        line += [t**2, t, 1, -(t**2), -t, -1]
        for _ in range(n + 2, N):
            line += [0, 0, 0]

        return line

    def distance_eq(n, tns_cumsum, N):
        line = []
        for i in range(n + 1):
            tn_prev = tns_cumsum[i - 1] if n != 0 else 0
            tn = tns_cumsum[i]
            line += [2 * (tn ** 3 - tn_prev ** 3), 3 * (tn ** 2 - tn_prev ** 2), 6 * (tn - tn_prev)]
        for _ in range(n + 1, N):
            line += [0, 0, 0]

        return line

    a = []
    b = [v0, 0]
    a.append([0, 0, 1] + [0 for i in range(3, 3 * N)])
    a.append([0 for i in range(3, 3 * N)] + [2 * tns_cumsum[-1], 1, 0])

    for n in range(N):
        tn = tns_cumsum[n]
        a.append(distance_eq(n, tns_cumsum, N))
        b.append(6 * (n + 1))
        if n == N - 1:
            continue
        a.append(derivative_eq(n, tn, N))
        a.append(continuity_eq(n, tn, N))
        b += [0, 0]

    a = np.array(a)
    b = np.array(b)
    solved = np.linalg.solve(a, b)

    return solved


def multiple_extra(filename):
    plt.rcParams.update({"font.size": 22})
    N = 6
    tns = np.random.uniform(0.1, 1, N)
    tns_cumsum = np.cumsum(tns)

    v0_range = np.linspace(-1, 5, 10)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 0.05]}
    )

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=v0_range.min(), vmax=v0_range.max())

    for v0_start in v0_range:
        solved = get_parameters_extra(N, tns_cumsum, v0_start)
        v_global = []
        t_global = []
        for n in range(N):
            tn_prev = tns_cumsum[n - 1] if n != 0 else 0
            tn = tns_cumsum[n]
            t = np.linspace(tn_prev, tn, 200)
            sol = solved[n * 3 : (n + 1) * 3]
            v_fun = lambda t: sol[0] * np.power(t, 2) + sol[1] * t + sol[2]
            v = v_fun(t)
            t_global.extend(t)
            v_global.extend(v)

        axes[0].plot(t_global, v_global, c=cmap(norm(v0_start)))

    axes[0].set_title(r"$\mathcal{V(T)}$")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for n in tns_cumsum:
        axes[0].axvline(x=n, color="black", linestyle="--", linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=axes[1])
    cbar.set_label(r"$\mathcal{V}_0$")

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    fig.savefig(f"images/{filename}")


# unknown_final_v()
# other_functionals()
# limit_v("sol2.pdf")
# draw_multiple_naive("sol_multiple_naive.pdf")
# multiple("sol_multiple.pdf")
# extra()
# multiple_extra("sol_extra1.pdf")
