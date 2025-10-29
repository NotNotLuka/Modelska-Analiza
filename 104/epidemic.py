import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def classic(x, y, alpha, beta):
    D, B, I = y
    return [-alpha * D * B, alpha * D * B - beta * B, beta * B]


def simulate_normal(alpha):
    beta = 0.1
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 5000)

    fun = lambda x, y: classic(x, y, alpha, beta)
    sol = scipy.integrate.solve_ivp(
        fun, t_span, [0.99, 0.01, 0], method="RK45", t_eval=t_eval
    )

    return sol


def normal():
    alpha = 1
    sol = simulate_normal(alpha)
    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["D", "B", "I"]
    for i in range(3):
        ax.plot(sol.t, sol.y[i], label=labels[i])

    ax.set_xlabel("t")
    ax.set_ylabel("$N/N_0$")
    ax.legend()
    # fig.savefig("images/basic1.pdf")

    maximums = []
    reproduction = []
    for alpha in np.linspace(0, 1, 2000):
        reproduction.append(alpha * 0.99 / 0.1)
        sol = simulate_normal(alpha)
        maximums.append(1 - sol.y[0][-1])
        # maximums.append(np.max(sol.y[1]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(reproduction, maximums)
    ax.set_xlabel("R")
    ax.set_ylabel(r"$1 - D$")
    # ax.set_ylabel(r"$\max{(B)}$")
    ax.set_xticks(range(int(max(reproduction)) + 1))
    ax.set_xlim(min(reproduction), max(reproduction))
    ax.grid()
    fig.savefig("images/sums.pdf")
    plt.show()


def time_maximums():
    maximums = []
    after_maximums = []
    reproduction = []
    reproduction_after = []
    for alpha in np.linspace(0, 1, 2000):
        reproduction.append(alpha * 0.99 / 0.1)
        sol = simulate_normal(alpha)
        t_max = np.argmax(sol.y[1])
        diff = np.abs(sol.y[1] - sol.y[0])
        t_eq = np.argmin(diff)
        while t_max - t_eq < 0 and 10 < t_eq:
            t_eq = np.argmin(diff[: t_eq - 10])
        if t_eq <= 10:
            t_eq = 0

        maximums.append(t_max)

        # print(t_eq, t_max)

        if np.abs(sol.y[1] - sol.y[0])[t_eq] < 1e-3:  # and 0 <= t_max - t_eq:
            after_maximums.append(t_max - t_eq)
            reproduction_after.append(alpha * 0.99 / 0.1)
        else:
            print(np.abs(sol.y[1] - sol.y[0])[t_eq], t_max - t_eq)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(reproduction_after, after_maximums)
    ax.set_xlabel("R")
    ax.set_ylabel(r"$t_{max} - t_{eq}$")
    # ax.set_ylabel(r"$\max{(B)}$")
    ax.set_xticks(range(int(max(reproduction)) + 1))
    ax.set_xlim(min(reproduction), max(reproduction))
    ax.grid()
    fig.savefig("images/afterequalityt.pdf")
    # fig.savefig("images/maximumt.pdf")
    plt.show()


def vaccination(x, y, alpha, beta, gamma, t0):
    D, B, I, V = y
    if x < t0:
        gamma = 0
    return [-alpha * D * B - gamma * D, alpha * D * B - beta * B, beta * B, gamma * D]


def simulate_vaccinated(alpha, t0=10):
    beta = 0.1
    gamma = 1e-1
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 5000)

    fun = lambda x, y: vaccination(x, y, alpha, beta, gamma, t0)
    sol = scipy.integrate.solve_ivp(
        fun, t_span, [0.99, 0.01, 0, 0], method="RK45", t_eval=t_eval
    )

    return sol


def vaccinated_maximums():
    maximums = []
    reproduction = []
    for alpha in np.linspace(0, 1, 2000):
        reproduction.append(alpha * 0.99 / 0.1)
        sol0 = simulate_normal(alpha)
        maximum0 = np.max(sol0.y[1])

        sol1 = simulate_vaccinated(alpha)
        maximum1 = np.max(sol1.y[1])

        maximums.append((maximum0 - maximum1) / maximum0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(reproduction, maximums)
    ax.set_xlabel("R")
    ax.set_ylabel(r"$(\max{(B)} - \max{(B_{vac})}) / \max{(B)}$")
    ax.set_xticks(range(int(max(reproduction)) + 1))
    ax.set_xlim(min(reproduction), max(reproduction))
    ax.grid()
    fig.savefig("images/vaccmaxrel.pdf")
    plt.show()


def normal_length():
    alphas = np.linspace(0, 1, 1000)
    length = []
    maximums = []
    reproduction = []
    ts = []
    for alpha in alphas:
        R = alpha * 0.99 / 0.1
        reproduction.append(R)
        sol1 = simulate_normal(alpha)
        R_eff = alpha * sol1.y[0] / 0.1 < 1
        length.append(np.argmax(R_eff))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(reproduction, length)
    ax.set_xlabel("$R$")
    ax.set_ylabel(r"$t_{epidemic}$")
    ax.grid()
    fig.savefig("images/epidemiclengthnorm.pdf")
    plt.show()


def vaccinated_length():
    alphas = np.linspace(0, 0.3, 5)  # values of alpha
    t0s = np.linspace(0, 100, 2000)

    fig, ax = plt.subplots(figsize=(8, 6))
    R_min = alphas[0] * 0.99 / 0.1
    R_max = alphas[-1] * 0.99 / 0.1
    norm = Normalize(vmin=R_min, vmax=R_max)
    cmap = plt.cm.coolwarm

    for alpha in alphas:
        R = alpha * 0.99 / 0.1
        length = []
        maximums = []
        for t0 in t0s:
            sol1 = simulate_vaccinated(alpha, t0=t0)
            R_eff = alpha * sol1.y[0] / 0.1 < 1
            length.append(np.argmax(R_eff))
            maximum = np.max(sol1.y[1])
            maximums.append(maximum)

        ax.plot(t0s, maximums, color=cmap(norm(R)), label=rf"$R={R}$")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$R$")

    ax.set_xlabel(r"$t_0$")
    # ax.set_ylabel(r"$t_{epidemic}$")
    ax.set_ylabel(r"$\max{(B)}$")
    ax.grid()
    fig.tight_layout()
    fig.savefig("images/epidemiclengtht.pdf")
    plt.show()


def vac():
    alpha = 1
    beta = 0.1
    gamma = 1e-1
    t_span = (0, 30)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    fun = lambda x, y: vaccination(x, y, alpha, beta, gamma, 1)
    sol = scipy.integrate.solve_ivp(
        fun, t_span, [1e4 / 10100, 100 / 10100, 0, 0], method="RK45", t_eval=t_eval
    )

    # Plot results
    for i in range(4):
        plt.plot(sol.t, sol.y[i], label=f"y{i+1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def stages(x, y, alpha, beta, gamma, delta, epsilon):
    D, S, B, I, M = y
    return [
        -alpha * D * B - beta * D * S,
        alpha * D * B + beta * D * S - (gamma + delta) * S,
        gamma * S - epsilon * B,
        epsilon * B,
        delta * S,
    ]


def stag_simulation(alpha, beta):
    gamma = 1e-1
    delta = 1e-2
    epsilon = 2e-2
    t_span = (0, 400)
    t_eval = np.linspace(t_span[0], t_span[1], 4000)

    fun = lambda x, y: stages(x, y, alpha, beta, gamma, delta, epsilon)
    sol = scipy.integrate.solve_ivp(
        fun,
        t_span,
        [1e4 / 10100, 100 / 10100, 0, 0, 0],
        method="RK45",
        t_eval=t_eval,
    )

    return sol


def stag():
    fig, ax = plt.subplots(figsize=(8, 6))
    alpha = 2e-1
    beta = 8e-1
    sol = stag_simulation(alpha, beta)
    # Plot results
    labels = ["D", "S", "B", "I", "M"]
    for i in range(5):
        ax.plot(sol.t, sol.y[i], label=labels[i])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$N/N_0$")
    ax.legend()
    fig.savefig("images/basicstag.pdf")
    plt.show()


def stag_maxs():
    alphas = np.linspace(0, 0.5, 1000)
    fig, ax = plt.subplots(figsize=(8, 6))
    reproduction = []
    deaths = []
    maximums = []
    max_t = []
    for a in alphas:
        alpha = a / 3
        beta = a / 3 * 2
        R = (alpha + beta) * 0.99 / (1e-2 + 2e-2 + 2e-2)
        sol = stag_simulation(alpha, beta)
        reproduction.append(R)
        t_max = np.argmax(sol.y[1] + sol.y[2])
        max_t.append(t_max)
        maximums.append((sol.y[1] + sol.y[2])[t_max])
        deaths.append(sol.y[4][-1])

    ax.plot(reproduction, max_t)
    ax.set_xlabel("$R$")
    ax.set_ylabel(r"$t_{max}$")
    ax.set_xticks(range(11))
    ax.grid()
    fig.savefig("images/stagmaxt.pdf")
    plt.show()


# stag()
stag_maxs()
# vac()
# normal()
# time_maximums()
# vaccinated_maximums()
# vaccinated_length()
# normal_length()

