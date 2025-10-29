import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import scipy
import scipy.integrate
import matplotlib.cm as cm


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def laser(x, y, p, R):
    f, a = y
    return [p * f * (a - 1), -a / p * (1 + f) + R / p]


def classic():
    p = 2
    R = 1.5
    t_span = (0, 40)
    t_eval = np.linspace(*t_span, 10000)

    fun = lambda x, y: laser(x, y, p, R)
    sol = scipy.integrate.solve_ivp(
        fun, t_span, [R - 1 + 1e-2, 1 - 1e-2], method="RK45", t_eval=t_eval
    )

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    l1 = ax.plot(sol.t, sol.y[0], label=r"$\mathcal{F}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathcal{F}$")
    ax1 = ax.twinx()

    l2 = ax1.plot(sol.t, sol.y[1], color="red", label=r"$\mathcal{A}$")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\mathcal{A}$")
    ax.legend(l1 + l2, [h.get_label() for h in (l1 + l2)])

    fig.tight_layout()
    fig.savefig("images/damped.pdf")

    plt.show()


def stream():
    p = 1
    R = 0.5
    x = np.linspace(0, 2, 1000)
    y = np.linspace(0, 2, 1000)
    X, Y = np.meshgrid(x, y)
    U, V = laser(None, (X, Y), p, R)

    speed = np.hypot(U, V)  # magnitude for color mapping
    speed = np.log1p(speed)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.streamplot(X, Y, U, V, color=speed, cmap=cm.seismic, density=1.0, linewidth=0.8)
    ax.scatter([0, 1 - R], [R, 1], marker="o", color="black", label="Stacionarne točke")

    ax.set_xlabel(r"$\mathcal{F}$")
    ax.set_ylabel(r"$\mathcal{A}$")
    fig.legend()
    fig.savefig("images/laserphase.pdf")
    plt.show()


classic()
# stream()
