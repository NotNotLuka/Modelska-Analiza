import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import matplotlib.cm as cm
import matplotlib.colors as mcolors


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def lotka(x, y, p):
    z, l = y
    return [p * z * (1 - l), l / p * (z - 1)]


def classic():
    p = 0.01
    t_span = (0, 36)
    t_eval = np.linspace(t_span[0], t_span[1], 10_000)

    fun = lambda x, y: lotka(x, y, p)
    sol = scipy.integrate.solve_ivp(
        fun, t_span, [0.85, 1.5], method="RK45", t_eval=t_eval
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot results
    labels = ["z", "l"]
    for i in range(2):
        ax.plot(sol.t, sol.y[i], label=labels[i])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$z$")
    ax.legend()
    ax.grid(alpha=0.8)
    # fig.savefig("images/lotka2.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sol.y[1], sol.y[0])
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$l$")
    ax.grid(alpha=0.8)
    # fig.savefig("images/phase2.pdf")
    plt.show()


def phase():
    p = 0.01
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 1000)

    fun = lambda x, y: lotka(x, y, p)

    n_fox_values = np.linspace(0.1, 2, 20)  # smaller range
    norm = mcolors.Normalize(vmin=min(n_fox_values), vmax=max(n_fox_values))
    cmap = cm.coolwarm

    fig, ax = plt.subplots(figsize=(8, 6))
    print(n_fox_values)
    for n_fox in n_fox_values:
        sol = scipy.integrate.solve_ivp(fun, t_span, [0.85, n_fox], t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1], color=cmap(norm(n_fox)), lw=1.8)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, label=r"$z_0$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$l$")
    ax.grid(alpha=0.8)
    fig.savefig("images/phase3.pdf")
    plt.show()


def draw_periods():
    def find_period(y, atol=1e-1, min_sep=5):
        y = (y - np.min(y)) / np.max(y)
        y = y - np.max(y)

        for tol in range(1, 10):
            cand = np.where(np.isclose(y, 0, atol=tol * atol))[0]
            if len(cand) < 2:
                print("first")
                return np.nan

            peaks = []
            cluster = []
            for i in cand:
                if len(cluster) != 0 and min_sep <= i - cluster[-1]:
                    peak = np.argmax(y[cluster])
                    peaks.append(cluster[peak])
                    cluster = []
                else:
                    cluster.append(i)
            if len(cluster) != 0:
                peak = np.argmax(y[cluster])
                peaks.append(cluster[peak])
                cluster = []

            peaks = np.asarray(peaks)
            if 2 < len(peaks):
                break

        if len(peaks) < 2:
            return np.nan
        mask = np.arange(0, len(peaks) - 1)
        return np.average(peaks[mask + 1] - peaks[mask])
        return peaks[1] - peaks[0]

    p = 0.7
    fun = lambda t, y: lotka(t, y, p)

    t_span = (0, 30)
    t_eval = np.linspace(*t_span, 500)
    z0_values = np.linspace(0.1, 5.0, 100)
    l0_values = np.linspace(0.1, 5.0, 100)

    mat = np.full((len(l0_values), len(z0_values)), np.nan)

    for i, l0 in enumerate(l0_values):
        for j, z0 in enumerate(z0_values):
            sol = scipy.integrate.solve_ivp(fun, t_span, [z0, l0], t_eval=t_eval)
            z, l = sol.y

            period_z = find_period(z)
            period_l = find_period(l)
            #period = (period_z + period_l) // 2
            period = period_z
            if np.isnan(period_l):
                period = period_z
            if np.isnan(period_z):
                period = period_l
            if np.isnan(period):
                print(l0, z0)
                # plt.plot(z)
                # plt.show()
                continue
            mat[i, j] = t_eval[int(period)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        mat,
        extent=[z0_values.min(), z0_values.max(), l0_values.min(), l0_values.max()],
        origin="lower",
        aspect="auto",
        cmap="seismic"
    )
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$l_0$")
    fig.colorbar(im, label="Perioda")
    fig.savefig("images/periods.pdf")
    plt.show()


def stream():
    x = np.linspace(0, 5, 1000)
    y = np.linspace(0, 5, 1000)
    X, Y = np.meshgrid(x, y)
    U, V = lotka(None, (X, Y), 0.01)

    speed = np.hypot(U, V)  # magnitude for color mapping
    speed = np.log1p(speed)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.streamplot(
        X, Y, U, V, color=speed, cmap=cm.seismic, density=1.0, linewidth=0.8
    )
    ax.set_xlabel("z")
    ax.set_ylabel("l")
    fig.savefig("images/phase3.pdf")
    plt.show()


# classic()
# stream()
# phase()
draw_periods()
# draw_ones()

# TO DO: investigate ratios, fitting real world data
