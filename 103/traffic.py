import numpy as np
import scipy
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def constraint(x, v0, dist=1):
    dt = 1 / x.shape[0]
    return dist - dt * (0.5 * v0 + np.sum(x[:-1]) + 0.5 * x[-1])


def lagrangian(x, v0, lambda_=None, speed_limit=None, dist_limit=None, above_zero=False):
    dt = 1 / x.shape[0]

    v_i1 = x
    v_i = np.insert(x, 0, v0)[:-1]

    acc = ((v_i1 - v_i) / dt) ** 2
    L = acc
    if lambda_ is not None:
        L += - lambda_ * v_i

    S = np.sum(L * dt)
    if speed_limit is not None:
        beta = 3
        violation = np.maximum(0, np.abs(v_i1) - speed_limit)
        S += np.sum(np.exp(beta * violation) - 1)
    if above_zero:
        beta = 8
        violation = np.maximum(0, - v_i1)
        S += np.sum(np.exp(beta * violation) - 1)
        if False and np.sum(violation) > 0.2:
            print(L)
    if dist_limit is not None:
        beta = 40
        S += np.exp(beta * (constraint(x, v0, dist=dist_limit) ** 2))

    return S


def find_lambda(lambda_, v0, x0, speed_limit=None, dist_limit=None, above_zero=False):
    minimize = lambda x: lagrangian(x, v0, lambda_, speed_limit=speed_limit, dist_limit=dist_limit, above_zero=True)
    res = scipy.optimize.minimize(minimize, x0=x0, method="Powell")

    dist = constraint(res.x, v0)
    print(dist)
    return dist


def draw_basic():
    v0s = np.arange(-1, 6, 1)
    t = np.linspace(0, 1, 100)
    cmap = plt.cm.seismic
    norm = plt.Normalize(v0s.min(), v0s.max())

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for v0 in v0s:
        x0 = np.random.randn(100)
        final = lambda lambda_: find_lambda(lambda_, v0, x0)
        lambda_ = scipy.optimize.brentq(final, -5000, 5000, xtol=1e-1)
        minimize = lambda x: lagrangian(x, v0, lambda_)
        res = scipy.optimize.minimize(minimize, x0=x0, method="Powell")

        ax.plot(t, res.x, color=cmap(norm(v0)))
        print("done ", v0)

    # Add colorbar instead of legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.set_label("$v_0$")

    ax.set_xlabel("t")
    ax.set_ylabel("v")
    fig.savefig("images/numerical.pdf")

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for v0 in v0s:
        v = v0 + 3 * (1 - v0) / 2 * (2 * t - t ** 2)
        ax.plot(t, v, color=cmap(norm(v0)))

    # Add colorbar instead of legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.set_label("$v_0$")

    ax.set_xlabel("t")
    ax.set_ylabel("v")
    fig.savefig("images/analytical.pdf")
    plt.show()


def draw_speed_limit():
    v0s = np.arange(-1, 6, 1)
    t = np.linspace(0, 1, 100)
    cmap = plt.cm.seismic
    norm = plt.Normalize(v0s.min(), v0s.max())

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for v0 in v0s[::-1]:
        x0 = np.random.randn(100)
        final = lambda lambda_: find_lambda(lambda_, v0, x0, speed_limit=1.5)
        lambda_ = scipy.optimize.brentq(final, -50, 50, xtol=1e-1)
        minimize = lambda x: lagrangian(x, v0, lambda_)
        res = scipy.optimize.minimize(minimize, x0=x0, method="Powell")

        ax.plot(t, res.x, color=cmap(norm(v0)))
        print("done ", v0)

    # Add colorbar instead of legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.set_label("$v_0$")

    ax.set_xlabel("t")
    ax.set_ylabel("v")
    fig.savefig("images/speed1numerical.pdf")
    plt.show()


def draw_dist_limit():
    v0s = np.arange(-1, 6, 1)
    t = np.linspace(0, 1, 100)
    cmap = plt.cm.seismic
    norm = plt.Normalize(v0s.min(), v0s.max())

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for v0 in v0s:
        x0 = np.random.randn(100)
        minimize = lambda x: lagrangian(x, v0, dist_limit=1)
        res = scipy.optimize.minimize(minimize, x0=x0, method="Powell")
        print(constraint(res.x, v0, 1))
        ax.plot(t, res.x, color=cmap(norm(v0)))
        print("done ", v0)

    # Add colorbar instead of legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.set_label("$v_0$")

    ax.set_xlabel("t")
    ax.set_ylabel("v")
    fig.savefig("images/distnumerical.pdf")
    plt.show()


draw_speed_limit()
# draw_dist_limit()
