import matplotlib.pyplot as plt
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


def fit_to_data(linear=False, jacobian=False, p0=1):
    def jacobian_full(x, a0, a1, p):
        df_da0 = np.ones_like(x)
        df_da1 = x**p
        df_dp = a1 * (x**p) * np.log(x)
        return np.vstack((df_da0, df_da1, df_dp)).T

    def jacobian_linear(x, a0, a1):
        df_da0 = np.ones_like(x)
        df_da1 = x
        return np.vstack((df_da0, df_da1)).T

    xinv = 1.0 / x
    yinv = 1.0 / y
    dyinv = dy / (y**2)  # propagated error of 1/y

    full_model = lambda x, a0, a1, p: a0 + a1 * (x**p)
    if linear:
        jac = jacobian_linear
        model = lambda x, a0, a1: full_model(x, a0, a1, 1)
        initial = None
    else:
        a0_guess = yinv[-1]
        a1_guess = yinv[0] - yinv[-1]
        initial = [a0_guess, a1_guess, p0]
        jac = jacobian_full
        model = full_model

    const, cov = scipy.optimize.curve_fit(
        model,
        xinv,
        yinv,
        sigma=dyinv,
        absolute_sigma=True,
        jac=jac if jacobian else None,
        p0=initial,
    )

    res = yinv - model(xinv, *const)
    chi2_val = np.sum((res / dyinv) ** 2)
    ndof = len(yinv) - len(const)
    chi2_red = chi2_val / ndof
    p_value = 1 - scipy.stats.chi2.cdf(chi2_val, ndof)

    print(f"χ² = {chi2_val:.3f}")
    print(f"reduced χ² = {chi2_red:.3f}")
    print(f"p-value = {p_value:.3f}")
    print(f"certainty = {(2 * p_value):.1f}")

    x_sim = np.linspace(x.min(), x.max(), 1000)
    y_sim = 1 / model(1 / x_sim, *const)

    J = jac(1 / x_sim, *const)
    J = J * (-y_sim[:, None] ** 2)

    # var_y(x_i) = J_i * cov * J_i^T
    var_y = np.einsum("ij,jk,ik->i", J, cov, J)
    err = np.sqrt(var_y)

    y1 = y_sim - err
    y2 = y_sim + err

    fig, ax = plt.subplots(figsize=(8, 6))
    r = (yinv - model(xinv, *const)) / dyinv
    ax.errorbar(range(len(r)), r, fmt='o', color='black')
    ax.axhline(0, color='#1f77b4', lw=1)
    ax.set_xlabel(r"Indeks meritve")
    ax.set_ylabel(r"$\frac{y_i-f_i}{\sigma_i}$")
    ax.grid()
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig("images/reslin.pdf")

    return x_sim, y_sim, y1, y2, chi2_val, const


def surface(x, y, dy, const):
    xinv = 1.0 / x
    yinv = 1.0 / y
    dyinv = dy / (y**2)  # propagated error of 1/y

    model = lambda x, a0, a1, p: a0 + a1 * (x**p)

    a0, a1, p = const
    a0_grid = np.linspace(a0 * 0.5, a0 * 1.7, 200)
    a1_grid = np.linspace(a1 * 0.5, a1 * 1.7, 200)
    p_grid = np.linspace(p * 0.5, p * 1.7, 200)

    chi2_map = np.zeros((len(a1_grid), len(p_grid)))

    for i, a0_ in enumerate(a0_grid):
        for j, a1_ in enumerate(a1_grid):
            yfit = model(xinv, a0_, a1_, p)
            res = (yinv - yfit) / dyinv
            chi2_map[i, j] = np.sum(res**2)

    pval_map = 1 - scipy.stats.chi2.cdf(chi2_map, len(yinv) - len(const))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pval_map.T,
        origin="lower",
        extent=[a1_grid[0], a1_grid[-1], p_grid[0], p_grid[-1]],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("$a_0$")
    ax.set_ylabel("$a_1$")
    fig.colorbar(im, ax=ax)
    fig.savefig("images/a0a1.pdf")
    plt.show()


def draw_chi_val(chi2_val, ndof):
    xvals = np.linspace(0, chi2_val * 3, 400)
    pdf = scipy.stats.chi2.pdf(xvals, ndof)

    plt.plot(xvals, pdf)
    plt.axvline(chi2_val, color="r")
    plt.fill_between(xvals, 0, pdf, where=xvals >= chi2_val, color="r", alpha=0.3)

    plt.xlabel(r"$\chi^2$")
    plt.ylabel("Verjetnostna gostota")
    plt.legend()
    plt.show()


def draw(x_sim, y_sim, y1, y2, filename=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x_sim, y1, y2, color="skyblue", alpha=0.5)
    ax.grid()
    ax.plot(x_sim, y_sim)
    ax.errorbar(
        df["x[doza]"],
        df["y[odziv]"],
        yerr=df["dy[odziv]"],
        fmt="o",
        capsize=3,
        color="black",
    )
    ax.set_xscale("log")
    ax.set_xlabel("x[doza]")
    ax.set_ylabel("y[odziv]")
    if filename is not None:
        plt.savefig(f"images/{filename}.pdf")
    plt.show()


df = pd.read_csv("data/farmakoloski.dat", sep=r"\t+", engine="python")
print(len(df))
# data
x = df["x[doza]"].to_numpy()
y = df["y[odziv]"].to_numpy()
dy = df["dy[odziv]"].to_numpy()

x_sim, y_sim, y1, y2, chi2_val, const = fit_to_data(linear=True)
# draw_chi_val(chi2_val, 6)
# draw(x_sim, y_sim, y1, y2)
# surface(x, y, dy, const)
