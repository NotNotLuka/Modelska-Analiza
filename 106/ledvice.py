import pandas as pd
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


def enorazde_sqrt():
    def jacobian(t, c, A, lambda_):
        dcd_lnA = c
        dcd_lambda = -np.sqrt(t_sim) * c
        return np.vstack((dcd_lnA, dcd_lambda))

    oneraz = lambda t, lnA, lambda_: lnA - lambda_ * np.sqrt(t)

    const, cov = scipy.optimize.curve_fit(
        oneraz, t, np.log(N), sigma=np.sqrt(N) / N, absolute_sigma=False
    )
    t_sim = np.linspace(t.min(), t.max(), 200)
    c = np.exp(oneraz(t_sim, *const))

    jac = jacobian(t, c, *const)
    err = np.sqrt(
        jac[0] ** 2 * cov[0, 0]
        + jac[1] ** 2 * cov[1, 1]
        + 2 * jac[0] * jac[1] * cov[0, 1]
    )

    return t_sim, c, err



def enorazde_sqrt():
    def jacobian(t, A, lambda_):
        e = np.exp(-lambda_ * np.sqrt(t))
        dcd_A = e
        dcd_lambda = -np.sqrt(t) * A * e
        return np.vstack((dcd_A, dcd_lambda))

    def model(t, A, lambda_):
        return A * np.exp(-lambda_ * np.sqrt(t))

    const, cov = scipy.optimize.curve_fit(
        model, t, N, sigma=np.sqrt(N), absolute_sigma=True,
        p0=[N.max(), 0.1 / np.sqrt(t.max())],
        bounds=(0, np.inf)
    )

    t_sim = np.linspace(t.min(), t.max(), 200)
    c = model(t_sim, *const)
    jac = jacobian(t_sim, *const)

    err = np.sqrt(
        jac[0]**2 * cov[0, 0]
        + jac[1]**2 * cov[1, 1]
        + 2 * jac[0] * jac[1] * cov[0, 1]
    )

    return t_sim, c, err, np.sum((N - model(t, *const)) ** 2)



def enorazde_background():
    def jacobian(t, c, A, lambda_, C):
        dcd_A = np.exp(-lambda_ * t)
        dcd_lambda = -t * c
        dcd_C = np.ones_like(t)
        return np.vstack((dcd_A, dcd_lambda, dcd_C))

    model = lambda t, A, lambda_, C: A * np.exp(-lambda_ * t) + C

    const, cov = scipy.optimize.curve_fit(
        model,
        t,
        N,
        sigma=np.sqrt(N),
        absolute_sigma=True,
        p0=[N.max(), 0.1 / (t.max() - t.min()), N.min()],
    )
    t_sim = np.linspace(t.min(), t.max(), 200)
    c = model(t_sim, *const)

    jac = jacobian(t_sim, c, *const)
    err = np.sqrt(
        jac[0] ** 2 * cov[0, 0]
        + jac[1] ** 2 * cov[1, 1]
        + jac[2] ** 2 * cov[2, 2]
        + 2 * jac[0] * jac[1] * cov[0, 1]
        + 2 * jac[0] * jac[2] * cov[0, 2]
        + 2 * jac[1] * jac[2] * cov[1, 2]
    )

    return t_sim, c, err, np.sum((N - model(t, *const)) ** 2)


def enorazde():
    def jacobian(t, c, A, lambda_):
        dcd_lnA = c
        dcd_lambda = -t_sim * c
        return np.vstack((dcd_lnA, dcd_lambda))

    oneraz = lambda t, lnA, lambda_: lnA - lambda_ * t

    const, cov = scipy.optimize.curve_fit(
        oneraz, t, np.log(N), sigma=np.sqrt(N) / N, absolute_sigma=False
    )
    t_sim = np.linspace(t.min(), t.max(), 200)
    c = np.exp(oneraz(t_sim, *const))

    jac = jacobian(t, c, *const)
    err = np.sqrt(
        jac[0] ** 2 * cov[0, 0]
        + jac[1] ** 2 * cov[1, 1]
        + 2 * jac[0] * jac[1] * cov[0, 1]
    )

    return t_sim, c, err



def enorazde():
    def jacobian(t, A, lambda_):
        e = np.exp(-lambda_ * t)
        dcd_A = e
        dcd_lambda = -t * A * e
        return np.vstack((dcd_A, dcd_lambda))

    def model(t, A, lambda_):
        return A * np.exp(-lambda_ * t)

    const, cov = scipy.optimize.curve_fit(
        model, t, N, sigma=np.sqrt(N), absolute_sigma=True,
        p0=[N.max(), 0.1 / (t.max() - t.min())],
        bounds=(0, np.inf)
    )

    t_sim = np.linspace(t.min(), t.max(), 200)
    c = model(t_sim, *const)
    jac = jacobian(t_sim, *const)

    err = np.sqrt(
        jac[0]**2 * cov[0, 0]
        + jac[1]**2 * cov[1, 1]
        + 2 * jac[0] * jac[1] * cov[0, 1]
    )

    return t_sim, c, err, np.sum((N - model(t, *const)) ** 2)



def dvorazde():
    def jacobian(t, A1, lambda1, A2, lambda2):
        dcd_A1 = np.exp(-lambda1 * t)
        dcd_lambda1 = -t * A1 * np.exp(-lambda1 * t)
        dcd_A2 = np.exp(-lambda2 * t)
        dcd_lambda2 = -t * A2 * np.exp(-lambda2 * t)
        return np.vstack((dcd_A1, dcd_lambda1, dcd_A2, dcd_lambda2))

    def model(t, A1, lambda1, A2, lambda2):
        return A1 * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t)

    const, cov = scipy.optimize.curve_fit(
        model,
        t,
        N,
        sigma=np.sqrt(N),
        absolute_sigma=True,
        p0=[
            N.max(),
            0.1 / (t.max() - t.min()),
            N.max() / 2,
            0.01 / (t.max() - t.min()),
        ],
    )

    t_sim = np.linspace(t.min(), t.max(), 200)
    c = model(t_sim, *const)
    jac = jacobian(t_sim, *const)

    err = np.sqrt(
        jac[0] ** 2 * cov[0, 0]
        + jac[1] ** 2 * cov[1, 1]
        + jac[2] ** 2 * cov[2, 2]
        + jac[3] ** 2 * cov[3, 3]
        + 2
        * (
            jac[0] * jac[1] * cov[0, 1]
            + jac[0] * jac[2] * cov[0, 2]
            + jac[0] * jac[3] * cov[0, 3]
            + jac[1] * jac[2] * cov[1, 2]
            + jac[1] * jac[3] * cov[1, 3]
            + jac[2] * jac[3] * cov[2, 3]
        )
    )

    return t_sim, c, err, np.sum((N - model(t, *const)) ** 2)



def dvorazde_sqrt():
    def jacobian(t, A1, lambda1, A2, lambda2):
        e1 = np.exp(-lambda1 * np.sqrt(t))
        e2 = np.exp(-lambda2 * np.sqrt(t))
        dcd_A1 = e1
        dcd_lambda1 = -np.sqrt(t) * A1 * e1
        dcd_A2 = e2
        dcd_lambda2 = -np.sqrt(t) * A2 * e2
        return np.vstack((dcd_A1, dcd_lambda1, dcd_A2, dcd_lambda2))

    def model(t, A1, lambda1, A2, lambda2):
        return A1 * np.exp(-lambda1 * np.sqrt(t)) + A2 * np.exp(-lambda2 * np.sqrt(t))

    const, cov = scipy.optimize.curve_fit(
        model,
        t,
        N,
        sigma=np.sqrt(N),
        absolute_sigma=True,
        p0=[N.max(), 0.1 / np.sqrt(t.max()), N.max() / 2, 0.01 / np.sqrt(t.max())],
        bounds=(0, np.inf),
    )

    t_sim = np.linspace(t.min(), t.max(), 200)
    c = model(t_sim, *const)
    jac = jacobian(t_sim, *const)

    err = np.sqrt(
        sum(jac[i] ** 2 * cov[i, i] for i in range(4))
        + 2
        * sum(jac[i] * jac[j] * cov[i, j] for i in range(4) for j in range(i + 1, 4))
    )

    return t_sim, c, err, np.sum((N - model(t, *const)) ** 2)



def dvorazde_background():
    def jacobian(t, A1, lambda1, A2, lambda2, C):
        e1 = np.exp(-lambda1 * t)
        e2 = np.exp(-lambda2 * t)
        dcd_A1 = e1
        dcd_lambda1 = -t * A1 * e1
        dcd_A2 = e2
        dcd_lambda2 = -t * A2 * e2
        dcd_C = np.ones_like(t)
        return np.vstack((dcd_A1, dcd_lambda1, dcd_A2, dcd_lambda2, dcd_C))

    def model(t, A1, lambda1, A2, lambda2, C):
        return A1 * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t) + C

    const, cov = scipy.optimize.curve_fit(
        model,
        t,
        N,
        sigma=np.sqrt(N),
        absolute_sigma=True,
        p0=[
            N.max(),
            0.1 / (t.max() - t.min()),
            N.max() / 2,
            0.01 / (t.max() - t.min()),
            N.min(),
        ],
        bounds=(0, np.inf),
    )

    t_sim = np.linspace(t.min(), t.max(), 200)
    c = model(t_sim, *const)
    c_og = model(t, *const)
    jac = jacobian(t_sim, *const)

    err = np.sqrt(
        sum(jac[i] ** 2 * cov[i, i] for i in range(5))
        + 2
        * sum(jac[i] * jac[j] * cov[i, j] for i in range(5) for j in range(i + 1, 5))
    )

    return t_sim, c, err, np.sum((N - model(t, *const)) ** 2)


def f_test(fun1, fun2, k1, k2, n):
    _, _, _, RSS1 = fun1()
    _, _, _, RSS2 = fun2()

    num = (RSS1 - RSS2) / (k2 - k1)
    den = RSS2 / (n - k2)
    Fval = num / den

    p = 1 - scipy.stats.f.cdf(Fval, k2 - k1, n - k2)
    return Fval, p



df = pd.read_csv("data/ledvice.dat", sep=r"\t+", engine="python")
t = df["t[čas]"]
N = df["N[sunkov na detektorju]"]

functions = [
    (enorazde, "blue", r"$Ae^{-\lambda t}$"),
    (enorazde_sqrt, "skyblue", r"$Ae^{-\lambda \sqrt{t}}$"),
    (enorazde_background, "green", r"$Ae^{-\lambda t}+C$"),
    (dvorazde, "blue", r"$A_1 e^{-\lambda_1 t}+A_2 e^{-\lambda_2 t}$"),
    (
        dvorazde_sqrt,
        "skyblue",
        r"$A_1 e^{-\lambda_1\sqrt{t}}+A_2 e^{-\lambda_2\sqrt{t}}$",
    ),
    (dvorazde_background, "green", r"$A_1 e^{-\lambda_1 t}+A_2 e^{-\lambda_2 t}+C$"),
]
print(f_test(functions[0][0], functions[2][0], 2, 3, len(t)))
print(f_test(functions[3][0], functions[5][0], 4, 5, len(t)))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["t[čas]"], df["N[sunkov na detektorju]"])
for fun, colour, lab in functions[:3]:
    t_sim, c, err, _ = fun()
    y1 = c - err
    y2 = c + err
    ax.fill_between(t_sim, y1, y2, color="r", alpha=0.3)
    ax.plot(t_sim, c, color=colour, label=lab)
ax.grid()
plt.legend()
plt.savefig("images/enorazdelcni.pdf")
plt.show()
