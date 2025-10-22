import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import time


sqrt2 = np.sqrt(2)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def helix_to_cartesian(points=None, t=None, phi=None):
    if points is not None:
        points = points.reshape(-1, 2)
        t = points[:, 0]
        phi = points[:, 1]

    t = t % (4 * np.pi)
    phi = phi % (2 * np.pi)

    R = 0.1

    x = np.cos(t) + R * np.cos(phi) * np.cos(t) - R / sqrt2 * np.sin(phi) * np.sin(t)
    y = np.sin(t) + R * np.cos(phi) * np.sin(t) + R / sqrt2 * np.sin(phi) * np.cos(t)
    z = t - R / sqrt2 * np.sin(phi)

    return x, y, z


def two_helixes_to_cartesian(points=None, t=None, phi=None):
    if points is not None:
        points = points.reshape(-1, 2)
        t = points[:, 0]
        phi = points[:, 1]
    max_t = 12 * np.pi
    t = t % (2 * max_t)
    phi = phi % (2 * np.pi)

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)

    R = 0.1
    smaller_mask = t < max_t
    larger_mask = ~smaller_mask

    # first helix
    t0 = t[smaller_mask]
    phi0 = phi[smaller_mask]
    x[smaller_mask] = np.cos(t0) + R * (
        np.cos(phi0) * np.cos(t0) - np.sin(phi0) * np.sin(t0)
    )
    y[smaller_mask] = np.sin(t0) + R * (
        np.cos(phi0) * np.sin(t0) + np.sin(phi0) * np.cos(t0)
    )
    z[smaller_mask] = t0 + R * np.sin(phi0)

    # second helix
    phase = np.pi
    t1 = t[larger_mask] - max_t
    phi1 = phi[larger_mask]
    r1 = 0.5
    x[larger_mask] = (
        r1 + np.cos(t1 + phase) + R * (np.cos(phi1) * np.cos(t1 + phase) - 1 / sqrt2 * np.sin(phi1) * np.sin(t1 + phase))
    )
    y[larger_mask] = (
        r1 + np.sin(t1 + phase) + R * (np.cos(phi1) * np.sin(t1 + phase) + 1 / sqrt2 * np.sin(phi1) * np.cos(t1 + phase))
    )
    z[larger_mask] = t1 - R / sqrt2 * np.sin(phi1)

    return x, y, z


def visualize(points, M=1, name=None):
    helix = helix_to_cartesian if M == 1 else two_helixes_to_cartesian
    points = helix(points)
    max_t = 12 * np.pi if M == 2 else 4 * np.pi
    t = np.linspace(0, max_t, 500, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    t0, phi0 = np.meshgrid(t, phi)

    x0, y0, z0 = helix(t=t0, phi=phi0)

    if M == 2:
        t = np.linspace(max_t, 2 * max_t, 500, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        t1, phi1 = np.meshgrid(t, phi)

        x1, y1, z1 = helix(t=t1, phi=phi1)
    ls = LightSource(azdeg=45, altdeg=45)

    rgb0 = ls.shade(
        z0,
        cmap=plt.cm.viridis,
        vert_exag=1,
        blend_mode="soft",
        dx=x0[1, 0] - x0[0, 0],
        dy=y0[0, 1] - y0[0, 0],
    )
    if M == 2:
        rgb1 = ls.shade(
            z1,
            cmap=plt.cm.viridis,
            vert_exag=1,
            blend_mode="soft",
            dx=x1[1, 0] - x1[0, 0],
            dy=y1[0, 1] - y1[0, 0],
        )

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[0], points[1], points[2], color="r", s=50)
    ax.plot_surface(x0, y0, z0, facecolors=rgb0, linewidth=0, antialiased=True, shade=False)
    if M == 2:
        ax.plot_surface(x1, y1, z1, facecolors=rgb1, linewidth=0, antialiased=True, shade=False)
    ax.set_box_aspect([1, 1, 2])
    ax.set_axis_off()
    if name is not None:
        fig.savefig(f'{name}.pdf')
    else:
        plt.show()


def energy(x, N, helix, infinity):
    t = x.reshape(-1, 2)[:, 0]
    if infinity and np.any((4 * np.pi < t) | (t < 0)):
        return 1e20
    pts = np.zeros((N, 3))
    pts[:, 0], pts[:, 1], pts[:, 2] = helix(x)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(N, 1)
    E = np.sum(1.0 / dist[i, j])

    return E


def diff():
    Ns = list(range(30))
    diffs = []
    its = []
    for N in Ns:
        diff = 0
        it = 0
        for i in range(5):
            x0 = np.random.rand(2 * N)
            M = 1
            helix = helix_to_cartesian if M == 1 else two_helixes_to_cartesian
            energy_N = lambda x: energy(x, N, helix, infinity=True)
            res0 = scipy.optimize.minimize(
                energy_N, x0=x0, options={"maxiter": 100000}, method="Powell"
            )
            # print(res0)
            # print(time.time() - t)

            energy_N = lambda x: energy(x, N, helix, infinity=False)
            res1 = scipy.optimize.minimize(
                energy_N, x0=x0, options={"maxiter": 100000}, method="Powell"
            )
            # print(res1)
            # print(time.time() - t)
            diff += np.sum(np.abs(res1.x - res0.x))
            it += res0.nit - res1.nit
        diffs.append(diff / 5)
        its.append(it / 5)

    fig, ax = plt.subplots()
    ax.plot(Ns, its)
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$nit_{infty} - nit_{cycle}$")
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig("images/iterations.pdf")
    fig, ax = plt.subplots()
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\sum_i \|x_{infty} - x_{cycle}\|$")
    ax.plot(Ns, diffs)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig("images/diffs.pdf")
    plt.show()


# diff()

N = 30
x0 = np.random.rand(2 * N)
M = 1
helix = helix_to_cartesian if M == 1 else two_helixes_to_cartesian
energy_N = lambda x: energy(x, N, helix, infinity=False)
t = time.time()
res = scipy.optimize.minimize(
    energy_N, x0=x0, options={"maxiter": 100000}, method="Nelder-Mead"
)
print(time.time() - t)
print(res)

visualize(res.x, M=M, name="nelder-mead-single")
plt.show()
