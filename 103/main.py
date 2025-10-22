import numpy as np
import time
import scipy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def visualize_charges_on_sphere(spheres, points, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    for offset, r, N in spheres:
        u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
        x = r * np.cos(u) * np.sin(v) + offset[0]
        y = r * np.sin(u) * np.sin(v) + offset[1]
        z = r * np.cos(v) + offset[2]
        ax.plot_wireframe(x, y, z, color="b", alpha=0.2)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r", s=50)
    ax.set_box_aspect([1, 1, 1])

    fig.patch.set_alpha(0)
    all_coords = []
    for offset, r, N in spheres:
        all_coords.append(offset[0] + np.array([-r, r]))
        all_coords.append(offset[1] + np.array([-r, r]))
        all_coords.append(offset[2] + np.array([-r, r]))
    all_coords = np.concatenate(all_coords)
    min_lim, max_lim = np.min(all_coords), np.max(all_coords)
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_zlim(min_lim, max_lim)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    if filename is None:
        plt.show()
    else:
        fig.savefig(f'images/{filename}.pdf')


def visualize_spheres_density_2d(spheres, points, cmap=plt.cm.viridis, sigma=0.2, filename=None):

    points = points.reshape(-1, 3)

    for i, (offset, r, N) in enumerate(spheres):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        sphere_points = points[:N]
        points = points[N:]

        rel_points = sphere_points - offset
        x, y, z = rel_points[:, 0], rel_points[:, 1], rel_points[:, 2]
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        u = np.linspace(-np.pi, np.pi, 400)
        v = np.linspace(0, np.pi, 200)
        phi_grid, theta_grid = np.meshgrid(u, v)
        mesh_points = np.vstack([theta_grid.ravel(), phi_grid.ravel()]).T

        sample_points = np.vstack([theta, phi]).T

        dists = cdist(mesh_points, sample_points)
        density = np.sum(np.exp(-(dists**2) / (2 * (sigma) ** 2)), axis=1)
        density = density.reshape(theta_grid.shape)
        density /= density.max()

        im = ax.imshow(
            density,
            extent=[-180, 180, 0, 180],
            origin="lower",
            cmap=cmap,
            aspect="auto",
        )

        fig.colorbar(im, ax=ax, label="Gostota naboja")
        fig.savefig(f'images/{filename}{i}.pdf')
    plt.show()


def energy(x, N):
    pts = x.reshape(N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(N, 1)
    E = np.sum(1.0 / dist[i, j])
    return E


def random_points_on_sphere(center, radius, n):
    vecs = np.random.normal(size=(n, 3))
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    return center + radius * vecs


spheres = [
    (np.array([3, 0, 0]), 1, 10),
    (np.array([3, 3, 0]), 1, 20),
    (np.array([0, 3, 0]), 1, 15),
]
# spheres = [(np.array([0, 0, 0]), 1, 50)]
N = np.sum([n for _, _, n in spheres])
x0 = np.vstack([random_points_on_sphere(c, r, n) for c, r, n in spheres]).flatten()

energy_N = lambda x: energy(x, N)

constraints = []
i = 0
for offset, r, n in spheres:
    for _ in range(n):
        cons = {
            "type": "eq",
            "fun": lambda x, ind=i, radius=r, c=offset: (
                vec := x[ind * 3 : (ind + 1) * 3] - c,
                np.dot(vec, vec) - radius**2,
            )[1],
        }
        constraints.append(cons)
        i += 1


t = time.time()
res = scipy.optimize.minimize(energy_N, x0=x0, constraints=constraints, options={"maxiter": 1000})
print(res)
print(time.time() - t)
filename = f'spheres{len(spheres)}neq'
# filename = f'sphere{N}charges'
visualize_charges_on_sphere(spheres, res.x.reshape(N, 3), filename=filename)
visualize_spheres_density_2d(spheres, res.x, filename=filename)
