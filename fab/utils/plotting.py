import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import matplotlib as mpl
import itertools

def plot_history(history):
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3*len(history.keys())))
    for i, key in enumerate(history):
        if len(history) == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.plot(history[key])
        ax.set_title(key)
    plt.tight_layout()

def plot_3D(x, z, n, ax, title=None):
    x1 = x[:, 0].reshape(n, n)
    x2 = x[:, 1].reshape(n, n)
    z = z.reshape(n, n)
    offset = -np.abs(z).max() * 4
    trisurf = ax.plot_trisurf(x1.flatten(), x2.flatten(), z.flatten(), cmap=mpl.cm.jet)
    cs = ax.contour(x1, x2, z, offset=offset, cmap=mpl.cm.jet, stride=0.5, linewidths=0.5)
    ax.set_zlim(offset, z.max())
    if title is not None:
        ax.set_title(title)

def plot_contours(log_prob_func, bound=3):
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3D(x_points, np.exp(log_probs), n_points, ax, title="log p(x)")
    plt.show()

def plot_contours_2D(log_prob_func, bound=3):
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    plt.contour(x1, x2, z, levels=20)