from typing import Optional, Tuple
import chex
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import matplotlib as mpl
import itertools
import pandas as pd


def plot_history(history):
    """
    Rudimentary plotting for history where we plot everything given.
    """
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3*len(history.keys())))
    for i, key in enumerate(history):
        if len(history) == 1:
            ax = axs
        else:
            ax = axs[i]
        data = np.asarray(history[key]).squeeze()
        if len(data.shape) == 3:
            split_axis = 1
            data_split = np.split(data, indices_or_sections=data.shape[1],
                                  axis=split_axis)
            for i, data_chunk in enumerate(data_split):
                ax.plot(data_chunk.squeeze(split_axis), alpha=0.4)
                ax.set_title(key + f"axis1_element{i}")
        else:
            assert len(data.shape) < 3
            ax.plot(data)
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

def plot_contours_3D(log_prob_func, bound=3):
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

def plot_contours_2D(log_prob_func,
                     ax: Optional[plt.Axes] = None,
                     bound=3, levels=20):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels)


def plot_marginal_pair(samples: chex.Array,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[int, int] = (-5, 5),):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=0.5)